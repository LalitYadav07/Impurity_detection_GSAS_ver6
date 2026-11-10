"""Forward-only magnetic construction using the shared audited native engine."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def execute(operation, request, output):
    from gpx_recipe.snapshot import gsas, json_safe
    G = gsas()
    from gpx_recipe.live.gpx_index import install_fresh_gpx_index
    install_fresh_gpx_index()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'symmetry/adapters'))
    from gpx_recipe.symmetry.native_audit import inspect_source
    from gpx_recipe.live.validated_backend import corrected_backend
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    source = Path(request['source_gpx']).resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != request['source_gpx_sha256']:
        raise ValueError('Planning parent changed after preparation')
    reference = request.get('nuclear_reference')
    reference_multiplier = 1.
    if reference is not None:
        if (not isinstance(reference, dict) or
            reference.get('schema') != 'powgen-planning-nuclear-reference/v1' or
            reference.get('source_gpx_sha256') != digest or
            not isinstance(reference.get('structural_state'), dict)):
            raise ValueError('The structural nuclear reference does not match this parent')
        reference_multiplier = reference.get('nuclear_cell_multiplier')
        if (type(reference_multiplier) not in (int, float) or
            not np.isfinite(reference_multiplier) or not 1/64 <= reference_multiplier <= 64):
            raise ValueError('The structural reference requires a finite cell multiplier between 1/64 and 64')
    if operation == 'context':
        import importlib
        elements=importlib.import_module((G.__package__+'.' if G.__package__ else '')+'GSASIIElem')
        context=inspect_source(source,include_runtime=False)
        context['magnetic_form_factors']={atom['label']:[row['Symbol'].capitalize()
            for row in elements.GetMagFormFacCoeff(elements.getElSym(atom['type']))]
            for phase in context['phases'] if phase['type']=='nuclear' for atom in phase['atoms']}
        return context
    if operation == 'export-parent':
        project=G.G2Project(str(source))
        parents=[p for p in project.phases() if p.data['General']['Type']=='nuclear']
        if len(parents)!=1:raise ValueError('Select one nuclear parent')
        path=output/'parent.cif';parents[0].export_CIF(str(path))
        return {'cif':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if operation == 'structural-match':
        from gpx_recipe.symmetry.isodistort_native import match_k
        return match_k(request['cif_text'],request['k'],request['options'])
    if operation == 'structural-modes':
        from gpx_recipe.symmetry.isodistort_native import modes
        return modes(request['child_cif'],output/'mode-inventory.gpx')
    if operation == 'structural-prepare':
        import shutil
        from gpx_recipe.symmetry.native_structural import run
        from gpx_recipe.symmetry.worker import Artifacts
        candidate=dict(request['candidate'])
        child=Path(request['child_cif']).resolve(strict=True)
        if hashlib.sha256(child.read_bytes()).hexdigest()!=candidate['child']['sha256']:
            raise ValueError('Constructed structural CIF changed after discovery')
        shutil.copyfile(child,output/'child.cif');candidate['child']={**candidate['child'],'path':'child.cif'}
        context=inspect_source(source,include_runtime=False);parent=next(p for p in context['phases'] if p['type']=='nuclear')
        if parent['setting_sha256']!=request['parent_setting_sha256']:
            raise ValueError('Structural child belongs to a different parent setting')
        values=request['mode_values']
        if set(values)!=set(candidate['modes']) or any(type(v) not in (int,float) or not np.isfinite(v) or abs(v)>2 for v in values.values()):
            raise ValueError('Supply every discovered structural mode with a finite amplitude within ±2 Å')
        protocol={'parent_phase':parent['name'],'histogram':context['histograms'][0]['name'],
                  'parent_setting_sha256':parent['setting_sha256'],'policy':{'maximum_cell_multiplier':64}}
        with corrected_backend():
            result=run(source,Artifacts(output,input_root=output),protocol,candidate,
                       {'mode_values':values,'cell':request.get('cell')},forward_only=True)
        path=output/result['gpx']
        multiplier = reference_multiplier*float(abs(np.linalg.det(candidate['transformation_matrix'])))
        if not 1/64 <= multiplier <= 64:
            raise ValueError('The composed structural cell multiplier exceeds the supported reference range')
        from .transition_controls import structural_controls
        controls = structural_controls(G, path)
        return {'schema':'powgen-planning-structural-state/v1','gpx':str(path),
                'gpx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'no_refinement':True,
                'modes':values,'cell':result['cell'],'zero_control_error':result['zero_control_error'],
                'controls':controls,
                'nuclear_cell_multiplier':multiplier,
                **({'structural_parent':reference} if reference is not None else {}),
                'parent_setting_sha256':parent['setting_sha256'],'candidate':request['candidate'],'k':request['k']}
    if operation != 'prepare': raise ValueError('Unknown magnetic planning operation')
    with corrected_backend():
        from gpx_recipe.symmetry.native_magnetic import prepare
        from gpx_recipe.symmetry.worker import Artifacts
        from gpx_recipe.live.native_profile import NativeReducedProfile
        from gpx_recipe.live.gpx_export import apply_magnetic_parameters
        from gpx_recipe.native_forward import forward_profile
        context = inspect_source(source,include_runtime=False)
        parents = [p for p in context['phases'] if p['type'] == 'nuclear']
        if len(parents) != 1 or len(context['phases']) != 1:
            raise ValueError('Prepare the nuclear-only parent before discovering magnetic subgroups')
        parent = parents[0]
        if parent['setting_sha256'] != request['parent_setting_sha256']:
            raise ValueError('Magnetic catalog belongs to a different parent setting')
        from .transition_controls import magnetic_sites_with_defaults, magnetic_controls
        sites = magnetic_sites_with_defaults(G, source, request['sites'])
        protocol = {'parent_phase': parent['name'], 'histogram': context['histograms'][0]['name'],
            'sites': sites, 'k': request['k'], 'source': {'sha256': digest},
            'parent_setting_sha256': parent['setting_sha256'],
            'policy': {'moment_model': 'symmetry_allowed', 'magnetic_optimizer': 'profiled_quadratic',
                       'refine_histogram_scale': False, 'refine_background': False, 'cycles': 0,
                       'maximum_moment': 30., 'maximum_cell_multiplier': 64,
                       'site_policy': request.get('site_policy', 'all'), 'magnetic_d_min': 0.}}
        seed = {'id': 'planning', 'magnitude': 1., 'axis': 'x', 'parent_direction': None, 'site_signs': {}}
        result = prepare(source, Artifacts(output, input_root=output), protocol, request['candidate'], seed)
        path = output / result['gpx']
        native = NativeReducedProfile(path)
        parameters = [{'name': name, 'value': float(value)} for name,value in zip(native.vary, native.x0)]
        if not parameters or any('::AM' not in row['name'] and not row['name'].startswith('::constr') for row in parameters):
            raise ValueError('The prepared independent parameter set contains nonmagnetic variables')
        controls = magnetic_controls(native, result['seed_template'])
        supplied = request.get('parameters')
        if supplied is not None:
            if not isinstance(supplied,dict) or set(supplied) != set(native.vary):
                raise ValueError('Supply every independent magnetic parameter from this exact prepared model')
            if any(type(v) not in (int,float) for v in supplied.values()):
                raise ValueError('Magnetic coordinates must be numeric')
            values = np.asarray([supplied[name] for name in native.vary], float)
            if not np.isfinite(values).all() or np.any(abs(values) > 30):
                raise ValueError('Magnetic coordinates must be finite and within ±30')
            project = G.G2Project(str(path))
            expected = native.profile(values).copy()
            mapped = apply_magnetic_parameters(project, native, values)
            project.save()
            actual, _ = forward_profile(path, protocol['histogram'])
            if actual.shape != expected.shape or not np.allclose(actual,expected,rtol=1e-8,atol=1e-8):
                raise ValueError('Saved magnetic model disagrees with its independent-coordinate profile')
            parameters = [{'name': name, 'value': float(value)} for name,value in zip(native.vary, values)]
        else:
            project = G.G2Project(str(path)); mapped = []
            for phase in project.phases():
                if phase.data['General']['Type'] != 'magnetic': continue
                cx,ct,_,_ = phase.data['General']['AtomPtrs']
                mapped.extend({'label':a[ct-1],'components':list(map(float,a[cx+4:cx+7]))} for a in phase.data['Atoms'])
        return json_safe({'schema':'powgen-planning-magnetic-state/v1','gpx':str(path),
            'gpx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'parameters':parameters,
            'site_moments':mapped, 'coordinate_convention':controls['convention'],
            'controls':controls, 'magnetic_ions':sites,
            'parent_setting_sha256':parent['setting_sha256'],'source_gpx_sha256':digest,
            'nuclear_cell_multiplier':reference_multiplier,
            **({'structural_parent':reference} if reference is not None else {}),
            'candidate':request['candidate'],'k':request['k'],'no_refinement':True})


if __name__ == '__main__':
    operation, request_path, output_path = sys.argv[1:]
    output = Path(output_path); output.mkdir(parents=True,exist_ok=True)
    try:
        result = execute(operation,json.loads(Path(request_path).read_text()),output)
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'message':str(exc),'error_type':type(exc).__name__})+'\n')
        raise
