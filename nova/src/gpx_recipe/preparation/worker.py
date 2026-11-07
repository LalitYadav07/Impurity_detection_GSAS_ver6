"""Network-free preparation through the same audited native scientific engines.

The public inputs are a CIF, bank profile and typed model choices. Generated
GPXs are private construction state. Portable outputs contain data only, and
none of these operations fits observations or declares a scientific solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..execution import write_json
from ..symmetry.compiler import file_sha, package_signature
from ..symmetry.contracts import digest, fields, number, validate_candidate, validate_protocol
from .reference import create_reference, validate_grid


def validate_settings(kind, value):
    """Closed model choices; credentials, executable recipes and scores do not belong here."""
    if kind == 'reference':
        value = fields(value, [], {'maximum_denominator': 8})
        value['maximum_denominator'] = number(value['maximum_denominator'], 1, 8, integer=True)
    elif kind == 'magnetic':
        value = fields(value, ['candidate', 'k', 'sites', 'maximum_moment',
            'refine_histogram_scale', 'refine_background'],
            {'maximum_cell_multiplier': 8, 'magnetic_d_min': 0.})
        value['candidate'] = validate_candidate(value['candidate'])
        if not value['candidate']['magnetic_space_group'] or value['candidate']['child']:
            raise ValueError('Magnetic preparation requires an audited MSG embedding')
        # Reuse the complete existing scientific protocol validator below,
        # after native import provides the exact parent setting and roles.
        number(value['maximum_moment'], 0, 20, positive=True)
        number(value['maximum_cell_multiplier'], 1, 8, integer=True)
        number(value['magnetic_d_min'], 0, 10)
        if any(type(value[k]) is not bool for k in ('refine_histogram_scale', 'refine_background')):
            raise ValueError('Explicit scale/background freedom is required')
    elif kind in {'structural','kernel'}:
        value = fields(value, ['candidate', 'mode_radius', 'cell_radius']+
                       (['cell_index'] if kind=='kernel' else []))
        value['candidate'] = validate_candidate(value['candidate'])
        if value['candidate']['magnetic_space_group']:
            raise ValueError('Structural preparation requires a displacement-mode child')
        number(value['mode_radius'], 0, .5, positive=True)
        number(value['cell_radius'], 0, .02, positive=True)
        if kind=='kernel':number(value['cell_index'],0,511,integer=True)
    else:
        raise ValueError('Unsupported experiment preparation operation')
    return value


def validate_request(kind, request):
    if kind=='batch':
        from .batch import validate_request as validate_batch
        return validate_batch(request)
    required = ['source', 'instrument', 'grid', 'settings', 'destination']
    if kind in {'structural','kernel'}:
        required.append('child')
    request = fields(request, required)
    request['settings'] = validate_settings(kind, request['settings'])
    request['grid'] = fields(request['grid'], ['kind', 'lower', 'upper', 'bins', 'bank'])
    request['grid'] = validate_grid(**request['grid'])
    if kind=='kernel' and request['grid']['kind']!='PNT':
        raise ValueError('Native raw-kernel preparation currently requires a TOF profile')
    return request


def verify_structural_child(settings, checksum):
    """Bind declared scientific content, independently of transport location."""
    declared = settings['candidate'].get('child')
    if declared is not None and declared['sha256'] != checksum:
        raise ValueError('Structural preparation child differs from the declared candidate')


def execute(kind, request):
    """One owned native process prepares a single bank and candidate context."""
    if kind=='batch':
        from .batch import execute as execute_batch
        return execute_batch(request)
    request = validate_request(kind, request)
    settings, grid = request['settings'], request['grid']
    started = time.monotonic()
    inputs = [Path(request[k]).resolve() for k in ('source', 'instrument')]
    if kind in {'structural','kernel'}:
        inputs.append(Path(request['child']).resolve())
    checksums = [file_sha(p) for p in inputs]
    if kind in {'structural','kernel'}:
        verify_structural_child(settings, checksums[2])
    output = Path(request['destination']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = create_reference(inputs[0], inputs[1], output/'native-reference', **grid)
    from ..symmetry.native_audit import inspect_source
    context = inspect_source(reference['path'])
    parent = context['phases'][0]
    artifacts = []
    if kind == 'reference':
        from ..symmetry.parent_k import parent_k_points
        detail = dict(parent=parent, histogram=reference['histogram'],
            points=parent_k_points(parent, maximum_denominator=settings['maximum_denominator']),
            k_scope='Finite supported symmetry k points, not every rational vector or multi-k state')
        if grid['kind']=='PNT':
            from ..live.native_profile import NativeReducedProfile
            from ..structural_live.optimization.tof_profile_table import CompiledTOFTable
            from ..structural_live.optimization.native_cell_proposals import ProposalUnavailable
            from .kernel_runtime import RuntimeUnavailable
            from .tof_profile_arrays import capture,encode,origin_identity
            try:
                native=NativeReducedProfile(reference['path'])
                table=CompiledTOFTable.from_reference(native,reference['histogram'],reference['parent_phase'],
                    tof_range=(grid['lower'],grid['upper']))
                data=capture(table);identity=origin_identity(data.metadata)
                (output/'profile.npz').write_bytes(encode(data));write_json(output/'profile.json',data.metadata)
                artifacts+=['profile.npz','profile.json']
                detail.update(physics_sha256=identity,profile_table=dict(status='prepared',
                    role='starting_proposals',physics_sha256=identity,observations_included=False,
                    native_likelihood_authority=False,numeric_bytes=table.numeric_bytes,
                    compilation_seconds=table.build_seconds))
            except (ProposalUnavailable,RuntimeUnavailable) as error:
                detail['profile_table']=dict(status='unavailable',reason=str(error)[:300],
                    native_likelihood_authority=False)
    elif kind == 'magnetic':
        from ..symmetry.native_magnetic import construct_theory
        from ..symmetry.worker import Artifacts
        from ..live.native_profile import NativeReducedProfile
        from .native_magnetic import compile_response
        from ..live.reflection_factors import ScatteringFactorCache
        protocol = validate_protocol(dict(kind='magnetic',
            source=dict(path='reference.gpx', sha256=file_sha(reference['path'])),
            catalog=dict(path='catalog.json', sha256=digest(settings['candidate'])),
            parent_phase=parent['name'], histogram=reference['histogram'],
            parent_setting_sha256=parent['setting_sha256'],
            magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],
            implementation_sha256=package_signature(), k=settings['k'], sites=settings['sites'],
            candidates=[settings['candidate']], seeds=[dict(id='preparation', axis='x',
                magnitude=min(1., settings['maximum_moment']), independent_mode_seed=17)],
            policy=dict(cycles=1, timeout_seconds=600, maximum_trials=1,
                maximum_cell_multiplier=settings['maximum_cell_multiplier'],
                maximum_moment=settings['maximum_moment'], magnetic_d_min=settings['magnetic_d_min'],
                fit_band_percent=0., maximum_mode=.5, adequate_rwp_percent=100.,
                refine_cell=False, refine_histogram_scale=settings['refine_histogram_scale'],
                refine_background=settings['refine_background'], site_policy='all',
                moment_model='symmetry_allowed')))
        native_dir = output/'native-model'
        native_dir.mkdir()
        construction = construct_theory(reference['path'], Artifacts(native_dir), protocol,
            protocol['candidates'][0], protocol['seeds'][0])
        if construction['status'] != 'theory_constructed' or construction['observations_included']:
            raise ValueError('Preparation did not construct an observation-free magnetic model')
        cache_directory = output/'scattering-private'
        response = compile_response(NativeReducedProfile(native_dir/construction['gpx']),
            factor_cache=ScatteringFactorCache(cache_directory))
        response.save(output/'response.npz')
        artifacts.append('response.npz')
        compiler = response.proof['compiler']
        scattering_proof = compiler.get('native_reflection_diagnostics') or {}
        scattering_identity = scattering_proof.get('scattering_identity')
        scattering_file = cache_directory/(scattering_identity+'.npz') if scattering_identity else None
        if scattering_file is not None and scattering_file.is_file():
            scattering_file.replace(output/'scattering.npz')
            artifacts.append('scattering.npz')
            scattering = dict(status='prepared', identity=scattering_identity,
                artifact='scattering.npz', native_qualification_required_on_import=True,
                scope='Unweighted native amplitudes; actual geometry dependencies are checked before reuse')
        else:
            scattering = dict(status='unavailable', reason=compiler.get('fallback_reason')
                or 'The native preparation produced no portable scattering block')
        detail = dict(physics_sha256=response.physics_sha256,
            dimensions=response.quadratic.shape[1], bins=len(response.x), proof=response.proof,
            scattering=scattering,
            operator_audit=construction['audit'], periodicity=construction['periodicity'],
            geometry_scope='Exact reference geometry; current-scan rendering/qualification required')
    else:
        from ..structural_live.native import prepare
        from ..structural_live.service import make_source, SearchSettings
        from ..structural_live.optimization.offline_native_start_bank import build
        candidate = settings['candidate']
        metadata = prepare(reference['path'], inputs[2], candidate['transformation_matrix'],
            candidate['origin_shift'], output/'native-model')
        source = make_source(output/'native-model', output/'native-evaluation',
            SearchSettings(mode_radius=settings['mode_radius'], cell_radius=settings['cell_radius']))
        if kind=='kernel':
            from .kernel_geometry import specification
            from .kernel_runtime import fingerprint,origin_identity
            from .native_tof_kernels import capture
            from .tof_kernel_arrays import encode
            from ..structural_live.optimization.offline_native_start_bank import policy_points
            points=policy_points(source,'joint_cell_v1')*source.radii
            spec,representatives,mapping=specification(source,points,
                model_sha256=checksums[2],bank=f"PNT profile bank {grid['bank']}")
            index=settings['cell_index']
            if index>=len(representatives):raise ValueError('Kernel cell index is outside the complete preparation portfolio')
            kernel=capture(source,representatives[index]);runtime,runtime_sha=fingerprint(source)
            (output/'kernel.npz').write_bytes(encode(kernel.data))
            write_json(output/'kernel.json',kernel.data.metadata)
            artifacts+=['kernel.npz','kernel.json']
            detail=dict(physics_sha256=origin_identity(kernel.data.metadata['arrays_sha256'],runtime_sha),
                arrays_sha256=kernel.data.metadata['arrays_sha256'],kernel_runtime=runtime,
                kernel_runtime_sha256=runtime_sha,native_kernel_proof=kernel.diagnostics(),
                specification=spec,cell_index=index,proposal_to_cell=mapping,
                scope='One native profile component, not a scientific model or a fitted result')
        else:
            bank = build(source, output/'starts.npz', policy='joint_cell_v1', reuse_fixed_cell_profiles=True)
            artifacts += ['starts.npz', 'starts.json']
            detail = dict(physics_sha256=bank['context']['physics_sha256'],
                mode_dimensions=source.mode_dimensions, cell_dimensions=source.cell_dimensions,
                starting_curves=bank['native_curves'], proposed_starts=bank['proposed_points'],
                zero_profile_error=metadata['zero_profile_error'],
                scope='Native nonlinear starting curves and proposals; optimization and new-cell adaptation remain online')
    if checksums != [file_sha(p) for p in inputs]:
        raise ValueError('A preparation input changed during the native operation')
    manifest = dict(schema='radar-experiment-preparation-entry/v1', kind=kind,
        grid=grid, settings=settings, input_sha256=checksums,
        implementation_sha256=package_signature(), native_runtime=context['runtime'],
        detail=detail, observations_included=False, scientific_result=False,
        artifacts={name: file_sha(output/name) for name in artifacts})
    write_json(output/'entry.json', manifest)
    return dict(schema='radar-experiment-preparation-worker/v1', status='prepared',
        kind=kind, observations_included=False, scientific_result=False,
        output=str(output), artifacts=artifacts+['entry.json'],
        entry_sha256=file_sha(output/'entry.json'), seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['reference', 'magnetic', 'structural', 'kernel', 'batch'])
    parser.add_argument('request')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.kind=='batch':
        import threading
        from ..process import cancellation_signals
        from .batch import execute as execute_batch
        cancel=threading.Event()
        with cancellation_signals(cancel):
            result=execute_batch(json.loads(Path(args.request).read_text()),cancel=cancel)
        write_json(args.output,result)
    else:
        write_json(args.output, execute(args.kind, json.loads(Path(args.request).read_text())))


if __name__ == '__main__':
    main()
