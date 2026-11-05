"""Reconstruct prepared-search bounds from sealed inputs, not output claims."""
from pathlib import Path
import tempfile
import numpy as np

from .compiler import confined,file_sha
from .contracts import policy,structural_cell_budget
from .worker import TrialDisposition


def check_prepared(source,output,protocol,candidate,project,*,input_root=None,reference_source=None):
    from ..structural_live.native import prepare,modules
    from ..structural_live.optimization.refinable_gpx import configure
    from ..structural_live.optimization.physical_prior_geometry import displacement_gram,native_reciprocal_matrix
    from ..structural_live.optimization.chemical_geometry import direct_metric_from_reciprocal
    from ..live.native_profile import NativeReducedProfile
    try:
        scope=policy(protocol['policy'])
        structural_cell_budget(candidate,scope)
    except ValueError as exc:raise TrialDisposition('effect_mismatch',str(exc)) from exc
    root=Path(input_root) if input_root is not None else Path(source).parent
    cif=confined(root,candidate['child']['path'])
    if file_sha(cif)!=candidate['child']['sha256']:
        raise TrialDisposition('integrity_failure','Authoritative structural CIF changed before audit')
    child=project.phase(protocol['parent_phase']);hist=project.histogram(protocol['histogram'])
    _,_,lattice,spc,*_=modules()
    # This rebuild does not construct an optimization/start bank. It derives
    # the physical basis, parent sites and native active set independently.
    with tempfile.TemporaryDirectory(prefix='prepared-audit-',dir=Path(output).parent) as directory:
        directory=Path(directory)
        metadata=prepare(source,cif,candidate['transformation_matrix'],candidate['origin_shift'],directory/'geometry',
            phase_name=protocol['parent_phase'],histogram_name=protocol['histogram'],reference_gpx=reference_source)
        G,*_=modules();expected=G.G2Project(gpxfile=str(directory/'geometry'/'prepared.gpx'))
        zero=expected.phase(metadata['phase']);eh=expected.histogram(protocol['histogram'])
        with np.load(directory/'geometry'/'basis.npz',allow_pickle=False) as data:
            basis={k:data[k].copy() for k in data.files}
        cx,ct,_,_=zero.data['General']['AtomPtrs']
        labels=[a[ct-1] for a in zero.data['Atoms']]
        actual={a[ct-1]:a for a in child.data['Atoms']}
        if len(actual)!=len(labels) or set(actual)!=set(labels):
            raise TrialDisposition('effect_mismatch','The output child-site identities changed')
        delta=np.asarray([actual[label][cx:cx+3] for label in labels])-np.asarray([a[cx:cx+3] for a in zero.data['Atoms']])
        # Native exports retain the anchored, unwrapped coordinates. Wrapping
        # each atom independently can change a valid multi-mode displacement
        # into a different point in the declared physical-coordinate box.
        mapping=basis['coordinate_basis'];matrix=mapping.reshape(delta.size,mapping.shape[-1])
        amplitudes=np.linalg.lstsq(matrix,delta.ravel(),rcond=None)[0] if matrix.shape[1] else np.empty(0)
        represented=(matrix@amplitudes).reshape(delta.shape)
        physical=mapping@basis['physical_basis']
        weights=np.array([spc.SytSym(a[cx:cx+3],zero.data['General']['SGData'])[1]*a[cx+3] for a in zero.data['Atoms']])
        metric=direct_metric_from_reciprocal(native_reciprocal_matrix(basis['baseline_a']))
        gram=displacement_gram(physical,metric,weights)
        coordinates=(basis['physical_basis'].T@amplitudes)*np.sqrt(np.diag(gram))
        a0=basis['baseline_a'];change=(np.asarray(lattice.cell2A(child.data['General']['Cell'][1:7]))-a0)/np.mean(a0[:3])
        cells=np.linalg.lstsq(basis['cell_basis'],change,rcond=None)[0]
        contract=configure(expected,phase_name=zero.name,histogram_name=eh.name)
        # configure() rebinds native process-global maps; reopen output last.
        varied=list(NativeReducedProfile(str(output),refresh_cell_reflections=True).vary)
        source_scale=eh.data['Sample Parameters']['Scale'][0];scale=hist.data['Sample Parameters']['Scale'][0]
        ratio=scale/source_scale if source_scale>0 else float('nan')
        background=eh.data['Background'][0][3];background_scale=max(abs(background),1.)
        final_background=hist.data['Background'][0][3]
        from gsas_magnetic_preflight import setting_fingerprint
        from .native_audit import same
        checks=dict(
            phase_inventory_preserved={(p.name,p.data['General']['Type']) for p in project.phases()}=={(p.name,p.data['General']['Type']) for p in expected.phases()},
            nuclear_nonmodulated_child=child.data['General']['Type']=='nuclear' and not child.data['General'].get('Modulated'),
            authoritative_child_symmetry_preserved=setting_fingerprint(child.data['General']['SGData'])==setting_fingerprint(zero.data['General']['SGData']),
            child_histogram_terms_fixed=same(child.data['Histograms'][hist.name],zero.data['Histograms'][eh.name]),
            child_isotope_selection_fixed=same(child.data['General'].get('Isotope'),zero.data['General'].get('Isotope')),
            child_species_fixed=all(actual[label][ct]==atom[ct] for label,atom in zip(labels,zero.data['Atoms'])),
            structural_active_set_closed=set(varied)==set(contract['native_active_variables']) and len(varied)==len(set(varied)),
            physical_mode_rank_preserved=len(coordinates)==metadata['inventory']['physical_rank'],
            coordinates_in_authoritative_mode_space=bool(np.allclose(delta,represented,rtol=0,atol=1e-8)),
            physical_modes_in_reviewed_bounds=bool(np.isfinite(coordinates).all() and np.all(abs(coordinates)<=scope['structural_mode_radius']+1e-9)),
            cell_in_authoritative_symmetry_space=bool(np.allclose(basis['cell_basis']@cells,change,rtol=0,atol=1e-9)),
            cell_coordinates_in_reviewed_bounds=bool(np.isfinite(cells).all() and np.all(abs(cells)<=scope['structural_cell_radius']+1e-9)),
            positive_bounded_common_scale=bool(np.isfinite(ratio) and np.exp(-.3)-1e-10<=ratio<=np.exp(.3)+1e-10),
            constant_background_in_reviewed_bounds=bool(np.isfinite(final_background) and background-.75*background_scale-1e-9<=final_background<=background+3*background_scale+1e-9))
        return checks,metadata['site_mapping'],varied
