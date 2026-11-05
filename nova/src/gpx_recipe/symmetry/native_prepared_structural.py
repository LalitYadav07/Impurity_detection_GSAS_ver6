"""Model-campaign adapter to the shared prepared nonlinear structural service."""
import shutil
from ..structural_live.service import SearchSettings, fit_snapshot, prepare_model
from .compiler import confined
from .worker import TrialDisposition


def run(source_gpx,ctx,protocol,candidate,seed):
    scope=protocol['policy']
    root=ctx.run_dir/'structural'
    child=confined(ctx.input_root,candidate['child']['path'])
    settings=SearchSettings(mode_radius=scope['structural_mode_radius'],cell_radius=scope['structural_cell_radius'],
        seconds=scope['structural_seconds'],starts=scope['structural_starts'],recovery_seconds=scope['structural_recovery_seconds'])
    metadata=prepare_model(source_gpx,child,candidate['transformation_matrix'],candidate['origin_shift'],
                     root/'prepared',phase_name=protocol['parent_phase'],histogram_name=protocol['histogram'],
                     reference_gpx=confined(ctx.input_root,protocol['structural_reference']['path']) if 'structural_reference' in protocol else None,settings=settings)
    if set(candidate['modes'])!=set(metadata['inventory']['native_mode_names']):
        raise TrialDisposition('integrity_failure','Prepared search requires the complete discovered native mode inventory')
    result=fit_snapshot(root/'prepared',root/'fit',settings=settings)
    output=ctx.run_dir/'child-refinement.gpx'
    shutil.copyfile(result['continuation']['path'],output)
    ctx.save_json('zero-control.json',dict(relative_profile_error=metadata['zero_profile_error'],
                  site_mapping=metadata['site_mapping'],inventory=metadata['inventory']))
    ctx.save_json('structural-fit.json',result)
    ctx.save_json('structural-export.json',result['continuation'])
    # A checked prediction is not a convergence certificate. Keep it available
    # for inspection/continuation without promoting it to a qualified model.
    return dict(status='optimization_unresolved',gpx=output.name,
        reason='Independently verified structural point fit; optimization and structure assignment require review.',
        rwp=result['metrics']['rwp_native'],nvars=result['physical_mode_count']+result['cell_parameter_count']+2,
        zero_control_error=metadata['zero_profile_error'],
        structural_fit=dict(all_physical_modes_retained=True,physical_modes=result['physical_mode_count'],
            origin_gauges=metadata['inventory']['gauge_rank'],covariance_available=False,
            timings=result['timings'],optimization_qualified=False,
            native_active_set_verified=result['continuation']['native_active_set_verified']),
        reviewable_gpx=True)
