"""Model-campaign adapter to the shared prepared nonlinear structural service."""
import shutil
from ..structural_live.service import SearchSettings, fit_snapshot, prepare_model
from .compiler import confined
from .worker import TrialDisposition


def optimization_progress(result):
    """Expose incomplete local work separately from a checked starting curve."""
    fitted=result['result'];blocks=fitted.get('blocks',[]);attempts=fitted.get('attempts',[])
    progress=dict(starts_attempted=len(attempts),
        starts_time_limited=sum(a.get('reason')=='per_start_time_budget' for a in attempts),
        complete_lattice_blocks=sum(b['kind'] in {'cell','joint'} for b in blocks),
        complete_displacement_blocks=sum(b['kind'] in {'modes','joint'} for b in blocks),
        complete_joint_blocks=sum(b['kind']=='joint' for b in blocks),
        accepted_local_steps=sum(len(b.get('accepted_costs',[])) for b in blocks))
    if not blocks:
        message='Prepared starting curves were checked, but no complete local-update block finished. Increase the per-model allocation before comparing structures.'
    else:
        message=(f"{progress['starts_attempted']} starts attempted; {progress['starts_time_limited']} reached their allocation; "
            f"{progress['complete_lattice_blocks']} lattice and {progress['complete_displacement_blocks']} displacement blocks completed. "
            'Optimization and structure assignment require review.')
        if progress['complete_joint_blocks']:
            message+=f" {progress['complete_joint_blocks']} blocks updated lattice and displacement coordinates together."
    return progress,message


def run(source_gpx,ctx,protocol,candidate,seed):
    scope=protocol['policy']
    from .contracts import structural_cell_budget,policy
    try:
        structural_cell_budget(candidate,scope)
        policy(scope)
    except ValueError as exc:raise TrialDisposition('unsupported',str(exc)) from exc
    root=ctx.run_dir/'structural'
    child=confined(ctx.input_root,candidate['child']['path'])
    settings=SearchSettings(mode_radius=scope['structural_mode_radius'],cell_radius=scope['structural_cell_radius'],
        seconds=scope['structural_seconds'],starts=scope['structural_starts'],recovery_seconds=scope['structural_recovery_seconds'])
    from ..preparation.structural_hints import candidates as prepared_candidates
    prepared_banks,preparation=prepared_candidates(ctx.input_root,protocol,candidate['id'])
    metadata=prepare_model(source_gpx,child,candidate['transformation_matrix'],candidate['origin_shift'],
                     root/'prepared',phase_name=protocol['parent_phase'],histogram_name=protocol['histogram'],
                     reference_gpx=confined(ctx.input_root,protocol['structural_reference']['path']) if 'structural_reference' in protocol else None,
                     settings=settings,prepared_banks=prepared_banks)
    if set(candidate['modes'])!=set(metadata['inventory']['native_mode_names']):
        raise TrialDisposition('integrity_failure','Prepared search requires the complete discovered native mode inventory')
    result=fit_snapshot(root/'prepared',root/'fit',settings=settings)
    result['preparation_attachment']=preparation
    output=ctx.run_dir/'child-refinement.gpx'
    shutil.copyfile(result['continuation']['path'],output)
    ctx.save_json('zero-control.json',dict(relative_profile_error=metadata['zero_profile_error'],
                  site_mapping=metadata['site_mapping'],inventory=metadata['inventory']))
    ctx.save_json('structural-fit.json',result)
    ctx.save_json('structural-export.json',result['continuation'])
    progress,message=optimization_progress(result)
    # A checked prediction is not a convergence certificate. Keep it available
    # for inspection/continuation without promoting it to a qualified model.
    return dict(status='optimization_unresolved',gpx=output.name,
        reason='Independently verified structural point fit. '+message,
        rwp=result['metrics']['rwp_native'],nvars=result['physical_mode_count']+result['cell_parameter_count']+2,
        zero_control_error=metadata['zero_profile_error'],
        structural_fit=dict(all_physical_modes_retained=True,physical_modes=result['physical_mode_count'],
            origin_gauges=metadata['inventory']['gauge_rank'],covariance_available=False,
            timings=result['timings'],optimization_qualified=False,optimization_progress=progress,
            portable_preparation=result['portable_preparation'],preparation_attachment=preparation,
            initialization_execution=result['initialization_execution'],
            native_active_set_verified=result['continuation']['native_active_set_verified']),
        reviewable_gpx=True)
