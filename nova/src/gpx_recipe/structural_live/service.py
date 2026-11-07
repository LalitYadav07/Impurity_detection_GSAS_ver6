"""One structural fitting service for completed scans and reduced live updates.

Call in an owned scientific subprocess. GSAS-II constraint maps are global;
parallelism belongs across processes/candidates, never across threads here.
Prepared candidates are immutable. An observation replaces the objective;
only parameter guesses may be reused from an earlier observation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import numpy as np

from ..execution import write_json
from ..symmetry.compiler import package_signature
from ..symmetry.native_audit import profile_residual_metrics
from .native import NativeCandidate, prepare, sha
from .reduced import from_arrays, native_limits, verify_native_observation, ReducedObservation
from .defaults import DEFAULT_CELL_RADIUS, DEFAULT_FIT_SECONDS


@dataclass(frozen=True)
class SearchSettings:
    mode_radius: float = .05
    cell_radius: float = DEFAULT_CELL_RADIUS
    seconds: float = DEFAULT_FIT_SECONDS
    starts: int = 4
    recovery_seconds: float = 0.
    background_scale: float | None = None

    def __post_init__(self):
        for value, maximum in ((self.mode_radius,.5),(self.cell_radius,.02),(self.seconds,300.)):
            if type(value) not in (int,float) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError('Structural bounds and per-model allocation must be finite and positive')
        if type(self.starts) is not int or not 2 <= self.starts <= 16:
            raise ValueError('Two to sixteen starts per model are required')
        if type(self.recovery_seconds) not in (int,float) or not np.isfinite(self.recovery_seconds) or not 0 <= self.recovery_seconds <= 120:
            raise ValueError('Invalid optional structural recovery allocation')
        if self.background_scale is not None and (type(self.background_scale) not in (int,float) or not np.isfinite(self.background_scale) or self.background_scale<=0):
            raise ValueError('Background scale must be unknown or finite and positive')


def make_source(prepared, work, settings):
    from .optimization.native_fast_candidate import NativeFastCandidate
    from .optimization.full_mode_slice import FullModeSlice
    return FullModeSlice(NativeFastCandidate(prepared,work),
                         mode_radius=settings.mode_radius,cell_radius=settings.cell_radius)


def prepare_model(source_gpx, child_cif, transform, origin, output, *, settings=None, prepared_banks=(), **roles):
    """Build immutable native geometry and reusable starting curves once.

    Starting curves contain no fitted likelihood or evidence. Their identity
    includes the native implementation, instrument/grid, chemistry and bounds.
    """
    from ..preparation.structural_consume import prepare_starting_bank
    settings=settings or SearchSettings()
    if not isinstance(settings,SearchSettings): raise TypeError('Explicit SearchSettings required')
    metadata=prepare(source_gpx,child_cif,transform,origin,output,**roles)
    output=Path(output)
    source=make_source(output,output/'bank-construction',settings)
    bank,qualification=prepare_starting_bank(source,output/'starting-curves.npz',prepared_banks)
    write_json(output/'starting-curve-contract.json',dict(bank,portable_preparation=qualification))
    return metadata


def fit_snapshot(prepared, output, *, settings=None, observation=None, warm_starts=()):
    """Fit every physical mode in this candidate; independently reopen its GPX.

    This is a point-estimate service, not a posterior or stopping policy.
    Unknown exposure is accepted for reduced data without any count conversion.
    The supplied covariance is diagonal and conditional on fixed instrument,
    occupancy, ADPs, secondary phases and all higher background coefficients.
    """
    from .optimization.native_background_context import fit_table
    from .optimization.starting_proposals import load_current_bank
    from ..preparation.structural_consume import prepare_starting_bank
    from .optimization.profiled_native_fit import fit
    from .optimization.captured_cell_derivative import CapturedCellDerivative
    from .optimization.refinable_gpx import export_refinable
    settings = settings or SearchSettings()
    if not isinstance(settings,SearchSettings): raise TypeError('Explicit SearchSettings required')
    if observation is not None and not isinstance(observation,ReducedObservation):
        raise TypeError('A supplied-variance ReducedObservation is required')
    if observation is not None and observation.protocol_sha256!=sha(Path(prepared)/'prepared.json'):
        raise ValueError('Observation does not bind this prepared structural context')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();timings={}
    source=make_source(prepared,output/'context',settings)
    arrays=source.fast.slow.hist.data['data'][1]
    if observation is None:
        observation=from_arrays(arrays,source.fast.slow.hist.data['Limits'],
            acquisition=source.meta['source_gpx_sha256'] if hasattr(source,'meta') else source.fast.meta['source_gpx_sha256'],
            protocol_sha256=sha(Path(prepared)/'prepared.json'))
    if observation.protocol_sha256!=sha(Path(prepared)/'prepared.json'):
        raise ValueError('Observation does not bind this prepared structural context')
    if not np.array_equal(observation.x,source.fast.x):
        raise ValueError('Observation grid differs from this prepared native context')
    native_limits(observation,source.fast.slow.hist.data['Limits'])
    if np.any(observation.mask&np.ma.getmaskarray(arrays[0])):
        raise ValueError('Observation enables bins masked in the reviewed native grid')
    background=source.fast.slow.background0
    if not np.isfinite(background): raise ValueError('Nonfinite native background coefficient')
    background_scale=settings.background_scale if settings.background_scale is not None else max(abs(background),1.)
    table=fit_table(source,background_scale)
    # Center the positive auxiliary coefficient on the saved signed Back;0.
    # This is an additive correction, not a requirement that Back;0 be positive.
    table.fixed_background=table.renderer.background-background_scale*table.background_response
    timings['context_seconds']=time.monotonic()-started
    checkpoint=time.monotonic();bank_path=Path(prepared)/'starting-curves.npz'
    bank_reused=bank_path.is_file()
    if not bank_reused:
        bank_path=output/'starts.npz'
        prepare_starting_bank(source,bank_path)
    bank,initialization_execution=load_current_bank(source,bank_path,output)
    if initialization_execution['regenerated']:bank_reused=False
    portable_preparation=None
    contract=Path(prepared)/'starting-curve-contract.json'
    if contract.is_file():
        if contract.stat().st_size>4*1024*1024:raise ValueError('Starting-curve provenance exceeds its budget')
        import json
        portable_preparation=json.loads(contract.read_text()).get('portable_preparation')
    timings['start_bank_seconds']=time.monotonic()-checkpoint
    checkpoint=time.monotonic()
    evaluator=CapturedCellDerivative(source,max_cells=2)
    fitted,prediction=fit(source,table,observation,bank,seconds=settings.seconds,
        starts=settings.starts,warm_pool=list(warm_starts) or None,evaluator=evaluator)
    if settings.recovery_seconds:
        from .optimization.automatic_fit_recovery import improve, RecoveryPolicy
        budget=settings.recovery_seconds/3
        recovered,prediction=improve(source,table,observation,fitted['best']['parameters'],bank=bank,
            policy=RecoveryPolicy(local_seconds=budget,proposal_seconds=budget,joint_seconds=budget,
                                  starts=settings.starts),evaluator=evaluator)
        fitted=dict(fitted,best=recovered['best'],recovery=recovered,
            rwp_percent=recovered['rwp_percent'],exact_native_deviance=recovered['best']['deviance'])
    timings['optimization_seconds']=time.monotonic()-checkpoint
    z=np.asarray(fitted['best']['parameters'],float);m=source.mode_dimensions;d=len(source.radii)
    physical=source.lift@(z[:d]*source.radii)
    phase_name=source.fast.slow.phase.name;histogram_name=source.fast.slow.hist.name
    # End in-memory use before the independent saved-project calculation
    # takes ownership of native constraint maps.
    checkpoint=time.monotonic();native=NativeCandidate(prepared,output/'independent')
    if observation is not None and observation.observation_sha256 != from_arrays(native.hist.data['data'][1],native.hist.data['Limits'],
            acquisition=observation.acquisition,protocol_sha256=observation.protocol_sha256,
            revision=observation.revision,exposure=observation.exposure).observation_sha256:
        native.bind_observation(observation)
    exact=native.evaluate(physical[:m],physical[m:],log_scale=z[-2],
        background0_delta=background_scale*np.expm1(z[-1]))
    delta=(exact-prediction)[observation.mask]
    relative=float(np.linalg.norm(delta)/max(np.linalg.norm(exact[observation.mask]),1e-30))
    sigma=float(np.max(abs(delta)/np.sqrt(observation.variance[observation.mask])))
    if not np.isfinite([relative,sigma]).all() or relative>1e-8 or sigma>.01:
        raise ValueError('Independent saved-GPX prediction differs from the fitted native profile')
    frozen=native.save_verified_prediction(exact)
    continuation=export_refinable(frozen['path'],output/'continuation',expected_sha256=frozen['sha256'],
                                  phase_name=phase_name,histogram_name=histogram_name)
    # Reopen each saved artifact. Curve agreement alone cannot establish that
    # the same measurements, likelihood mask and weighted objective survived.
    from .native import modules,curve
    import shutil
    G,*_=modules();contracts={}
    for role,artifact in [('frozen',frozen),('continuation',continuation)]:
        # Native calculation saves its project. Verify on a copy so the sealed
        # published artifact and its digest remain immutable.
        verification_dir=output/'verification'/role
        verification_dir.mkdir(parents=True)
        checked=verification_dir/'project.gpx'
        if sha(artifact['path'])!=artifact['sha256']:
            raise ValueError('Saved GPX changed before independent verification')
        shutil.copyfile(artifact['path'],checked)
        project=G.G2Project(gpxfile=str(checked));histogram=project.histogram(histogram_name)
        saved_curve=curve(project,histogram)
        contracts[role]=verify_native_observation(observation,histogram.data['data'][1][:3],
            histogram.data['Limits'],saved_curve,fitted['best']['deviance'],
            weight_factor=float(histogram.data['data'][0].get('wtFactor',1.)))
        if sha(artifact['path'])!=artifact['sha256']:
            raise ValueError('Published GPX changed during independent verification')
    metrics=profile_residual_metrics(native.hist.data['data'][1][:3],exact,native.hist.data['Limits'])
    timings['independent_export_seconds']=time.monotonic()-checkpoint
    result=dict(schema='radar-structural-fit/v1',state='verified_point_fit',
        observation_sha256=observation.observation_sha256,revision=observation.revision,
        prepared_sha256=sha(Path(prepared)/'prepared.json'),implementation_sha256=package_signature(),
        settings=asdict(settings),parameters=z.tolist(),physical_parameters=physical.tolist(),
        physical_mode_count=m,cell_parameter_count=d-m,all_physical_modes_retained=True,
        result=fitted,metrics=metrics,verification=dict(relative_error=relative,max_sigma_error=sigma,saved_observations=contracts),
        starting_curves_reused=bank_reused,
        initialization_execution=initialization_execution,
        portable_preparation=portable_preparation,
        frozen=frozen,continuation=continuation,timings=timings,wall_seconds=time.monotonic()-started,
        previous_likelihood_accumulated=False,instrument_parameters_fixed=True,
        higher_background_coefficients_fixed=True,covariance_available=False,
        background_execution=table.background_execution,
        nuisance_scope=dict(common_scale_bounds=[float(np.exp(-.3)),float(np.exp(.3))],
            background0_initial=background,background_scale=background_scale,
            background0_bounds=[background-.75*background_scale,background+3*background_scale]),
        optimization_qualified=False,global_optimum_certified=False,
        model_probability=None,model_exclusion_authorized=False,counting_stop_authorized=False)
    write_json(output/'fit.json',result)
    return result
