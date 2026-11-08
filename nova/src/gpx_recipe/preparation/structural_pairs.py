"""Native paired-pattern pilot; separate from admitted portable ranker data.

Generate child curves first and native parent references second. GSAS-II's
process-global constraint maps must never be shared between these adapters.
This module supplies measured controls, not a trained or qualified classifier.
"""
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..symmetry.compiler import file_sha, package_signature
from ..structural_live.native import prepare
from ..structural_live.service import SearchSettings, make_source
from .reference import create_reference
from .structural_examples import declared_group, render_native
from .structural_sampling import from_source
from .structural_conditioning import ParentConditioner,condition
from .structural_parent_reference import NativeParentReference, ParentMetricSpace, rotations
from ..structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
from .worker import validate_request, verify_structural_child


def generate(request, *, seed, parent_seed, trajectories=64, steps=4,
             parent_relative_range=(.0001, .001), trajectory_slice=None, report_size=None):
    """Generate complete physical child/parent pairs, with null controls.

    The parent seed is shared across every child of the same parent/bank.
    Each amplitude step and null control retains its whole trajectory identity.
    A failed physical check fails the block; it cannot silently drop a family.
    """
    from GSASII import GSASIIscriptable as G, GSASIIlattice as L
    request = validate_request('structural', request)
    if type(trajectories) is not int or not 1 <= trajectories <= 4096:
        raise ValueError('A bounded trajectory count is required')
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError('A bounded amplitude-step count is required')
    if trajectories*steps>32768:
        raise ValueError('Generate the paired trajectories in a smaller complete portfolio')
    interval = (0, trajectories) if trajectory_slice is None else trajectory_slice
    if (not isinstance(interval, (tuple, list)) or len(interval) != 2
            or any(type(i) is not int for i in interval)
            or not 0 <= interval[0] < interval[1] <= trajectories):
        raise ValueError('A nonempty whole-trajectory slice is required')
    inputs = [Path(request[k]).resolve() for k in ('source', 'instrument', 'child')]
    hashes = [file_sha(p) for p in inputs]
    settings = request['settings']; candidate = settings['candidate']
    verify_structural_child(settings, hashes[2])
    out = Path(request['destination']).resolve(); out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic(); denominator = 1+(interval[1]-interval[0])*(steps+1)
    stages={};stage_began=began

    def mark(name):
        nonlocal stage_began
        now=time.monotonic();stages[name]=now-stage_began;stage_began=now

    def progress(state, processed):
        write_json(out/'generation-progress.json', dict(state=state, processed=processed,
            denominator=denominator, elapsed_seconds=time.monotonic()-began,
            observations_included=False))

    progress('preparing_native_context', 0)
    reference = create_reference(inputs[0], inputs[1], out/'reference', **request['grid'])
    mark('preparing_reference')
    phase = G.G2Project(gpxfile=reference['path']).phase(reference['parent_phase'])
    parent_space = ParentMetricSpace(native_reciprocal_matrix(L.cell2A(phase.data['General']['Cell'][1:7])),
        rotations(phase.data['General']['SGData']))
    # Construct these from the identical parsed parent, never the child's
    # slightly rounded back-transformed cell or candidate-dependent basis.
    parents = parent_space.draw(seed=parent_seed, count=trajectories, relative_range=parent_relative_range)
    mark('sampling_parent_cells')
    prepare(reference['path'], inputs[2], candidate['transformation_matrix'], candidate['origin_shift'], out/'prepared')
    source = make_source(out/'prepared', out/'context', SearchSettings(
        mode_radius=settings['mode_radius'], cell_radius=settings['cell_radius']))
    mark('preparing_child')
    child_group = declared_group(source)
    conditioning = ParentConditioner(source,parent_space,candidate['transformation_matrix'],parents)
    original = from_source(source, seed=seed, trajectories=trajectories, steps=steps,
        trajectory_feasible=conditioning.feasible)
    mark('sampling_child_trajectories')
    paired = condition(source, original, parent_space, candidate['transformation_matrix'], parents)
    selected = (paired.trajectory == -1) | ((paired.trajectory >= interval[0]) & (paired.trajectory < interval[1]))
    points = paired.points[selected]; parent_a = paired.parent_a[selected]
    role = paired.role[selected]; trajectory = paired.trajectory[selected]
    if len(points)*len(source.fast.x)*8*2 > 128*1024**2:
        raise ValueError('Generate smaller paired blocks within the 128 MiB curve-array limit')
    from .structural_paired_blocks import MAX_BYTES
    numeric_bytes=(3*len(points)*len(source.fast.x)*8+2*len(source.fast.x)*8
        +points.nbytes+parent_a.nbytes+trajectory.nbytes+role.nbytes+len(points)*(8+1+8+8))
    if numeric_bytes>MAX_BYTES:
        raise ValueError('Shard complete paired trajectories to fit the bounded numerical block; retain every coordinate')
    if report_size is not None:
        if not callable(report_size):raise TypeError('Native output sizing requires an internal callback')
        report_size(rows=len(points),bins=len(source.fast.x),mode_dimensions=source.mode_dimensions,
                    cell_dimensions=source.cell_dimensions,numeric_bytes=numeric_bytes)
    mark('conditioning_and_selecting_whole_trajectories')
    checks = set(np.flatnonzero(role != 2)) | set(np.flatnonzero(np.r_[trajectory[:-1] != trajectory[1:], True]))
    rendered = render_native(source, points, kind=request['grid']['kind'], check_indices=checks, progress=progress)
    mark('rendering_children')
    # No child callback is allowed after constructing the owned parent below.
    progress('rendering_parent_references', 0)
    parent = NativeParentReference(inputs[0], inputs[1], out/'parent-forward', grid=request['grid'])
    if not np.array_equal(parent.q, rendered['q']):
        raise ValueError('Parent and child native instruments produced different Q grids')
    reference_curves = []; reference_background = []
    for i, a in enumerate(parent_a):
        curve, background = parent.evaluate(a)
        reference_curves.append(curve); reference_background.append(background)
        progress('rendering_parent_references', i+1)
    reference_curves = np.asarray(reference_curves); reference_background = np.asarray(reference_background)
    mark('rendering_parent_references')
    controls = []
    for i in np.flatnonzero(role != 2):
        child = rendered['curves'][i]-rendered['background']
        theory = reference_curves[i]-reference_background[i]
        denominator_norm = float(np.linalg.norm(theory))
        if denominator_norm <= 0:
            raise ValueError('A native parent control has no crystalline intensity')
        factor = float(theory@child/(theory@theory))
        error = float(np.linalg.norm(child-factor*theory)/max(np.linalg.norm(child), np.finfo(float).tiny))
        controls.append(dict(row=int(i), trajectory=int(trajectory[i]),
            intensity_scale=factor, relative_shape_error=error))
    if hashes != [file_sha(p) for p in inputs]:
        raise ValueError('A paired-pattern input changed during native calculation')
    mark('checking_parent_controls')
    destination = out/'pairs.npz'
    np.savez_compressed(destination, q=rendered['q'], theory_intensity=rendered['curves'],
        background=rendered['background'], parent_intensity=reference_curves,
        parent_background=reference_background, parent_a=parent_a, parameters=points,
        trajectory=trajectory, role=role, geometry_groups=rendered['labels'],
        ambiguous_geometry=rendered['labels'][:, 0] != rendered['labels'][:, 1],
        displacement_rms=paired.displacement_rms[selected], relative_metric_norm=paired.relative_metric_norm[selected])
    mark('writing_native_arrays')
    manifest = dict(schema='radar-structural-ranking-paired-pilot/v1',
        source_sha256=hashes[0], instrument_sha256=hashes[1], child_sha256=hashes[2],
        candidate=candidate, declared_child_group=child_group, grid=request['grid'],
        seed=seed, parent_seed=parent_seed, parent_relative_metric_range=list(parent_relative_range),
        trajectories=trajectories, steps=steps, trajectory_slice=list(interval), examples=len(points), bins=len(parent.q),
        groups=[('parent:'+hashes[0] if i == -1 else f'parent:{hashes[0]}:{seed}:{parent_seed}:trajectory-{i}') for i in trajectory],
        roles={'0': 'original_parent_anchor', '1': 'shared_parent_only_control', '2': 'child_hypothesis'},
        conditioning='Replace parent-invariant metric only; retain every displacement and symmetry-breaking metric coordinate',
        trajectory_admission='whole-child-rejection-at-fixed-shared-parent-context/v1',
        mode_dimensions=source.mode_dimensions, cell_dimensions=source.cell_dimensions,
        maximum_cell_reconstruction_error=paired.maximum_reconstruction_error,
        parent_control_comparisons=controls,
        parent_controls_shape_qualified=all(c['relative_shape_error'] <= 1e-6 and c['intensity_scale'] > 0 for c in controls),
        parent_native_calls=parent.native_calls, renderer=rendered['renderer'],
        independent_native_checks=rendered['native_checks'], original_profile_fallback=rendered['original_only'],
        regenerated_examples=rendered['regenerated'], rejected_original_directions=original.rejected_directions,
        partition_assigned=False, observations_included=False, classifier_trained=False,
        screening_qualified=False, scientific_result=False, pairs_sha256=file_sha(destination),
        package_sha256=package_signature(), seconds=time.monotonic()-began)
    manifest['timing']=dict(stages_seconds=stages,child_rendering_seconds=rendered['timing_seconds'],
        scope='Worker wall time; native forward includes its own mandatory anchor; independent checks and regeneration are separate')
    write_json(out/'pairs.json', manifest); progress('complete', len(points))
    return manifest
