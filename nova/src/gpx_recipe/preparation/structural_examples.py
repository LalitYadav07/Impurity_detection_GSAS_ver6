"""Observation-free native examples for structural candidate ranking.

The supplied parent and child are scientific inputs, not predicted labels.
Actual geometry symmetry is checked at two explicit tolerances. Ambiguous
examples remain present, rather than receiving a compulsory child label.
This module neither trains a classifier nor qualifies one for screening.
"""
from pathlib import Path
import time
import numpy as np

from ..execution import write_json
from ..symmetry.compiler import file_sha, package_signature
from ..structural_live.native import prepare
from ..structural_live.service import SearchSettings, make_source
from ..structural_live.optimization.native_forward import for_source
from .reference import create_reference
from .worker import validate_request, verify_structural_child
from .structural_sampling import from_source


def declared_group(source):
    """Identify the model's declared operations separately from its zero state."""
    import spglib
    sg = source.fast.slow.phase.data['General']['SGData']
    rotations = []; translations = []
    signs = [1, -1] if sg['SGInv'] and not sg.get('SGFixed', False) else [1]
    for center in sg['SGCen']:
        for sign in signs:
            for rotation, translation in sg['SGOps']:
                rotations.append(sign*np.asarray(rotation, np.int32))
                translations.append((sign*np.asarray(translation, float)+center) % 1.)
    geometry = source.fast.empirical_constraint.geometry
    _, metric = geometry.geometry(np.zeros(source.mode_dimensions), np.zeros(source.cell_dimensions))
    group = spglib.get_spacegroup_type_from_symmetry(rotations, translations,
        lattice=np.linalg.cholesky(metric), symprec=1e-6)
    if group is None:
        raise ValueError('Declared child operators have no resolved space-group label')
    return dict(number=int(group.number), symbol=group.international_short,
        hall_number=int(group.hall_number), source='complete native child operations')


def geometry_labels(source, point, *, tolerances=(1e-6, 1e-5)):
    """Classify the complete expanded geometry, without snapping fitted atoms.

    This label includes species/occupancy, not an audit of ADP tensor symmetry
    or an assertion that a finite-resolution powder experiment distinguishes
    the returned groups. The classifier's observed-pattern ambiguity requires
    its own validation.
    """
    import spglib
    point = source.check(np.asarray(point, float))
    theta = source.lift@point; nm = source.mode_dimensions
    geometry = source.fast.empirical_constraint.geometry
    xyz, metric = geometry.geometry(theta[:nm], theta[nm:])
    lattice = np.linalg.cholesky(metric)
    identities = [(species, float(occupancy)) for species, occupancy in
                  zip(geometry.species, geometry.occupancy)]
    types = {identity: i+1 for i, identity in enumerate(sorted(set(identities)))}
    numbers = np.asarray([types[x] for x in identities], np.int32)
    labels = []
    for tolerance in tolerances:
        dataset = spglib.get_symmetry_dataset((lattice, xyz % 1., numbers), symprec=tolerance)
        if dataset is None:
            raise ValueError('Native example geometry has unresolved symmetry; no class label was invented')
        labels.append(int(dataset.number))
    return labels


def render_native(source,points,*,kind,check_indices,progress):
    """Shared exact child renderer for fixed and paired training examples."""
    x=source.fast.x
    hist=source.fast.slow.hist.data
    if np.any(hist['data'][1][1]) or not hist['data'][0].get('Dummy'):
        raise ValueError('Only observation-free native references belong in training preparation')
    d=np.asarray(source.fast.native.lattice.Pos2dsp(hist['Instrument Parameters'][0],x),float)
    q=2*np.pi/d;order=np.argsort(q);q=q[order]
    if not np.isfinite(q).all() or np.any(q<=0) or np.any(np.diff(q)<=0):
        raise ValueError('Instrument conversion has no unique finite positive Q grid')
    engine=for_source(source) if kind=='PNT' else None
    curves=[];labels=[];native_checks=0;regenerated=0;original_only=False
    for index,point in enumerate(points):
        labels.append(geometry_labels(source,point))
        curve=source.native(point) if engine is None or original_only else engine.evaluate(point)
        if engine is not None and not original_only and index in check_indices:
            expected=source.native(point);native_checks+=1
            if not np.array_equal(curve,expected):
                curves=[source.native(previous)[order] for previous in points[:index]]
                regenerated=len(curves);curve=expected;original_only=True
        if curve.shape!=x.shape or not np.isfinite(curve).all():
            raise ValueError('Native structural example returned a nonfinite or incomplete curve')
        if index==0:
            native_hist=source.fast.native.histo_phases[0][source.fast.slow.hist.name]
            background=np.asarray(native_hist['Data'][4],float)[order].copy()
            if not np.isfinite(background).all():raise ValueError('Native structural example background is not finite')
        curves.append(curve[order]);progress('rendering',index+1)
    return dict(q=q,curves=np.asarray(curves),background=background,labels=np.asarray(labels,np.int32),
        renderer=(engine.diagnostics() if engine is not None else dict(method='original native profile')),
        native_checks=native_checks,original_only=original_only,regenerated=regenerated)


def generate(request, *, seed, trajectories=64, steps=4, trajectory_slice=None):
    """One isolated native process constructs a complete candidate/bank block.

    The existing request validator binds source CIF, child, profile and grid.
    Reference histograms are created here with no measured observations.
    """
    request = validate_request('structural', request)
    if type(trajectories) is not int or not 1 <= trajectories <= 4096:
        raise ValueError('A bounded integer trajectory count is required')
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError('A bounded integer count of trajectory amplitude steps is required')
    if trajectory_slice is None:
        trajectory_slice = (0, trajectories)
    if (not isinstance(trajectory_slice, (tuple, list)) or len(trajectory_slice) != 2
            or any(type(i) is not int for i in trajectory_slice)
            or not 0 <= trajectory_slice[0] < trajectory_slice[1] <= trajectories):
        raise ValueError('A nonempty slice within the complete declared trajectory portfolio is required')
    inputs = [Path(request[k]).resolve() for k in ('source', 'instrument', 'child')]
    checksums = [file_sha(p) for p in inputs]
    settings = request['settings']; candidate = settings['candidate']
    verify_structural_child(settings, checksums[2])
    out = Path(request['destination']).resolve(); out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    denominator = 1+(trajectory_slice[1]-trajectory_slice[0])*steps
    def progress(state, processed):
        write_json(out/'generation-progress.json', dict(state=state, processed=processed,
            denominator=denominator, trajectory_slice=list(trajectory_slice),
            elapsed_seconds=time.monotonic()-began, observations_included=False))
    progress('preparing_native_context', 0)
    reference = create_reference(inputs[0], inputs[1], out/'reference', **request['grid'])
    prepare(reference['path'], inputs[2], candidate['transformation_matrix'],
            candidate['origin_shift'], out/'prepared')
    source = make_source(out/'prepared', out/'context', SearchSettings(
        mode_radius=settings['mode_radius'], cell_radius=settings['cell_radius']))
    samples = from_source(source, seed=seed, trajectories=trajectories, steps=steps)
    selected = (samples.trajectory >= trajectory_slice[0]) & (samples.trajectory < trajectory_slice[1])
    child_group = declared_group(source)
    if (np.count_nonzero(selected)+1)*len(source.fast.x)*8 > 128*1024**2:
        raise ValueError('Generate smaller structural example blocks within the 128 MiB array limit')
    points = np.vstack([np.zeros(len(source.radii)), samples.points[selected]])
    # The single parent identity is shared across all candidate aliases/banks.
    # It must never appear in different dataset splits under different labels.
    # Hold the same trajectory index out across every child and bank of this
    # parent, too. Parent-compatible cell-only trajectories can otherwise
    # produce the same observations under different candidate aliases.
    identities = ['parent:'+checksums[0]] + [
        f'parent:{checksums[0]}:{seed}:trajectory-{i}' for i in samples.trajectory[selected]]
    progress('rendering', 0)
    rendered=render_native(source,points,kind=request['grid']['kind'],
        check_indices=set(range(0,len(points),steps)),progress=progress)
    q=rendered['q'];curves=rendered['curves'];background=rendered['background'];labels=rendered['labels']
    if checksums != [file_sha(p) for p in inputs]:
        raise ValueError('A structural training input changed during generation')
    destination = out/'examples.npz'
    np.savez_compressed(destination, q=q, theory_intensity=np.asarray(curves), background=background, parameters=points,
        trajectory=np.r_[-1, samples.trajectory[selected]],
        geometry_groups=labels, ambiguous_geometry=labels[:, 0] != labels[:, 1],
        displacement_rms=np.r_[0., samples.displacement_rms[selected]],
        relative_metric_norm=np.r_[0., samples.relative_metric_norm[selected]])
    manifest = dict(schema='radar-structural-ranking-examples/v3',
        source_sha256=checksums[0], instrument_sha256=checksums[1], child_sha256=checksums[2],
        candidate=candidate, declared_child_group=child_group,
        grid=request['grid'], seed=seed, trajectories=trajectories, steps=steps,
        trajectory_slice=list(trajectory_slice), generated_trajectories=trajectory_slice[1]-trajectory_slice[0],
        groups=identities, examples=len(curves), bins=len(q),
        mode_dimensions=source.mode_dimensions, cell_dimensions=source.cell_dimensions,
        rejected_directions=samples.rejected_directions,
        rejected_directions_scope='Complete physical portfolio before shard selection',
        partition_assigned=False,
        partition_policy='Balanced group-level holdout assigned once over the complete library, after physical-duplicate grouping',
        background_scope='Native theoretical background stored separately; experimental preprocessing needs its own estimated-background validation',
        geometry_symmetry=dict(tolerances_angstrom=[1e-6, 1e-5],
            scope='Species/occupancy and complete expanded geometry; no ADP-tensor or powder-identifiability claim'),
        actual_geometry_groups=sorted(set(map(int, labels.ravel()))),
        ambiguous_examples=int(np.count_nonzero(labels[:, 0] != labels[:, 1])),
        renderer=rendered['renderer'],
        independent_native_anchor_checks=rendered['native_checks'],
        original_profile_fallback=rendered['original_only'], regenerated_examples=rendered['regenerated'],
        package_sha256=package_signature(), examples_sha256=file_sha(destination),
        observations_included=False, classifier_trained=False, screening_qualified=False,
        scientific_result=False, seconds=time.monotonic()-began)
    write_json(out/'examples.json', manifest)
    progress('complete', len(curves))
    return manifest
