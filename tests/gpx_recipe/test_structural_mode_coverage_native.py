"""Named-mode coverage through native geometry, conditioning and diffraction."""
import numpy as np
import pytest
from pathlib import Path

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIlattice as L
from gpx_recipe.preparation.structural_sampling import from_source
from gpx_recipe.preparation.structural_mode_directions import from_source as native_directions
from gpx_recipe.preparation.structural_mode_coverage import supplement,SCHEMA,TARGETED_SCHEMA
from gpx_recipe.preparation.structural_parent_reference import ParentMetricSpace,rotations
from gpx_recipe.preparation.structural_conditioning import ParentConditioner,condition
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
from gpx_recipe.structural_live.optimization.native_forward import for_source
from test_native_profile_capture import native_source


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('targeted',[False,True])
def test_named_native_points_keep_original_conditioned_rays_and_exact_tof_profiles(tmp_path,kind,targeted):
    original,source=native_source(tmp_path/'input',kind)
    parent=G.G2Project(gpxfile=str(original)).phase(0)
    space=ParentMetricSpace(native_reciprocal_matrix(L.cell2A(parent.data['General']['Cell'][1:7])),
        rotations(parent.data['General']['SGData']))
    parents=space.draw(seed=291,count=4,relative_range=(.0001,.0005))
    transform=source.fast.meta['transformation_matrix']
    context=ParentConditioner(source,space,transform,parents)
    base=from_source(source,seed=17,trajectories=4,steps=4,
        displacement_range=(.001,.005),relative_metric_range=(.0001,.0003),
        trajectory_feasible=context.feasible)
    coverage=supplement(source,base,seed=19,steps=4,contexts=3,
        displacement_range=(.001,.004),trajectory_feasible=context.feasible)
    if targeted:
        from gpx_recipe.preparation.structural_mode_targets import append
        coverage=append(source,coverage,context)
        assert len(coverage.strong_targets['rays'])==len(coverage.rays)
    directions=native_directions(source)
    assert len(coverage.rays)==6*np.count_nonzero(~directions.gauge_only)
    assert directions.maximum_reconstruction_error<=1e-10
    first=condition(source,base,space,transform,parents)
    complete=condition(source,coverage.samples,space,transform,parents)
    assert len(complete.points)==1+len(parents)+len(coverage.samples.points)
    np.testing.assert_array_equal(complete.points[0],first.points[0])
    for parent_id in range(4):
        original_rows=np.flatnonzero(first.trajectory==parent_id)
        rows=np.flatnonzero(complete.trajectory==parent_id)
        for name in ['points','parent_a','role','displacement_rms','relative_metric_norm']:
            np.testing.assert_array_equal(getattr(complete,name)[rows[:len(original_rows)]],
                getattr(first,name)[original_rows])
        np.testing.assert_array_equal(complete.parent_a[rows],np.broadcast_to(parents[parent_id],complete.parent_a[rows].shape))
    engine=for_source(source) if kind=='PNT' else None
    assert kind!='PNT' or engine is not None
    n=source.fast.native;hist=n.histo_phases[0][source.fast.slow.hist.name]
    observed=[np.ma.array(value,copy=True) for value in hist['Data'][:3]]
    for point in complete.points:
        expected=source.native(point).copy()
        assert np.isfinite(expected).all()
        if engine is not None:np.testing.assert_array_equal(engine.evaluate(point),expected)
    if engine is not None:
        assert engine.anchor_comparison['array_identical'] and engine.anchor_checks==1
    for expected,actual in zip(observed,hist['Data'][:3]):
        np.testing.assert_array_equal(np.ma.getdata(actual),np.ma.getdata(expected))
        np.testing.assert_array_equal(np.ma.getmaskarray(actual),np.ma.getmaskarray(expected))


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('policy',[SCHEMA,TARGETED_SCHEMA])
def test_native_mode_producer_packs_complete_rays_and_shards_without_changing_original_points(tmp_path,kind,policy):
    from gpx_recipe.preparation.structural_pairs import generate
    from gpx_recipe.preparation.structural_mode_coverage import SCHEMA
    from gpx_recipe.preparation.structural_paired_blocks import read,pack_block
    from gpx_recipe.preparation import structural_mode_projection as projection
    from gpx_recipe.symmetry.compiler import file_sha
    original,_=native_source(tmp_path/'input',kind)
    parent=tmp_path/'parent.cif';G.G2Project(gpxfile=str(original)).phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    request=dict(source=str(parent),instrument=str(tmp_path/'input/profile.instprm'),child=str(child),
        grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,
            upper=125. if kind=='PNC' else 40000.,bins=701,bank=1),
        settings=dict(candidate=dict(id='public-ersi',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0],child=dict(path='child.cif',sha256=file_sha(child))),mode_radius=.05,cell_radius=.01))
    options=dict(seed=1947,parent_seed=8341,trajectories=4,steps=2,parent_relative_range=(.0001,.0005))
    generate(dict(request,destination=str(tmp_path/'old')),**options)
    old=read(tmp_path/'old')
    detail=generate(dict(request,destination=str(tmp_path/'new')),mode_coverage=policy,**options)
    new=read(tmp_path/'new');packed=pack_block(tmp_path/'new',tmp_path/'packed')
    assert detail['schema']==projection.RAW_SCHEMA and packed.manifest['schema']==projection.PACKED_SCHEMA
    assert packed.mode_projection is not None
    assert detail['mode_coverage']['rays'] and detail['parent_controls_shape_qualified']
    for parent_id in range(4):
        before=np.flatnonzero(old.arrays['trajectory']==parent_id)
        after=np.flatnonzero(new.arrays['trajectory']==parent_id)[:len(before)]
        for name,a in old.arrays.items():
            expected=a[before] if name not in {'q','background'} else a
            actual=new.arrays[name][after] if name not in {'q','background'} else new.arrays[name]
            np.testing.assert_array_equal(actual,expected)
    for name in new.arrays:np.testing.assert_array_equal(packed.arrays[name],new.arrays[name])
    shard=tmp_path/'shard'
    part=generate(dict(request,destination=str(shard)),mode_coverage=policy,trajectory_slice=[1,3],**options)
    fragment=read(shard);t=new.arrays['trajectory'];selected=(t==-1)|((t>=1)&(t<3))
    for name,a in new.arrays.items():
        np.testing.assert_array_equal(fragment.arrays[name],a[selected] if name not in {'q','background'} else a)
    extra=sum(1<=r['parent']<3 for r in part['mode_coverage']['rays'])
    assert part['independent_native_checks']==(1+2*2+extra*(1+int(policy==TARGETED_SCHEMA)) if kind=='PNT' else 0)
    # The same producer runs through the ordinary scientific-worker contract,
    # with exact row accounting before the scheduler releases result space.
    from gpx_recipe.preparation.worker import execute
    from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
    from gpx_recipe.preparation.native_size import read as read_size,PROJECTED_MODE_SCHEMA
    from gpx_recipe.symmetry.compiler import package_signature
    settings=dict(request['settings'],seed=options['seed'],parent_seed=options['parent_seed'],
        parent_relative_metric_range=list(options['parent_relative_range']),representation=REPRESENTATION,
        trajectories=4,steps=2,trajectory_slice=[1,3],mode_coverage=policy)
    worker=dict(request,settings=settings,destination=str(tmp_path/'worker'))
    result=execute('structural-examples',worker)
    published=read(Path(result['output']),names='examples');size=read_size(worker,package_signature())
    assert size['schema']==PROJECTED_MODE_SCHEMA and size['rows']==len(fragment.arrays['role'])
    assert size['numeric_bytes']==sum(a.nbytes for a in published.arrays.values())
    for name in fragment.arrays:np.testing.assert_array_equal(published.arrays[name],fragment.arrays[name])
