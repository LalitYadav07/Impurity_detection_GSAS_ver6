"""Strong geometry coverage adds information without deleting weak controls."""
import copy

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_layout as layout
from gpx_recipe.preparation.structural_mode_coverage import supplement,TARGETED_SCHEMA
from gpx_recipe.preparation.structural_mode_targets import append,validate_record
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
from test_structural_mode_coverage import source,portfolio


class Context:
    def __init__(self,s):
        self.source=s;self.controls=np.zeros((6,len(s.radii)))
        self.controls[:,-1]=np.linspace(.0001,.005,6)
    def point(self,parent,old):return self.controls[parent]+old
    def feasible(self,parent,rows):
        return np.all(abs(rows+self.controls[parent])<=self.source.radii,axis=1)


def setup(*,large_cell=False,amplitude=.0005):
    s=source();g=s.fast.empirical_constraint.geometry
    if large_cell:g.a0[:3]=.01
    xyz=np.array([[.1,.1,.1],[.45,.45,.2]])
    def geometry(modes,cells):
        a=g.a0+g.cell_mapping@cells
        return xyz+g.mapping@modes,np.linalg.inv(native_reciprocal_matrix(a))
    g.geometry=geometry
    base=portfolio(s);context=Context(s)
    old=supplement(s,base,seed=4,contexts=3,displacement_range=(amplitude,amplitude),
        trajectory_feasible=context.feasible)
    return s,context,old


def test_original_mixed_and_named_samples_remain_bit_identical_before_each_extra_endpoint():
    s,context,old=setup();result=append(s,old,context)
    n=old.base_examples
    for name in ['points','trajectory','displacement_rms','relative_metric_norm']:
        before=getattr(old.samples,name);after=getattr(result.samples,name)
        np.testing.assert_array_equal(before[:n],after[:n]);assert not after.flags.writeable
        for first,second in zip(old.rays,result.rays):
            a,b=first['rows'];c,d=second['rows']
            assert d-c==b-a+1
            np.testing.assert_array_equal(before[a:b],after[c:d-1])
    assert len(result.samples.points)==len(old.samples.points)+len(old.rays)
    assert all(r['status']=='appended_strong_target' and r['strong'] is True
               for r in result.strong_targets['rays'])
    assert result.samples.rejected_directions==old.samples.rejected_directions
    assert {r['parent'] for r in result.rays}=={r['parent'] for r in old.rays}
    document=layout.describe(result,contexts=3)
    assert document['schema']==TARGETED_SCHEMA
    ids,roles,raw,checks=layout.rows(dict(mode_coverage=document),6,4,(0,6))
    assert len(ids)==1+6*5+len(old.rays)*5
    assert len(checks)==6+2*len(old.rays)
    assert set(raw[raw>=0])==set(range(len(result.samples.points)))
    for ray in result.rays:
        a,b=ray['rows'];positions=[int(np.flatnonzero(raw==row)[0]) for row in (b-2,b-1)]
        assert set(positions)<=set(checks)


def test_unreachable_contact_target_repeats_and_explains_each_endpoint_without_dropping_a_mode():
    s,context,old=setup(large_cell=True);result=append(s,old,context)
    assert all(r['status']=='target_not_reached_inside_declared_ray_bound'
               and r['strong'] is False for r in result.strong_targets['rays'])
    for ray in result.rays:
        a,b=ray['rows']
        np.testing.assert_array_equal(result.samples.points[b-1],result.samples.points[b-2])
    assert len(result.rays)==len(old.rays)


def test_existing_strong_points_are_not_replaced_with_newly_sampled_easier_points():
    s,context,old=setup(amplitude=.035);result=append(s,old,context)
    reports=result.strong_targets['rays'];assert any(r['status']=='retained_strong_endpoint' for r in reports)
    for ray,report in zip(result.rays,reports):
        if report['status']=='retained_strong_endpoint':
            a,b=ray['rows'];np.testing.assert_array_equal(result.samples.points[b-1],result.samples.points[b-2])


def test_rejected_native_target_keeps_the_original_weak_ray_and_records_the_reason():
    s,context,old=setup();context.feasible=lambda parent,points:np.zeros(len(points),bool)
    result=append(s,old,context)
    assert all(r['status']=='native_admission_rejected' and r['strong'] is False
               for r in result.strong_targets['rays'])
    assert len(result.rays)==len(old.rays)


@pytest.mark.parametrize('result',[True,np.ones(1),np.ones(2,bool)])
def test_incomplete_or_untyped_native_admission_is_not_a_physical_rejection(result):
    s,context,old=setup();context.feasible=lambda *args:result
    with pytest.raises(ValueError,match='one native physical boolean'):append(s,old,context)


def test_technical_native_failure_is_not_disguised_as_an_unreachable_target():
    s,context,old=setup()
    def broken(*args):raise RuntimeError('native geometry unavailable')
    context.feasible=broken
    with pytest.raises(RuntimeError,match='unavailable'):append(s,old,context)


@pytest.mark.parametrize('change',['missing_ray','wrong_identity','false_strong','nonfinite','unknown_status','extra'])
def test_target_inventory_cannot_silently_drop_or_qualify_a_weak_case(change):
    s,context,old=setup();result=append(s,old,context);record=copy.deepcopy(result.strong_targets)
    if change=='missing_ray':record['rays'].pop()
    elif change=='wrong_identity':record['rays'][0]['ray']=1
    elif change=='false_strong':record['rays'][0]['strong']=False
    elif change=='nonfinite':record['rays'][0]['appended_amplitude']=float('nan')
    elif change=='unknown_status':record['rays'][0]['status']={}
    else:record['qualified']=True
    with pytest.raises(ValueError):validate_record(record,result.rays)


def test_a_second_extension_and_a_different_parent_context_are_refused():
    s,context,old=setup();result=append(s,old,context)
    with pytest.raises(ValueError,match='original coverage'):append(s,result,context)
    context.source=source()
    with pytest.raises(ValueError,match='matching parent'):append(s,old,context)


def test_worker_and_output_reservation_count_the_appended_endpoint(tmp_path):
    from test_structural_paired_preparation import inputs
    from gpx_recipe.preparation.structural_jobs import validate_examples
    from gpx_recipe.preparation import native_size
    from gpx_recipe.symmetry.compiler import package_signature
    from pathlib import Path
    request,_=inputs(tmp_path);request['settings']['mode_coverage']=TARGETED_SCHEMA
    assert validate_examples(request)==request
    Path(request['destination']).mkdir()
    counts=dict(schema=TARGETED_SCHEMA,native_modes=2,gauge_modes=1,contexts=3,rays=6,selected_rays=6)
    # Fixture: eight original two-row groups, six old single-step named rays
    # and six appended target rows, plus the one shared parent anchor.
    rows=29
    native_size.publish(request,rows=rows,bins=101,mode_dimensions=1,cell_dimensions=2,
        numeric_bytes=native_size.paired_array_bytes(rows,101,1,2),mode_coverage_rows=counts)
    d=native_size.read(request,package_signature());assert d['rows']==rows
    assert d['numeric_bytes']==native_size.paired_array_bytes(rows,101,1,2)


@pytest.mark.parametrize('interval',[(0,1),(1,3),(3,6)])
def test_whole_parent_shards_keep_both_old_and_extra_endpoint_checks(interval):
    s,context,old=setup();result=append(s,old,context);document=dict(mode_coverage=layout.describe(result))
    full=layout.rows(document,6,4,(0,6));part=layout.rows(document,6,4,interval)
    selected=(full[0]==-1)|((full[0]>=interval[0])&(full[0]<interval[1]))
    for index in range(3):np.testing.assert_array_equal(part[index],full[index][selected])
    expected_checks=set(np.flatnonzero(np.isin(np.flatnonzero(selected),full[3])))
    assert set(part[3])==expected_checks


@pytest.mark.parametrize('change',['missing_target_record','extra_point','missing_point','wrong_policy','false_weak'])
def test_layout_refuses_forged_or_incomplete_appended_endpoints(change):
    s,context,old=setup();result=append(s,old,context);record=layout.describe(result)
    if change=='missing_target_record':record.pop('strong_targets')
    elif change=='extra_point':record['rays'][0]['rows'][1]+=1
    elif change=='missing_point':record['rays'][0]['rows'][1]-=1
    elif change=='wrong_policy':record['schema']='radar-additive-native-mode-coverage/v1'
    else:
        row=record['strong_targets']['rays'][0];row.update(status='native_admission_rejected',strong=False)
    with pytest.raises(ValueError):layout.rows(dict(mode_coverage=record),6,4,(0,6))
