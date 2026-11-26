"""Model labels expose the subgroup and OPD, not service form identifiers."""
from copy import deepcopy
from radar_pd_nova.symmetry_workspace import candidate_display_name, structural_discovery_feedback


def row(direction='a,0'):
    return ('GM3+, k7t5 · 7589 12 *P1 ('+direction+') 12 C2/m, '
            'basis={(1,2,0),(-1,0,0),(-1/3,-2/3,1/3)}, origin=(0,0,0), s=2, i=6')


def candidate():
    return dict(id='iso2:test', irrep='GM3+, k7t5', magnetic_space_group=None,
        transformation_matrix=[[1,2,0],[-1,0,0],[-1/3,-2/3,1/3]], origin_shift=[0,0,0])


def test_full_method_two_row_displays_space_group_irrep_and_direction():
    c=candidate(); catalog=dict(coverage=dict(candidates=[dict(id=c['id'],label=row())]))
    before=deepcopy(catalog)
    assert candidate_display_name(catalog,c)=='C2/m (No. 12) · GM3+ · OPD (a,0)'
    assert catalog==before


def test_distinct_order_parameter_directions_remain_distinct():
    c=candidate()
    names=[candidate_display_name(dict(coverage=dict(candidates=[dict(id=c['id'],label=row(d))])),c)
           for d in ['a,0','a,b']]
    assert names==['C2/m (No. 12) · GM3+ · OPD (a,0)','C2/m (No. 12) · GM3+ · OPD (a,b)']


def test_mismatched_embedding_is_not_relabelled_from_the_printed_row():
    c=candidate(); c['origin_shift']=[.5,0,0]
    catalog=dict(coverage=dict(candidates=[dict(id=c['id'],label=row())]))
    assert candidate_display_name(catalog,c)==row().split(', basis=',1)[0]


def test_incomplete_service_row_keeps_its_original_display():
    c=candidate(); label='GM3+, k7t5 · 7589 12 *P1 (a,0) 12 C2/m, basis={(1,0,0)}'
    catalog=dict(coverage=dict(candidates=[dict(id=c['id'],label=label)]))
    assert candidate_display_name(catalog,c)==label.split(', basis=',1)[0]


def test_discovery_feedback_uses_the_same_readable_full_row():
    catalog=dict(candidates=[candidate()], coverage=dict(discovered_count=1,
        candidates=[dict(id='iso2:test',label=row(),state='constructed',reason='')]))
    _,rows,error=structural_discovery_feedback(catalog)
    assert not error
    assert rows[0]['model']=='C2/m (No. 12) · GM3+ · OPD (a,0)'


def test_magnetic_labels_are_unchanged():
    c=candidate(); c['magnetic_space_group']='F_S mm2'
    assert candidate_display_name({},c)=='F_S mm2'


def test_invalid_printed_transformation_keeps_the_record_inspectable():
    c=candidate(); label=row().replace('(-1,0,0)','(1,2,0)')
    catalog=dict(coverage=dict(candidates=[dict(id=c['id'],label=label)]))
    assert candidate_display_name(catalog,c)==label.split(', basis=',1)[0]
