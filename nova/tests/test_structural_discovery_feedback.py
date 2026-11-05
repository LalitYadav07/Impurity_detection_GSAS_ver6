from copy import deepcopy

from radar_pd_nova.symmetry_workspace import structural_discovery_feedback,candidate_display_name


def test_all_failed_native_children_do_not_invite_a_refinement_plan():
    catalog={'candidates':[], 'coverage':{'discovered_count':9,'candidates':[
        {'id':str(i),'label':'GM3+ · C2/m, basis={(1,0,0)}','state':'construction_failed',
         'reason':"ModuleNotFoundError: No module named 'cif_occupancy'"} for i in range(9)]}}
    before=deepcopy(catalog)
    summary,rows,error=structural_discovery_feedback(catalog)
    assert '9 OPDs discovered' in summary and '0 ready to refine' in summary
    assert '9 construction failures' in summary
    assert 'no refinement plan' in error
    assert len(rows)==9 and all('cif_occupancy' in row['reason'] for row in rows)
    assert rows[0]['model']=='GM3+ · C2/m'
    assert catalog==before


def test_partial_catalog_keeps_ready_failed_deferred_and_irrep_failures_separate():
    catalog={'candidates':[{'id':'ready'}], 'coverage':{'discovered_count':3,'candidates':[
        {'id':'ready','label':'R-3m','state':'constructed','reason':''},
        {'id':'failed','label':'C2/m','state':'construction_failed','reason':'native import failed'},
        {'id':'deferred','label':'P-1','state':'deferred','reason':'Outside declared tier'}],
        'irrep_failures':['service response incomplete']}}
    summary,rows,error=structural_discovery_feedback(catalog)
    assert '1 ready to refine' in summary and '1 construction failures' in summary and '1 deferred' in summary
    assert not error
    assert rows[-1]['state']=='discovery failed' and rows[-1]['reason']=='service response incomplete'
    assert {row['id'] for row in rows}=={'ready','failed','deferred','irrep-failure-1'}


def test_empty_catalog_has_a_reason_instead_of_a_success_instruction():
    summary,rows,error=structural_discovery_feedback({'candidates':[]})
    assert '0 ready to refine' in summary and rows==[]
    assert 'No structural models passed' in error


def test_child_space_group_stays_visible_in_model_choices_and_plan():
    candidate={'id':'iso2:child','magnetic_space_group':None,'irrep':'GM3+'}
    catalog={'coverage':{'candidates':[{'id':candidate['id'],'label':'P1 (a,0) 12 C2/m, basis={(1,2,0)}'}]}}
    assert candidate_display_name(catalog,candidate)=='P1 (a,0) 12 C2/m'
    assert candidate_display_name({},candidate)=='GM3+'
    assert candidate_display_name(catalog,dict(candidate,magnetic_space_group='F_S mm2'))=='F_S mm2'
