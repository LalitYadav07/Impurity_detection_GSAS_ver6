from copy import deepcopy

from radar_pd_nova.symmetry_workspace import structural_discovery_feedback,candidate_display_name
from radar_pd_nova.symmetry_workspace import structural_result_names


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


def test_legacy_prepared_labels_require_the_same_sealed_construction():
    from gpx_recipe.symmetry.contracts import digest
    candidate=dict(id='iso2:child',magnetic_space_group=None,irrep='GM3+',
        child=dict(path='prepared/child.cif',sha256='a'*64),modes=['mode1'],
        transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],origin_shift=[0,0,0])
    original=dict(candidates=[dict(candidate,child=dict(path='original.cif',sha256='a'*64))],
        coverage=dict(candidates=[dict(id=candidate['id'],label='GM3+ · C2/m, basis={(1,0,0)}')]))
    catalog=dict(coverage=dict(candidates=[dict(id=candidate['id'],label='GM3+')]),
        provenance=dict(prepared_library=dict(original_catalog=original,original_catalog_sha256=digest(original))))
    before=deepcopy(catalog)
    assert candidate_display_name(catalog,candidate)=='GM3+ · C2/m'
    assert candidate_display_name(catalog,dict(candidate,modes=['different-mode']))=='GM3+'
    assert candidate_display_name(catalog,dict(candidate,origin_shift=[.5,0,0]))=='GM3+'
    invalid=deepcopy(catalog)
    invalid['provenance']['prepared_library']['original_catalog']['coverage']['candidates'][0]['label']='Changed'
    assert candidate_display_name(invalid,candidate)=='GM3+'
    assert catalog==before


def test_result_labels_use_the_same_hashed_catalog_as_the_plan(tmp_path):
    import json,hashlib
    root=tmp_path/'inputs'/'protocol';root.mkdir(parents=True)
    raw=json.dumps({'coverage':{'candidates':[{'id':'child','label':'GM3+ · C2/m, basis={(1,2,0)}'}]}}).encode()
    (root/'catalog.json').write_bytes(raw)
    protocol={'kind':'structural','catalog':{'path':'catalog.json','sha256':hashlib.sha256(raw).hexdigest()},
              'candidates':[{'id':'child','magnetic_space_group':None,'irrep':'GM3+'}]}
    assert structural_result_names(tmp_path,protocol,'protocol')=={'child':'GM3+ · C2/m'}
    (root/'catalog.json').write_text('{}')
    assert structural_result_names(tmp_path,protocol,'protocol')=={}
    protocol['kind']='magnetic'
    assert structural_result_names(tmp_path,protocol,'protocol')=={}
