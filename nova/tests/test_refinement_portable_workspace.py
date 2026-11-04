import copy
import json

from radar_pd_nova.refinement_runtime import portable_refinement_workspace


def test_checkpoint_references_verified_tables_without_losing_recovery_or_review_data():
    rows=[{'name':'x','sigma':0.01,'value':0.3}]*1000
    manifest={'jobs':{'j':{'remote_action':{'galaxy_job_id':'accepted'},'status':'completed',
        'result':{'execution':'polished.execution.json','files':{'polished.execution.json':'a'*64},
                  'stage_outcomes':[{'id':'positions','status':'completed','metric_rows':rows,
                                     'convergence':{'native_converged':False},'rwp':{'final':27.8}},
                                    {'id':'uiso','status':'rejected','reason':'Negative Uiso'}]}}}}
    original=copy.deepcopy(manifest)
    compact=portable_refinement_workspace(manifest)
    assert manifest==original
    result=compact['jobs']['j']['result'];stage=result['stage_outcomes'][0]
    assert 'metric_rows' not in stage
    assert stage['metric_rows_source']=={'artifact':'polished.execution.json','sha256':'a'*64,'stage_id':'positions'}
    assert stage['convergence']=={'native_converged':False} and stage['rwp']=={'final':27.8}
    assert result['stage_outcomes'][1]==original['jobs']['j']['result']['stage_outcomes'][1]
    assert compact['jobs']['j']['remote_action']=={'galaxy_job_id':'accepted'}
    assert len(json.dumps(compact))<len(json.dumps(original))/10
    assert portable_refinement_workspace(compact)==compact


def test_legacy_tables_stay_inline_when_full_receipt_is_not_recoverable():
    manifest={'jobs':{'j':{'result':{'stage_outcomes':[{'id':'s','metric_rows':[{'name':'x'}]}]}}}}
    assert portable_refinement_workspace(manifest)==manifest
