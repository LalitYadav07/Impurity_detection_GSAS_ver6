"""Rollback must discard failed-stage mutations before an independent stage."""
import copy,hashlib,json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from gpx_recipe import execution as E

class Project:
    def __init__(self,gpxfile):
        self.path=gpxfile;self.data=json.loads(open(gpxfile).read())
    def save(self,path=None):
        if path:self.path=str(path)
        open(self.path,'w').write(json.dumps(self.data))
    def do_refinements(self,_):
        self.data['value']+=1
        self.data['last_refined']=self.data.get('current')
        self.data['Covariance']={'data':{'freshCOV':True}}
    def histograms(self):return []
    def phases(self):return []

@pytest.fixture
def setup(monkeypatch,tmp_path):
    source=tmp_path/'source.gpx';source.write_text(json.dumps({'value':0,'Covariance':{'data':{}}}))
    context={'source':{'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'model_signature':'model','phases':[]}
    seen=[]
    monkeypatch.setattr(E,'snapshot',lambda _:context)
    monkeypatch.setattr(E,'validate_plan',lambda *a:None)
    monkeypatch.setattr(E,'plan_hash',lambda _: 'plan')
    monkeypatch.setattr(E,'gsas',lambda:SimpleNamespace(G2Project=Project,__package__=''))
    monkeypatch.setattr(E,'stage_table',lambda plan,ctx:[{'id':s['id'],'title':s['id']} for s in plan['stages']])
    def configure(project,stage,*_):
        seen.append((stage['id'],project.data['value']))
        project.data['current']=stage['id']
    monkeypatch.setattr(E,'configure',configure)
    monkeypatch.setattr(E,'check_preflight',lambda *a:{})
    monkeypatch.setattr(E,'extract_gsasii_refinement_state',lambda *a,**k:{'parameter_space':{'effective_vary':[],'last_vary':[],'parameters':[],'frozen_variables':[]},'constraints':{},'covariance':{}})
    monkeypatch.setattr(E.importlib,'import_module',lambda _:SimpleNamespace(Dict2Map=lambda *a:None))
    monkeypatch.setattr(E,'constraint_bound_audit',lambda *a: {})
    monkeypatch.setattr(E,'native_storage_audit',lambda *a:{})
    monkeypatch.setattr(E,'parameter_changes',lambda *a:{})
    monkeypatch.setattr(E,'blocking_frozen_variables',lambda *a:[])
    monkeypatch.setattr(E,'convergence_evidence',lambda *a:{})
    for name in ['require_fixed_values_unchanged','require_native_covariance','require_constraint_bounds','require_native_storage']:
        monkeypatch.setattr(E,name,lambda *a:None)
    monkeypatch.setattr(E.advanced,'audit',lambda *a:{})
    monkeypatch.setattr(E.advanced,'relation_key',lambda _: '')
    monkeypatch.setattr(E.advanced,'check_requested_relations',lambda *a,**k:None)
    monkeypatch.setattr(E.advanced,'check_normalization',lambda *a:[])
    def totals(project,*_):
        if project.data.get('last_refined')=='bad' and project.data['Covariance']['data'].get('freshCOV'):
            raise ValueError('negative Uiso')
    monkeypatch.setattr(E.advanced,'check_site_totals',totals)
    stages=[{'id':i,'title':i,'constraints':[],'free':[]} for i in ['first','bad','last']]
    return source,context,seen,{'stages':stages,'questions':[],'unsupported':[],'failure_policy':'continue_independent'}

def test_rejected_stage_restores_values_and_continues(setup,tmp_path):
    source,ctx,seen,plan=setup
    result=E.run_plan(source,tmp_path/'out.gpx',plan,approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert [s['status'] for s in result['stages']]==['completed','rejected','completed']
    assert result['rejected_stages']==['bad']
    assert seen==[('first',0),('bad',1),('last',1)]
    assert json.loads((tmp_path/'out.gpx').read_text())['value']==2
    assert not (tmp_path/'out_bad.gpx').exists()
    assert json.loads(source.read_text())['value']==0

def test_worker_exports_completed_checkpoints_after_recovery(setup,tmp_path,monkeypatch):
    from gpx_recipe import compiler,worker
    source,context,seen,plan=setup
    approval={'approved_plan_sha256':'plan','model_signature':'model'}
    def script(*args):
        return ("from gpx_recipe.execution import run_plan\n"
                "import sys\n"
                f"run_plan(sys.argv[2],sys.argv[4],{plan!r},approved={approval!r})\n")
    monkeypatch.setattr(compiler,'compile_script',script)
    def view(path,destination):
        # Read the actual checkpoint: rejected and skipped paths do not exist.
        payload=json.loads(path.read_text())
        result={'histograms':[],'parameters':[], 'value':payload['value']}
        destination.write_text(json.dumps(result))
        return result
    monkeypatch.setattr(worker,'write_project_view',view)
    result=worker.work('run',source,tmp_path/'worker',plan=plan,approval=approval)
    assert result['status']=='completed'
    assert result['rejected_stages']==['bad']
    assert [row['id'] for row in result['stages']]==['first','last']
    assert result['final_gpx']=='verified_final.gpx'
    assert json.loads((tmp_path/'worker'/result['final_gpx']).read_text())['value']==2
    assert [row['status'] for row in result['stage_outcomes']]==['completed','rejected','completed']


def test_dependent_stage_is_skipped(setup,tmp_path):
    source,ctx,seen,plan=setup;plan['stages'][-1]['depends_on']=['BAD']
    result=E.run_plan(source,tmp_path/'out.gpx',plan,approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert result['stages'][-1]['status']=='skipped'
    assert seen==[('first',0),('bad',1)]

def test_contract_failure_is_not_recovered(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=setup
    def invalid(*args):raise ValueError('Unexpected active parameters')
    monkeypatch.setattr(E,'check_preflight',invalid)
    with pytest.raises(ValueError,match='Unexpected active parameters'):
        E.run_plan(source,tmp_path/'out.gpx',plan,approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert seen==[('first',0)]

def test_stop_policy_preserves_previous_behavior(setup,tmp_path):
    source,ctx,seen,plan=setup;plan['failure_policy']='stop'
    with pytest.raises(ValueError,match='negative Uiso'):
        E.run_plan(source,tmp_path/'out.gpx',plan,approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert seen==[('first',0),('bad',1)]


def report_native_abort(monkeypatch):
    from gpx_recipe.convergence import convergence_evidence
    original=E.extract_gsasii_refinement_state
    def extract(*args,**kwargs):
        state=original(*args,**kwargs)
        aborted=args[1].data.get('last_refined')=='bad'
        state['fit_statistics']={'Aborted':aborted,'converged':not aborted}
        return state
    monkeypatch.setattr(E,'extract_gsasii_refinement_state',extract)
    monkeypatch.setattr(E,'convergence_evidence',convergence_evidence)
    monkeypatch.setattr(E.advanced,'check_site_totals',lambda *a:None)


def test_native_abort_rolls_back_without_an_optional_convergence_screen(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=setup
    report_native_abort(monkeypatch)
    result=E.run_plan(source,tmp_path/'out.gpx',plan,
        approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert [s['status'] for s in result['stages']]==['completed','rejected','completed']
    assert result['stages'][1]['convergence']['aborted'] is True
    assert 'aborted' in result['stages'][1]['reason'].lower()
    assert seen==[('first',0),('bad',1),('last',1)]
    assert not (tmp_path/'out_bad.gpx').exists()
    assert json.loads((tmp_path/'out.gpx').read_text())['value']==2


def test_native_abort_stops_when_recovery_was_not_approved(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=setup;plan['failure_policy']='stop'
    report_native_abort(monkeypatch)
    with pytest.raises(E.StageRejected,match='aborted'):
        E.run_plan(source,tmp_path/'out.gpx',plan,
            approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert seen==[('first',0),('bad',1)]
    assert not (tmp_path/'out_bad.gpx').exists()


def test_nonconvergence_without_abort_retains_its_review_evidence(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=setup
    monkeypatch.setattr(E.advanced,'check_site_totals',lambda *a:None)
    monkeypatch.setattr(E,'convergence_evidence',lambda *a:{'aborted':False,'native_converged':False})
    result=E.run_plan(source,tmp_path/'out.gpx',plan,
        approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert all(s['status']=='completed' for s in result['stages'])
    assert all(s['convergence']['native_converged'] is False for s in result['stages'])


def numerical_setup(setup):
    source,ctx,seen,plan=setup
    data=json.loads(source.read_text())
    data['Controls']={'data':{'deriv type':'numeric'}}
    source.write_text(json.dumps(data))
    ctx['source']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    for stage in plan['stages']:stage['max_function_evaluations']=500
    return source,ctx,seen,plan


def test_unobserved_evaluation_budget_is_a_technical_failure(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=numerical_setup(setup)
    @contextmanager
    def unobserved(main,maximum):
        yield {'maximum':maximum,'evaluations':0,'exhausted':False}
    monkeypatch.setattr(E,'residual_budget',unobserved)
    with pytest.raises(RuntimeError,match='did not pass through'):
        E.run_plan(source,tmp_path/'out.gpx',plan,
            approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert seen==[('first',0)]
    assert not (tmp_path/'out_first.gpx').exists()


def test_unsuccessful_minpack_termination_is_not_published(setup,tmp_path,monkeypatch):
    source,ctx,seen,plan=numerical_setup(setup)
    plan['failure_policy']='stop'
    @contextmanager
    def budget(main,maximum):
        yield {'maximum':maximum,'evaluations':3,'exhausted':False}
    @contextmanager
    def failed_termination(main):
        yield {'state':'observed','final':{'solver_success':False,'termination':'function_evaluation_limit'}}
    monkeypatch.setattr(E,'residual_budget',budget)
    monkeypatch.setattr(E,'observe_leastsq',failed_termination)
    with pytest.raises(E.StageRejected,match='function_evaluation_limit'):
        E.run_plan(source,tmp_path/'out.gpx',plan,
            approved={'approved_plan_sha256':'plan','model_signature':'model'})
    assert seen==[('first',0)]
    assert not (tmp_path/'out_first.gpx').exists()
