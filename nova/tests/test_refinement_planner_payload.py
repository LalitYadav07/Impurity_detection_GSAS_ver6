"""Final planner messages keep host capabilities and distinct data/fit ranges."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.contracts import plan_hash
from gpx_recipe.planner import build_payload, expand_context, make_plan
from radar_pd_nova.refinement_workspace import RefinementWorkspace


def native_context():
    return dict(model_signature='synthetic-model', source={'name':'test.gpx','sha256':'test-source'},
        histograms=[dict(key='h0',name='PWDR synthetic',instrument={'Type':'PNT'},
                        limits=[[100.25,12000.75],[500.,8000.],[700.,720.]],point_count=4000,
                        background={'function':'chebyschev-1','coefficients':6},rwp='PRIVATE_RWP')],
        phases=[dict(key='p0',name='synthetic phase',histograms=['PWDR synthetic'],atoms=[])],
        covariance_summary={'private':'PRIVATE_COVARIANCE'},future_diagnostic='PRIVATE_DIAGNOSTIC')


def four_stage_plan(context, request):
    return dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],
        request=request,assumptions=[],questions=[],unsupported=[],stages=[
            dict(id=f'stage_{i}',title=f'Stage {i}',cycles=3,
                 free=[dict(kind='background',histogram='h0')],constraints=[],
                 **({'fit_ranges':[dict(histogram='h0',unit='native',lower=100.25,upper=12000.75)]}
                    if i==1 else {})) for i in range(1,5)])


class CapturingClient:
    def __init__(self, plan):
        self.plan=plan;self.calls=[]

    def complete(self, messages):
        self.calls.append(copy.deepcopy(messages))
        return json.dumps(self.plan),{'model':'test-client','label':'test-client','finish_reason':'stop'}


class Runtime:
    kind='local'

    def __init__(self, context):
        self.context=context;self.calls=[]

    def resolve(self, source):
        return copy.deepcopy(source)

    def call(self, action, source, directory, **kwargs):
        self.calls.append(action);Path(directory).mkdir(parents=True,exist_ok=True)
        if action=='inspect':
            return dict(status='inspected',context=copy.deepcopy(self.context))
        assert action=='preview'
        return dict(status='preview_passed',preview=dict(status='preview_passed',
            model_signature=self.context['model_signature'],source_sha256='test-source',
            plan_sha256=plan_hash(kwargs['plan']),stages=[]))


def test_workspace_capabilities_reach_the_complete_final_model_payload(tmp_path):
    context=native_context();original=copy.deepcopy(context)
    request=('Use exactly four stages over the full imported range. Apply the recipe to selected '
             'and future compatible scans; retain per-stage GPXs, lineage, and reviewed temperature mapping.')
    client=CapturingClient(four_stage_plan(context,request));runtime=Runtime(context)
    workspace=RefinementWorkspace(tmp_path,history_id='test-history',runtime=runtime,
        planner=lambda request,context,previous=None:make_plan(request,context,previous=previous,client=client))
    workspace.observe([dict(id='source',experiment='exp',scan_key='scan',source_sha256='test-source',
                            sample_key='sample',bank='2',scan_label='scan',run_number=1)])
    draft=workspace.prepare('exp','source',request)
    assert len(client.calls)==1 and draft['status']=='ready'
    payload=json.loads(client.calls[0][1]['content'])
    project=expand_context(payload['project']);host=project['host_execution']
    assert host['histogram_keys']==['h0']
    assert host['future_enrollment']['supported'] is True
    assert host['future_enrollment']['requires_approval'] is True
    assert host['output_artifacts']['stage_gpx'] is True
    assert host['output_artifacts']['final_gpx'] is True
    assert host['output_artifacts']['source_plan_lineage'] is True
    assert host['condition_mapping']['preserve_reviewed_temperature_axis'] is True
    assert payload['histogram_capabilities'][0]['native_data_range']==[100.25,12000.75]
    assert payload['histogram_capabilities'][0]['current_fit_range']==[500.,8000.]
    assert payload['user_request']==request
    assert 'PRIVATE_' not in json.dumps(payload)
    assert 'fit_ranges' in payload['fit_range_instruction']
    assert 'host_execution' in client.calls[0][0]['content']
    assert 'host_execution' not in draft['context']
    assert context==original
    assert runtime.calls==['inspect','preview']


def test_full_imported_range_is_an_explicit_native_edit_not_the_current_fit():
    context=native_context();payload=build_payload('Use the full imported range.',context)
    capability=payload['histogram_capabilities'][0]
    assert capability['native_data_range']==[100.25,12000.75]
    assert capability['current_fit_range']==[500.,8000.]
    assert capability['current_fit_covers_imported_range'] is False
    assert capability['full_imported_range_edit']==dict(histogram='h0',unit='native',lower=100.25,upper=12000.75)
    assert capability['current_exclusions']==[[700.,720.]]
    assert capability['point_count']==4000
    assert 'preserve' in payload['fit_range_instruction'].lower()


def test_equal_or_missing_ranges_do_not_invent_a_wider_domain():
    context=native_context();context['histograms'][0]['limits'][1]=[100.25,12000.75]
    cap=build_payload('Keep the current fit range.',context)['histogram_capabilities'][0]
    assert cap['current_fit_covers_imported_range'] is True
    del context['histograms'][0]['limits']
    cap=build_payload('Use the full imported range.',context)['histogram_capabilities'][0]
    assert cap['native_data_range'] is None and cap['current_fit_range'] is None
    assert cap['full_imported_range_edit'] is None
    assert cap['current_exclusions'] is None


@pytest.mark.parametrize('native',[[100.,100.],[True,100.],['10',100.],[]])
def test_invalid_imported_bounds_are_not_replaced_by_current_limits(native):
    context=native_context();context['histograms'][0]['limits']=[native,[500.,8000.]]
    cap=build_payload('Use the full imported range.',context)['histogram_capabilities'][0]
    assert cap['native_data_range'] is None
    assert cap['current_fit_range']==[500.,8000.]
    assert cap['full_imported_range_edit'] is None
    assert cap['current_fit_covers_imported_range'] is None


def test_nonfinite_native_context_is_rejected_before_any_model_call():
    context=native_context();context['histograms'][0]['limits'][0]=[float('nan'),12000.75]
    client=CapturingClient(four_stage_plan(context,'Use the full imported range.'))
    with pytest.raises(ValueError,match='JSON compliant'):
        make_plan('Use the full imported range.',context,client=client)
    assert client.calls==[]


def test_histogram_domains_remain_independent_and_zero_bound_is_not_an_executable_edit():
    context=native_context()
    context['histograms'].append(dict(key='h1',name='second',limits=[[0.,100.],[10.,90.]],point_count=200))
    caps=build_payload('Use the full imported range.',context)['histogram_capabilities']
    assert caps[0]['full_imported_range_edit']['lower']==100.25
    assert caps[1]['native_data_range']==[0.,100.]
    assert caps[1]['current_fit_range']==[10.,90.]
    assert caps[1]['full_imported_range_edit'] is None


def test_host_projection_is_allowlisted_at_each_level_and_binds_only_setup():
    context=native_context()
    context['host_execution']=dict(recipe_unit='One GPX',histogram_keys=['h0','invented'],
        future_enrollment=dict(supported=True,requires_approval=True,fit_result='PRIVATE_FUTURE_FIT'),
        output_artifacts=dict(stage_gpx=True,final_gpx=True,source_plan_lineage=True,
                              final_rwp='PRIVATE_ARTIFACT_RWP'),
        condition_mapping=dict(preserve_reviewed_temperature_axis=True,
                               covariance='PRIVATE_AXIS_COVARIANCE'),
        future_unknown='PRIVATE_UNKNOWN')
    payload=build_payload('Draft a recipe.',context);project=expand_context(payload['project'])
    assert 'PRIVATE_' not in json.dumps(payload)
    assert project['host_execution']['histogram_keys']==['h0']
    assert project['host_execution']['future_enrollment']==dict(supported=True,requires_approval=True)
    fingerprint=project['context_sha256']
    context['host_execution']['future_unknown']='DIFFERENT_PRIVATE_VALUE'
    assert expand_context(build_payload('Draft a recipe.',context)['project'])['context_sha256']==fingerprint
    context['host_execution']['future_enrollment']['supported']=False
    assert expand_context(build_payload('Draft a recipe.',context)['project'])['context_sha256']!=fingerprint


def test_genuinely_unsupported_operation_stays_blocked_after_capability_recheck():
    context=native_context();context['host_execution']=dict(future_enrollment=dict(supported=True))
    plan=four_stage_plan(context,'Refine an unsupported instrument parameter.')
    plan.update(stages=[],unsupported=['Instrument calibration changes are unsupported'])
    client=CapturingClient(plan)
    result=make_plan(plan['request'],context,client=client)
    assert len(client.calls)==2 and result['status']=='needs_input'
    assert result['plan']['unsupported']==plan['unsupported']
    assert result['plan']['stages']==[]
    repair=json.loads(client.calls[1][-1]['content'].rsplit('\n',1)[1])
    assert repair['host_execution']['future_enrollment']['supported'] is True
    assert repair['histogram_capabilities'][0]['current_fit_range']==[500.,8000.]
    assert repair['fit_range_instruction']
