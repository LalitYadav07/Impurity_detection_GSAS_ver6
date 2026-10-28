"""Literal lists bind user instructions, not benchmark answers or inferred chemistry."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from gpx_recipe.intent import literal_stage_cards,check_literal_free,parse_free_item


@pytest.fixture
def context():
    return {'model_signature':'intent-test-model','histograms':[{'key':'h0','name':'PWDR p1_h1.gsa'},{'key':'h1','name':'PWDR second.xye'}],
            'phases':[{'key':'p0','name':'Alpha 0.5 oxide','histograms':['PWDR p1_h1.gsa','PWDR second.xye'],'atoms':[{'label':'O1'},{'label':'O3'}]},
                      {'key':'p1','name':"B19'",'histograms':['PWDR p1_h1.gsa','PWDR second.xye'],'atoms':[{'label':'O1'},{'label':'Ni1'}]}]}


def test_histogram_background_cannot_leak_from_previous_stage(context):
    request="""Every stage lists its complete free set; everything unlisted stays fixed.
Stage 1: Free: existing polynomial background for histogram h0 ('PWDR p1_h1.gsa').
Stage 2: Free: existing polynomial background for histogram h1 ('PWDR second.xye').
"""
    cards=literal_stage_cards(request,context);assert [c['literal_free'] for c in cards]==[[{'kind':'background','histogram':'h0'}],[{'kind':'background','histogram':'h1'}]]
    proposed={'questions':[],'unsupported':[],'stages':[{'free':cards[0]['literal_free']},{'free':cards[0]['literal_free']+cards[1]['literal_free']}]}
    original=copy.deepcopy(proposed);corrections=check_literal_free(proposed,cards)
    assert proposed==original and corrections[0]['request_free']==cards[1]['literal_free']
    assert corrections[0]['stage']==2 and corrections[0]['model_free']==original['stages'][1]['free']


def test_phase_names_with_decimal_points_and_quotes_do_not_truncate_list(context):
    text="""Each stage has a complete active set.
Stage 1: Free: O1 in phase 'Alpha 0.5 oxide': symmetry-independent coordinates, Uiso; HAP phase scale of phase 'Alpha 0.5 oxide' for histogram h0 ('PWDR p1_h1.gsa'). Set h0 background to cosine.
"""
    free=literal_stage_cards(text,context)[0]['literal_free']
    assert free==[{'kind':'atom','phase':'p0','atom':'O1','flags':'XU'},{'kind':'phase_scale','phase':'p0','histogram':'h0'}]
    assert parse_free_item('Ni1 in phase "B19\'": Uiso',context)=={'kind':'atom','phase':'p1','atom':'Ni1','flags':'U'}


def test_canonical_shorthand_combines_joint_flags(context):
    c=literal_stage_cards('Complete free set.\nStage 1: Free: p0:O1 X; p0:O1 U; cell p1.',context)
    assert c[0]['literal_free']==[{'kind':'atom','phase':'p0','atom':'O1','flags':'XU'},{'kind':'cell','phase':'p1'}]


def test_unknown_or_wildcard_clause_never_becomes_a_partial_authoritative_list(context):
    for clause in ['background h0; all oxygen atoms','O99 in phase p0: Uiso','cell p9','background h99']:
        cards=literal_stage_cards('Complete free set.\nStage 1: Free: '+clause+'.',context)
        assert cards[0]['literal_free'] is None


def test_open_natural_language_and_questions_are_not_overridden(context):
    cards=literal_stage_cards('Stage 1: Refine oxygen and keep reasonable things active.',context)
    assert cards[0]['literal_free'] is None
    proposed={'questions':['Which phase?'],'unsupported':[],'stages':[]}
    assert check_literal_free(proposed,cards)==[]


def test_names_are_data_not_refinement_instructions(context):
    context['phases'][0]['name']='Uiso background h1 cell p1'
    target=parse_free_item("O1 in phase 'Uiso background h1 cell p1': positions",context)
    assert target=={'kind':'atom','phase':'p0','atom':'O1','flags':'X'}
    # The h1/p1 strings inside a quoted filename cannot choose an owner.
    assert parse_free_item("HAP phase scale of phase 'Alpha 0.5 oxide' for histogram h0 ('PWDR p1_h1.gsa')",{'phases':[{'key':'p0','name':'Alpha 0.5 oxide','histograms':['PWDR p1_h1.gsa','PWDR second.xye'],'atoms':[]}], 'histograms':context['histograms']})=={'kind':'phase_scale','phase':'p0','histogram':'h0'}


def test_no_histogram_local_fiction_for_shared_atom_or_phase(context):
    assert parse_free_item('p0:O1 positions only for histogram h0',context) is None
    assert parse_free_item('unit cell p0 only for histogram h1',context) is None
    assert parse_free_item('background of phase p0 for histogram h0',context) is None


def test_model_must_repair_carryover_instead_of_backend_silently_rewriting():
    import json
    from gpx_recipe.planner import make_plan
    ctx={'model_signature':'x','phases':[{'key':'p0','name':'phase','histograms':['first','second'],'atoms':[]}], 'histograms':[{'key':'h0','name':'first'},{'key':'h1','name':'second'}]}
    text='Use exactly 2 stages. Each has a complete free set.\nStage 1: Free: background h0.\nStage 2: Free: background h1.'
    plan={'schema':'radar-gpx-recipe/v1','model_signature':'x','request':text,'assumptions':[],'questions':[],'unsupported':[],'stages':[{'id':'s1','title':'first background','cycles':4,'free':[{'kind':'background','histogram':'h0'}],'constraints':[]},{'id':'s2','title':'second background','cycles':4,'free':[{'kind':'background','histogram':'h0'},{'kind':'background','histogram':'h1'}],'constraints':[]}]}
    good=copy.deepcopy(plan);good['stages'][1]['free']=good['stages'][1]['free'][1:]
    class Client:
        def __init__(self):self.calls=[]
        def complete(self,messages):
            self.calls.append(copy.deepcopy(messages));return json.dumps(plan if len(self.calls)==1 else good),{'model':'test'}
    client=Client();result=make_plan(text,ctx,client=client)
    assert len(client.calls)==2 and result['plan']==good
    assert result['literal_guard_rejections'][0]['attempt']==1
    assert 'titles and assumptions together' in client.calls[1][-1]['content']
    assert json.loads(result['calls'][0]['response'])==plan


def test_existing_equality_groups_distinguish_full_release_and_full_hold():
    from gpx_recipe.planner import build_payload
    ctx={'model_signature':'x','phases':[{'key':'p0','name':'oxide','histograms':['first'],'atoms':[{'label':name,'adp_type':'I','position_dof':1} for name in ['C1','O1','H1','H2']]}], 'histograms':[{'key':'h0','name':'first'}], 'uiso_equivalence_groups':[[{'phase':'p0','atom':'C1'},{'phase':'p0','atom':'O1'}],[{'phase':'p0','atom':'H1'},{'phase':'p0','atom':'H2'}]]}
    payload=build_payload('Each stage has a complete free set.\nStage 1: Free: p0:C1 Uiso; p0:O1 Uiso.\nStage 2: Free: background h0.',ctx)
    scopes=[c['existing_uiso_relation_scopes'] for c in payload['stage_instruction_cards']]
    assert [x['status'] for x in scopes[0]]==['all_released_compatible','all_fixed_compatible']
    assert [x['status'] for x in scopes[1]]==['all_fixed_compatible','all_fixed_compatible']
    assert scopes[0][0]['released_count']==2 and scopes[0][1]['held_count']==2


def test_explicit_partial_source_equality_is_actionable_without_an_api_call():
    from gpx_recipe.planner import make_plan
    ctx={'model_signature':'x','phases':[{'key':'p0','name':'phase','histograms':['first'],'atoms':[{'label':a,'adp_type':'I','position_dof':1} for a in ['C1','O1']]}], 'histograms':[{'key':'h0','name':'first'}], 'uiso_equivalence_groups':[[{'phase':'p0','atom':'C1'},{'phase':'p0','atom':'O1'}]]}
    class Client:
        def complete(self,messages):pytest.fail('No model call for a proven explicit constraint contradiction')
    result=make_plan('Complete free set.\nStage 1: Free: p0:O1 Uiso.',ctx,client=Client())
    assert result['status']=='needs_input' and result['calls']==[] and result['plan']['stages']==[]
    assert 'p0:C1' in result['plan']['questions'][0] and 'Authorize release' in result['plan']['questions'][0]


def test_tiny_phase_scale_never_changes_linked_phase_count():
    from gpx_recipe.planner import build_payload
    ctx={'model_signature':'x','phases':[{'key':p,'name':p,'histograms':['h'],'phase_scales':[{'histogram':'h0','value':v}],'atoms':[]} for p,v in [('p0',1.),('p1',1e-12)]],'histograms':[{'key':'h0','name':'h'}]}
    histogram=build_payload('Normalize both phases',ctx)['histogram_capabilities'][0]
    assert histogram['linked_phase_count']==2 and histogram['linked_phases']==['p0','p1']


def test_fully_bound_numbered_free_lists_also_bind_the_stage_count(context):
    from gpx_recipe.planner import build_payload
    payload=build_payload('Complete free set.\n1. Free: background h0.\n2. Free: background h1.',context)
    assert payload['explicit_stage_count']==2


def test_natural_single_atom_only_clause_forbids_an_invented_cell(context):
    from gpx_recipe.planner import build_payload
    text="Use exactly two stages. In the first stage refine the background and scales. In stage two fix both of those and refine only p0:O1 positions and Uiso jointly, for at most 5 cycles. Hold every other parameter fixed."
    payload=build_payload(text,context);cards=payload['stage_instruction_cards']
    assert cards[0]['literal_free'] is None
    assert cards[1]['literal_free']==[{'kind':'atom','phase':'p0','atom':'O1','flags':'XU'}]
    proposed={'questions':[],'unsupported':[],'stages':[{'free':[]},{'free':cards[1]['literal_free']+[{'kind':'cell','phase':'p0'}]}]}
    assert check_literal_free(proposed,cards)[0]['stage']==2


@pytest.mark.parametrize('clause',['p0:O1 Uiso, and the cell','p0:O1 Uiso; then refine p1:Ni1 Uiso','all oxygen sites','p0:O99 Uiso'])
def test_ambiguous_or_multiple_natural_targets_are_not_partially_bound(context,clause):
    from gpx_recipe.intent import natural_atom_only_cards
    cards=natural_atom_only_cards('Use one stage. Stage 1: refine only '+clause+'.',context,1)
    assert not cards


def test_one_stage_count_declaration_is_not_a_duplicate_stage_marker(context):
    from gpx_recipe.intent import natural_atom_only_cards
    for request in ['Use one stage. Stage 1: refine only p0:O1 Uiso.','Use exactly one stage: refine only p0:O1 Uiso for 4 cycles.']:
        cards=natural_atom_only_cards(request,context,1)
        assert cards[0]['literal_free']==[{'kind':'atom','phase':'p0','atom':'O1','flags':'U'}]


@pytest.mark.parametrize('text',["Use one stage: don't refine only p0:O1 Uiso.",'Use one stage. Stage 1: refine background then refine only p0:O1 Uiso.'])
def test_natural_only_parser_does_not_drop_prior_operations_or_negation(context,text):
    from gpx_recipe.intent import natural_atom_only_cards
    assert natural_atom_only_cards(text,context,1)==[]


def test_source_normalization_facts_are_explicit_and_not_guessed_for_missing_values():
    from gpx_recipe.planner import build_payload
    ctx={'model_signature':'x','phases':[{'key':p,'name':p,'histograms':['h'],'phase_scales':[{'histogram':'h0','value':v}],'atoms':[]} for p,v in [('p0',3.),('p1',1.)]],'histograms':[{'key':'h0','name':'h'}]}
    facts=build_payload('Normalize',ctx)['histogram_capabilities'][0]['source_starting_normalization']
    assert facts['raw_hap_scale_sum']==4. and facts['normalized_starting_hap_scales']=={'p0':.75,'p1':.25}
    del ctx['phases'][0]['phase_scales']
    assert 'source_starting_normalization' not in build_payload('Normalize',ctx)['histogram_capabilities'][0]


def test_leading_numeric_cycle_limit_is_bound_even_without_a_free_list(context):
    cards=literal_stage_cards('Stage 1: at most 4 cycles. Refine the existing background.\nStage 2: maximum of 6 cycles; keep the same model.',context)
    original={'questions':[],'unsupported':[],'stages':[{'cycles':5,'free':[]},{'cycles':6,'free':[]}]}
    before=copy.deepcopy(original);errors=check_literal_free(original,cards)
    assert original==before
    assert errors==[{'stage':1,'model_cycles':5,'request_cycles':4}]


@pytest.mark.parametrize('clause',['Do not use at most 4 cycles.','"at most 4 cycles" is old metadata.','At most 4 cycles if the fit converges; otherwise ask.','Refine the background; use the old 4-cycle result only for comparison.'])
def test_conditional_quoted_or_nonleading_cycles_are_not_invented(context,clause):
    cards=literal_stage_cards('Stage 1: '+clause,context)
    assert cards and 'literal_cycles' not in cards[0]
