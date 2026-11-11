import copy
import pytest

from gpx_recipe.phase_labels import phase_label
from gpx_recipe import guided
from radar_pd_nova.refinement_controller import recipe_cards


@pytest.mark.parametrize('name,group,expected',[
    ('Ba4SbRu3O12 (SG 166)','C 2/m','Ba4SbRu3O12 · C 2/m'),
    ('Silicate (SG P4/ncc, 130)','C 2/c','Silicate · C 2/c'),
    ('Sample (warm) (SG 166)','P -1','Sample (warm) · P -1'),
    ('Parent (SG 166)',None,'Parent (SG 166)'),
    ('Parent (SG 166)','','Parent (SG 166)'),
    ('Phase a',' P 1 ','Phase a · P 1'),
])
def test_label_uses_saved_symmetry_without_mutating_native_identity(name,group,expected):
    phase=dict(key='p0',name=name,space_group=group);original=copy.deepcopy(phase)
    assert phase_label(phase)==expected
    assert phase==original


def test_guided_selection_and_recipe_cards_show_the_same_saved_child():
    phase=dict(key='p0',name='Ba4SbRu3O12 (SG 166)',space_group='C 2/m',histograms=['PWDR one'],atoms=[])
    context=dict(model_signature='model',source={'sha256':'source'},phases=[phase],histograms=[dict(key='h0',name='PWDR one')])
    state=guided.start(context)
    state=guided.answer(context,state,'recipe','mixture')
    state=guided.answer(context,state,'model_review','reviewed')
    question=guided.next_question(context,state)
    assert question['options']==[dict(title='Ba4SbRu3O12 · C 2/m',value='p0')]
    state=guided.answer(context,state,'cell_phases',['p0'])
    plan=guided.build_plan(context,state)
    cards=recipe_cards(plan,context)
    assert any('Ba4SbRu3O12 · C 2/m' in label for card in cards for label in card['free'])
    assert phase['name']=='Ba4SbRu3O12 (SG 166)'
    assert all(t.get('phase','p0')=='p0' for s in plan['stages'] for t in s['free'])
