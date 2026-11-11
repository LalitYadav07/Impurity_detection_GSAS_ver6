import copy
from types import SimpleNamespace

from radar_pd_nova.symmetry_workspace import matching_members, model_search_text
from radar_pd_nova.symmetry_controller import SymmetryMixin


def members():
    return [
        dict(id='p1',name='P1',bns='1.1',candidate='mag:p1',seed='x-start',
             status='optimization_unresolved',has_view=False),
        dict(id='cm',name="Cm'm2'",bns='35.167',candidate='mag:cm',seed='y-start',
             status='optimization_unresolved',has_view=False),
    ]


def test_search_finds_models_inside_an_unresolved_group_without_changing_rows():
    rows=members();before=copy.deepcopy(rows)
    assert matching_members(rows,"  CM'M2' ")==[rows[1]]
    assert matching_members(rows,'35.167')==[rows[1]]
    assert matching_members(rows,'y-start')==[rows[1]]
    assert matching_members(rows,'optimization unresolved')==rows
    assert matching_members(rows,'   ')==rows
    assert matching_members(rows,'absent')==[]
    assert rows==before


def test_search_includes_the_displayed_disposition():
    assert 'rwp target not met' in model_search_text(dict(status='inadequate_fit'))
    assert 'equivalent screened start' in model_search_text(dict(status='profile_equivalent'))


def test_selecting_a_matching_failure_group_opens_the_requested_model():
    rows=members()
    state=SimpleNamespace(sx_group='unresolved',sx_search="Cm'm2'",sx_selected='p1')
    viewed=[]
    app=SimpleNamespace(server=SimpleNamespace(state=state),
        _symmetry_groups=[dict(id='unresolved',members=rows)],
        _symmetry_explicit_selection=True,
        _symmetry_plot_changed=lambda:viewed.append(state.sx_selected))
    SymmetryMixin._symmetry_group_changed(app)
    assert state.sx_members==[dict(title="Cm'm2' · mag:cm · y-start · optimization unresolved",value='cm')]
    assert state.sx_selected=='cm' and viewed==['cm']
    state.sx_search=''
    SymmetryMixin._symmetry_group_changed(app)
    assert len(state.sx_members)==2 and state.sx_selected=='cm'
