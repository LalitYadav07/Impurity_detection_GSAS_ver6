"""Prefer declared groups; retain native constructions for special states."""
from gpx_recipe.preparation.structural_ranker import _class_routes,_preferred_routes


def test_parent_null_controls_do_not_promote_every_child_before_other_ranked_families():
    classes=[166,12,6];groups={'parent':166,'monoclinic':12,'general':1}
    complete=_class_routes(classes,groups,{'166':['general','monoclinic','parent'],
        '12':['general','monoclinic'],'6':['general']})
    preferred=_preferred_routes(classes,groups,complete)
    assert preferred=={'166':['parent'],'12':['monoclinic'],'6':['general']}
    assert complete['166']==['general','monoclinic','parent']


def test_a_family_without_an_explicit_construction_keeps_all_capable_models():
    groups={'one':1,'two':2};routes={'6':['one','two']}
    assert _preferred_routes([6],groups,routes)==routes
