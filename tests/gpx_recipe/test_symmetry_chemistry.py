import pytest
from gpx_recipe.symmetry.chemistry import validate_sites


def test_selected_oxidation_state_may_change_but_the_element_may_not():
    atoms=[dict(label='Mn1',type='Mn',occupancy=1.)]
    validate_sites(atoms,[dict(label='Mn1',form_factor='Mn+3')])
    with pytest.raises(ValueError,match='chemical element'):validate_sites(atoms,[dict(label='Mn1',form_factor='Fe+3')])


@pytest.mark.parametrize('atoms',[[],[dict(label='Mn1',type='Mn',occupancy=0.)],[dict(label='Mn1',type='Mn',occupancy=1.)]*2])
def test_missing_empty_or_ambiguous_sites_do_not_run(atoms):
    with pytest.raises(ValueError):validate_sites(atoms,[dict(label='Mn1',form_factor='Mn+3')])
