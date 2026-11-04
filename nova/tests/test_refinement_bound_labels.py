from radar_pd_nova.refinement_controller import recipe_cards


def test_oxygen_displacement_bounds_show_phase_atom_and_physical_units():
    context = {'phases': [{'key': 'p5', 'name': 'NMC333', 'histograms': ['h'],
                          'atoms': [{'label': 'O1', 'element': 'O', 'adp_type': 'I'}]}],
               'histograms': [{'key': 'h0', 'name': 'h',
                               'background': {'function': 'chebyschev-1', 'coefficients': 6}}]}
    plan = {'schema': 'radar-gpx-recipe/v1', 'model_signature': 'model',
            'stages': [{'id': 'oxygen', 'title': 'Oxygen', 'cycles': 5,
                        'free': [{'kind': 'atom', 'phase': 'p5', 'atom': 'O1', 'flags': 'U'}],
                        'constraints': [], 'bounds': [{'phase': 'p5', 'atom': 'O1',
                            'parameter': 'U', 'lower': 0, 'upper': 0.05}]}]}
    card = recipe_cards(plan, context)[0]
    assert card['edits'] == ['NMC333 / O1: 0 ≤ Uiso ≤ 0.05 Å²']
    assert 'p5' not in ' '.join(card['edits'])


def test_stage_dependencies_are_visible_before_approval():
    context={'phases':[], 'histograms':[{'key':'h0','name':'pattern'}]}
    stages=[dict(id='background',title='Background fit',cycles=5,
                 free=[{'kind':'background','histogram':'h0'}],constraints=[]),
            dict(id='later',title='Dependent fit',cycles=5,
                 free=[{'kind':'background','histogram':'h0'}],constraints=[],
                 depends_on=['BACKGROUND'])]
    cards=recipe_cards({'stages':stages},context)
    assert cards[1]['edits']==['Requires successful stages: Background fit']
