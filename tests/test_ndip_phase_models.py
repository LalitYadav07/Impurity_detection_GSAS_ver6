import copy

from scripts.ndip_phase_models import (same_composition_warnings, component_chemistry_context,
                                      resolved_component_chemistry)


def phase(formula, *, histogram='PWDR test', nuclear=True, scale=1., isotope=None):
    atoms = [[name+'1', name, '', 0., 0., 0., occupancy, '1', multiplicity, 'I', 0.01]
             for name, occupancy, multiplicity in formula]
    return {'General': {'Type': 'nuclear' if nuclear else 'magnetic', 'AtomPtrs': [3, 1, 7, 9],
                        'Isotopes': isotope or {}}, 'Atoms': atoms,
            'Histograms': {histogram: {'Use': True, 'Scale': [scale, False]}}}


def records(**phases):
    return [[['Phases', {}], *[[name, value] for name, value in phases.items()]]]


def test_polymorphs_with_different_cell_multiplicities_need_review_without_merging():
    a = phase([('Pr', 1., 2), ('Fe', 1., 2), ('As', 1., 2), ('O', 1., 2)])
    b = phase([('Pr', 1., 4), ('Fe', 1., 4), ('As', 1., 4), ('O', 1., 4)])
    source = records(warm=a, cold=b); original = copy.deepcopy(source)
    warnings = same_composition_warnings(source, labels={'warm': 'Tetragonal parent', 'cold': 'Orthorhombic model'})
    assert len(warnings) == 1
    assert warnings[0]['code'] == 'same_composition_phase_models'
    assert warnings[0]['groups'] == [{'histogram': 'PWDR test', 'phase_ids': ['warm', 'cold'],
                                     'phase_labels': ['Tetragonal parent', 'Orthorhombic model']}]
    assert 'do not establish phase coexistence' in warnings[0]['message']
    assert source == original


def test_distinct_battery_lithiation_compositions_are_not_grouped():
    source = records(lic6=phase([('Li', 1., 1), ('C', 1., 6)]),
                     lic12=phase([('Li', 0.5, 1), ('C', 1., 6)]),
                     carbon=phase([('C', 1., 4)]))
    assert same_composition_warnings(source) == []


def test_magnetic_counterpart_different_histogram_and_zero_scale_do_not_trigger():
    formula = [('Fe', 1., 2), ('O', 1., 3)]
    source = records(nuclear=phase(formula), magnetic=phase(formula, nuclear=False),
                     other_histogram=phase(formula, histogram='PWDR second'),
                     inactive=phase(formula, scale=0))
    assert same_composition_warnings(source) == []


def test_isotope_choices_are_preserved_and_unknown_atom_syntax_is_not_guessed():
    source = records(boron10=phase([('B', 1., 1)], isotope={'B': '10'}),
                     boron11=phase([('B', 1., 1)], isotope={'B': '11'}),
                     unknown=phase([('Boron_unknown', 1., 1)]))
    assert same_composition_warnings(source) == []


def test_charge_labels_do_not_change_nuclear_chemical_stoichiometry():
    source = records(neutral=phase([('Fe', 1., 2), ('O', 1., 3)]),
                     ions=phase([('Fe+3', 1., 4), ('O-2', 1., 6)]))
    assert same_composition_warnings(source)[0]['groups'][0]['phase_ids'] == ['neutral', 'ions']


def test_declared_environment_role_and_masks_do_not_change_the_native_models():
    source=records(sample=phase([('Er',1.,2),('Be',1.,2),('Ge',1.,1),('O',1.,7)]),
                   aluminium=phase([('Al',1.,4)]))
    source.append([['PWDR test',{}],['Limits',[[5,120],[10,110],[35,39]]]])
    before=copy.deepcopy(source)
    context,warnings=component_chemistry_context(source,
        dict(sample_elements=['Er','Be','Ge','O'],environment_elements=['Al']),labels={'aluminium':'Al can model'})
    roles={r['phase_id']:r['component_role'] for r in context['phase_roles']}
    assert roles=={'sample':'sample_chemistry','aluminium':'environment_only_chemistry'}
    assert warnings[0]['phase_ids']==['aluminium'] and warnings[0]['excluded_regions_present']
    assert 'bulk sample composition' in warnings[0]['message'] and 'Al can model' in warnings[0]['message']
    assert source==before


def test_alumina_sample_is_not_classified_as_an_aluminium_container():
    source=records(alumina=phase([('Al',1.,2),('O',1.,3)]),mgo=phase([('Mg',1.,1),('O',1.,1)]))
    context,warnings=component_chemistry_context(source,dict(sample_elements=['Al','Mg','O'],environment_elements=[]))
    assert {r['component_role'] for r in context['phase_roles']}=={'sample_chemistry'}
    assert warnings==[]


def test_mixed_and_overlapping_chemistry_is_not_silently_classified_as_environment():
    source=records(mixed=phase([('Al',1.,2),('O',1.,3)]),pure=phase([('Al',1.,4)]))
    context,warnings=component_chemistry_context(source,dict(sample_elements=['O'],environment_elements=['Al']))
    assert context['phase_roles'][0]['component_role']=='mixed_or_unassigned_chemistry'
    assert warnings[0]['phase_ids']==['pure']
    overlap,warnings=component_chemistry_context(source,dict(sample_elements=['O','Al'],environment_elements=['Al']))
    assert overlap['phase_roles'][1]['component_role']=='overlapping_chemistry' and warnings==[]


def test_unlinked_zero_scale_magnetic_and_unknown_components_do_not_warn():
    source=records(unused=phase([('Al',1.,4)],scale=0),magnetic=phase([('Al',1.,4)],nuclear=False),
                   unknown=phase([('Undefined',1.,4)]))
    context,warnings=component_chemistry_context(source,dict(sample_elements=['Er'],environment_elements=['Al']))
    assert warnings==[] and [r['phase_id'] for r in context['phase_roles']]==['unused']
    assert component_chemistry_context(source,None)==({},[])


def test_masks_on_another_histogram_do_not_claim_to_hide_container_evidence():
    source=records(aluminium=phase([('Al',1.,4)],histogram='PWDR first'))
    source.append([['PWDR second',{}],['Limits',[[5,120],[10,110],[35,39]]]])
    context,warnings=component_chemistry_context(source,dict(sample_elements=['Er'],environment_elements=['Al']))
    assert not warnings[0]['excluded_regions_present']
    assert 'excludes pattern' not in warnings[0]['message']


def test_executed_chemistry_requires_agreement_across_selected_datasets():
    config=dict(allowed_elements=['Er','O'],element_filter={'sample_env':{'elements':['Al']}},datasets=[{},{}])
    assert resolved_component_chemistry(config)==dict(sample_elements=['Er','O'],environment_elements=['Al'])
    config['datasets'][1]['allowed_elements']=['Tb','O']
    assert resolved_component_chemistry(config) is None
    config['datasets']=[dict(allowed_elements=[])]
    assert resolved_component_chemistry(config) is None
