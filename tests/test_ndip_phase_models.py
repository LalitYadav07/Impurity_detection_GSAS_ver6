import copy

from scripts.ndip_phase_models import same_composition_warnings


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
