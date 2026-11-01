"""Synthetic raw and symmetry-expanded controls for reviewed CIF occupancy."""
from pathlib import Path
import json
import math
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cif_occupancy import (AUTH_ENV, load_authorized_structure, load_structure,
    make_review, policy_hash, sidecar, write_review, derive_review)
from test_cif_occupancy_policy_characterization import HEADER


def file(tmp_path, second='0.40066', *, first='0.6', sym=False):
    text=HEADER
    if sym:
        text=text.replace("'P 1'", "'P -1'").replace('_space_group_IT_number 1','_space_group_IT_number 2').replace("'x,y,z'", "'x,y,z'\n'-x,-y,-z'")
    xyz='0.1 0.2 0.3' if sym else '0 0 0'
    other='-0.1 -0.2 -0.3' if sym else xyz
    text+=f'C1 C {xyz} {first}\nLi1 Li {other} {second}\n'
    p=tmp_path/'control.cif';p.write_text(text);return p


def test_default_valid_unchanged(tmp_path):
    from pymatgen.core import Structure
    p=file(tmp_path,'0.4')
    baseline=Structure.from_file(p)
    actual=load_structure(p)
    assert actual==baseline


def test_default_rejects_small_overfill(tmp_path):
    with pytest.raises(ValueError):load_structure(file(tmp_path))


@pytest.mark.parametrize('sym',[False,True])
def test_review_preserves_complete_source_and_expanded_sites(tmp_path,sym):
    p=file(tmp_path,sym=sym);raw=p.read_bytes()
    review=make_review(raw,.001,'Synthetic source reviewed')
    structure=load_structure(p,review)
    assert len(structure)==(2 if sym else 1)
    assert all(sum(site.species.values())==pytest.approx(1.00066,abs=1e-13) for site in structure)
    assert all(site.species.get_el_amt_dict()=={'C':.6,'Li':.40066} for site in structure)
    assert len(review['affected_sites'])==len(structure)
    assert p.read_bytes()==raw


@pytest.mark.parametrize('bad',['-0.1','nan','inf','?','.','[0.4]'])
def test_raw_invalid_values_rejected_before_sanitizing(tmp_path,bad):
    p=file(tmp_path,bad)
    with pytest.raises(ValueError):load_structure(p)
    with pytest.raises(ValueError):make_review(p.read_bytes(),.001,'Synthetic review')


@pytest.mark.parametrize('limit',[True,None,'0.001',-1,math.nan,math.inf,.001001])
def test_allowance_must_be_finite_scalar_within_cap(tmp_path,limit):
    with pytest.raises(ValueError):make_review(file(tmp_path).read_bytes(),limit,'Synthetic review')


def test_insufficient_and_large_overfill_rejected(tmp_path):
    with pytest.raises(ValueError):make_review(file(tmp_path).read_bytes(),.0005,'Synthetic review')
    with pytest.raises(ValueError):make_review(file(tmp_path,'0.4011').read_bytes(),.001,'Synthetic review')
    with pytest.raises(ValueError):make_review(file(tmp_path,'1.2').read_bytes(),.001,'Synthetic review')


def test_zero_component_stays_zero_and_zero_only_site_disappears(tmp_path):
    p=file(tmp_path,'0',first='.8');raw=p.read_bytes()
    review=make_review(raw,.001,'Synthetic review')
    structure=load_structure(p,review)
    assert structure.composition.get_el_amt_dict()=={'C':.8}
    p.write_text(p.read_text()+'O1 O 0.2 0.3 0.4 0\n')
    review=make_review(p.read_bytes(),.001,'Synthetic review')
    assert len(load_structure(p,review))==1


def test_source_and_policy_hash_changes_rejected(tmp_path):
    p=file(tmp_path);review=make_review(p.read_bytes(),.001,'Synthetic review')
    p.write_text(p.read_text().replace('0.40066','0.40055'))
    with pytest.raises(ValueError,match='hash'):load_structure(p,review)
    review['maximum_site_excess']=.0008
    with pytest.raises(ValueError,match='policy hash'):load_structure(p,review)


def test_archive_declaration_alone_never_activates(tmp_path,monkeypatch):
    monkeypatch.delenv(AUTH_ENV,raising=False)
    p=file(tmp_path);review=make_review(p.read_bytes(),.001,'Synthetic review');write_review(p,review)
    with pytest.raises(ValueError):load_authorized_structure(p)


def test_activated_policy_and_lattice_derivative(tmp_path,monkeypatch):
    p=file(tmp_path);review=make_review(p.read_bytes(),.001,'Synthetic review');write_review(p,review)
    ledger=tmp_path/'activation.json'
    ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{review['policy_sha256']:review}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger))
    assert sum(load_authorized_structure(p)[0].species.values())==pytest.approx(1.00066)
    new=tmp_path/'nudged.cif';new.write_text(p.read_text().replace('_cell_length_a 4','_cell_length_a 4.01'))
    derive_review(p,new)
    assert load_authorized_structure(new).lattice.a==pytest.approx(4.01)
    new.write_text(new.read_text().replace('0.40066','0.4'))
    with pytest.raises(ValueError):load_authorized_structure(new)


def test_derived_atoms_cannot_change(tmp_path,monkeypatch):
    p=file(tmp_path);review=make_review(p.read_bytes(),.001,'Synthetic review');write_review(p,review)
    ledger=tmp_path/'activation.json';ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{review['policy_sha256']:review}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger));new=tmp_path/'bad.cif';new.write_text(p.read_text().replace('0.40066','0.4'))
    with pytest.raises(ValueError,match='changed source'):derive_review(p,new)


def test_symmetry_change_is_not_a_lattice_derivative(tmp_path,monkeypatch):
    p=file(tmp_path,sym=False)
    p.write_text(p.read_text().replace('0 0 0', '0.1 0.2 0.3'))
    review=make_review(p.read_bytes(),.001,'Synthetic review');write_review(p,review)
    ledger=tmp_path/'activation.json';ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{review['policy_sha256']:review}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger))
    changed=p.read_text().replace("'P 1'", "'P -1'").replace('_space_group_IT_number 1','_space_group_IT_number 2').replace("'x,y,z'", "'x,y,z'\n'-x,-y,-z'")
    new=tmp_path/'bad.cif';new.write_text(changed)
    with pytest.raises(ValueError,match='non-cell fields or symmetry'):derive_review(p,new)


def test_self_consistent_hash_cannot_forge_review_evidence(tmp_path):
    p=file(tmp_path);review=make_review(p.read_bytes(),.001,'Synthetic review')
    review['affected_sites']=[];review['composition']={'C':999}
    review['policy_sha256']=policy_hash(review)
    with pytest.raises(ValueError,match='evidence'):load_structure(p,review)


def test_forged_derived_signature_cannot_borrow_activated_symmetry(tmp_path,monkeypatch):
    p=file(tmp_path);root=make_review(p.read_bytes(),.001,'Synthetic review')
    ledger=tmp_path/'activation.json';ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{root['policy_sha256']:root}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger))
    changed=p.read_text().replace("'P 1'", "'P -1'").replace('_space_group_IT_number 1','_space_group_IT_number 2').replace("'x,y,z'", "'x,y,z'\n'-x,-y,-z'")
    other=tmp_path/'forged.cif';other.write_text(changed)
    forged=make_review(other.read_bytes(),.001,'Synthetic review')
    forged.update(root_policy_sha256=root['policy_sha256'],nuclear_model_signature=root['nuclear_model_signature'])
    forged['policy_sha256']=policy_hash(forged)
    sidecar(other).write_text(json.dumps(forged))
    with pytest.raises(ValueError,match='nuclear model/symmetry'):load_authorized_structure(other)


def test_strict_path_preserves_coincident_component_the_parser_would_drop(tmp_path):
    p=file(tmp_path,'0.4').read_text().replace('C1 C','O1 O').replace('Li1 Li','H1 H')
    source=tmp_path/'oxygen_hydrogen.cif';source.write_text(p)
    assert load_structure(source).composition.get_el_amt_dict()=={'O':.6,'H':.4}


@pytest.mark.parametrize('symbol',['Xxx','11B','7Li'])
def test_unsupported_dummy_or_isotope_is_rejected(tmp_path,symbol):
    p=file(tmp_path);p.write_text(p.read_text().replace('Li1 Li',f'{symbol}1 {symbol}'))
    with pytest.raises(ValueError,match='cannot preserve atom type'):load_structure(p)


@pytest.mark.parametrize('symbol',['D','T'])
def test_named_isotopes_remain_distinct_from_hydrogen(tmp_path,symbol):
    from cif_occupancy import species_inventory
    p=file(tmp_path,'0.4').read_text().replace('C1 C','H1 H').replace('Li1 Li',f'{symbol}1 {symbol}')
    source=tmp_path/'isotopes.cif';source.write_text(p)
    assert species_inventory(load_structure(source))=={'H':.6,symbol:.4}


def test_separate_oxidation_table_is_preserved(tmp_path):
    from pymatgen.core import Structure
    p=file(tmp_path,'0.4').read_text().replace('C1 C','Fe1 Fe').replace('Li1 Li','O1 O')
    p += 'loop_\n_atom_type_symbol\n_atom_type_oxidation_number\nFe 3\nO -2\n'
    source=tmp_path/'charged.cif';source.write_text(p)
    assert load_structure(source).composition==Structure.from_file(source).composition


def test_rounded_special_position_uses_existing_site_equivalence_tolerance(tmp_path):
    # Decimal thirds in an ordinary hexagonal special position must not
    # produce several near-coincident atom images under symmetry expansion.
    text=HEADER.replace('_cell_length_c 4','_cell_length_c 6').replace('_cell_angle_gamma 90','_cell_angle_gamma 120')
    text=text.replace("'P 1'", "'P 6/m m m'").replace('_space_group_IT_number 1','_space_group_IT_number 191')
    text=text.replace("loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n",'')
    source=tmp_path/'hexagonal.cif';source.write_text(text+'C1 C 0.3333 0.6667 0 1\n')
    structure=load_structure(source)
    assert len(structure)==2
    assert structure.composition.get_el_amt_dict()=={'C':2.0}


@pytest.mark.parametrize('value',['1e-9','1e-15','1e-400','-1e-400'])
def test_tiny_positive_is_not_silently_clamped(tmp_path,value):
    p=file(tmp_path,value,first='.6')
    with pytest.raises(ValueError,match='preservation|underflows'):load_structure(p)
