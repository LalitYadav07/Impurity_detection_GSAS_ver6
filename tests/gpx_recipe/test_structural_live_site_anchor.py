import copy
from types import SimpleNamespace

import numpy as np
import pytest

import gpx_recipe.structural_live.site_anchor as module


def phase(z,label='O1',identity=11):
    return {'General':{'AtomPtrs':[3,1,7,9],'SGData':{'SGOps':[(np.eye(3),np.zeros(3))],'SGCen':[np.zeros(3)],'SGInv':False}},
            'Atoms':[[label,'O','',.1,.2,z,1.,'1',1,'I',.01,identity]]}


def fixture(monkeypatch):
    parent=phase(.2713666666667);child=phase(.27137)
    child['ISODISTORT']={'G2parentCoords':np.array([[.1,.2,.27137]]),
        'G2VarList':[SimpleNamespace(name='dAz',atom=11)],'G2coordOffset':[.27137]}
    monkeypatch.setattr(module,'cif_precision',lambda _:({'O1':[(.1,5e-6,0.),(.2,5e-6,0.),(.27137,5e-6,0.)]},{}))
    return parent,child,SimpleNamespace(SytSym=lambda *args:('1',1))


def test_anchor_restores_exact_source_and_native_mode_zero(monkeypatch):
    parent,child,spc=fixture(monkeypatch);before=copy.deepcopy(parent['Atoms'])
    receipt=module.anchor_child(parent,child,np.eye(3),np.zeros(3),'child.cif',spc)
    assert parent['Atoms']==before
    assert child['Atoms'][0][5]==parent['Atoms'][0][5]
    assert child['ISODISTORT']['G2coordOffset']==[parent['Atoms'][0][5]]
    np.testing.assert_allclose(child['ISODISTORT']['G2parentCoords'][0],parent['Atoms'][0][3:6],atol=1e-12)
    assert receipt['maximum_shift']<5e-6


def test_two_parent_orbits_inside_interval_remain_ambiguous(monkeypatch):
    parent,child,spc=fixture(monkeypatch)
    parent['Atoms'].append(phase(.271369,label='O2',identity=22)['Atoms'][0])
    before=copy.deepcopy(child['Atoms'])
    with pytest.raises(ValueError,match='unique parent orbit'):
        module.anchor_child(parent,child,np.eye(3),np.zeros(3),'child.cif',spc)
    assert child['Atoms']==before


def test_outside_printed_precision_does_not_relax_orbit_matching(monkeypatch):
    parent,child,spc=fixture(monkeypatch);parent['Atoms'][0][5]=.27135
    with pytest.raises(ValueError,match='unique parent orbit'):
        module.anchor_child(parent,child,np.eye(3),np.zeros(3),'child.cif',spc)


def test_multiplicity_change_is_rejected(monkeypatch):
    parent,child,_=fixture(monkeypatch)
    spc=SimpleNamespace(SytSym=lambda xyz,sg:('1',2 if xyz[2]==.27137 else 1))
    with pytest.raises(ValueError,match='multiplicity'):
        module.anchor_child(parent,child,np.eye(3),np.zeros(3),'child.cif',spc)


def test_native_operation_roundoff_does_not_change_special_site_multiplicity(monkeypatch):
    parent=phase(1e-8);child=phase(0.)
    monkeypatch.setattr(module,'cif_precision',lambda _:({'O1':[(.1,5e-6,0.),(.2,5e-6,0.),(0.,0.,0.)]},{}))
    spc=SimpleNamespace(SytSym=lambda xyz,sg:('special',1) if xyz[2]==0 else ('general',2))
    receipt=module.anchor_child(parent,child,np.eye(3),np.zeros(3),'child.cif',spc)
    assert child['Atoms'][0][5]==0.
    assert 2 in receipt['sites'][0]['operation_roundoff_preserved_axes']
