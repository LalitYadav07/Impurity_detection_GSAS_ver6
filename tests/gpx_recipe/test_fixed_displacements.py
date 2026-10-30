import copy
import numpy as np
import pytest

from gpx_recipe.symmetry.displacements import metric,matrix,transform_uij,expand_magnetic_phase,transfer_fixed_displacements,verify_fixed_displacements


def six(u):return [u[i,j] for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]


def cell_from_metric(g):
    lengths=np.sqrt(np.diag(g))
    angles=np.rad2deg(np.arccos([g[1,2]/(lengths[1]*lengths[2]),g[0,2]/(lengths[0]*lengths[2]),g[0,1]/(lengths[0]*lengths[1])]))
    return [*lengths,*angles]


@pytest.mark.parametrize('basis',[
    np.eye(3),np.diag([2.,1.,1.]),np.diag([-1.,1.,1.]),
    [[1,1,0],[0,1,1],[0,0,1]],[[1/2,0,0],[0,1,0],[0,0,1/3]],
])
def test_ellipsoid_eigenvalues_and_debye_waller_exponents_survive_general_basis_changes(basis):
    cell=[5.,6.,7.,75.,89.,112.];g=metric(cell);basis=np.array(basis,dtype=float)
    child_cell=cell_from_metric(basis@g@basis.T)
    a=np.linalg.cholesky(g).T
    cart=np.array([[.003,.0003,.0002],[.0003,.006,-.0004],[.0002,-.0004,.009]])
    cov=np.linalg.inv(a)@cart@np.linalg.inv(a).T
    d=np.sqrt(np.diag(np.linalg.inv(g)));u=six(cov/np.outer(d,d))
    result=transform_uij(u,cell,child_cell,basis,np.eye(3))
    new_g=metric(child_cell);new_d=np.sqrt(np.diag(np.linalg.inv(new_g)))
    new_cov=matrix(result)*np.outer(new_d,new_d)
    new_a=np.linalg.cholesky(new_g).T;new_cart=new_a@new_cov@new_a.T
    assert np.linalg.eigvalsh(new_cart)==pytest.approx(np.linalg.eigvalsh(cart),abs=1e-13)
    # Independently compare the physical scattering attenuation for each hkl.
    for h in np.random.default_rng(371).normal(size=(25,3)):
        new_h=basis@h
        assert new_h@new_cov@new_h==pytest.approx(h@cov@h,abs=1e-13)


def test_supercell_does_not_scale_a_cartesian_isotropic_ellipsoid_by_its_determinant():
    result=transform_uij([.01,.01,.01,0,0,0],[5,6,7,90,90,90],[10,6,7,90,90,90],np.diag([2,1,1]),np.eye(3))
    assert result==pytest.approx([.01,.01,.01,0,0,0],abs=1e-13)


def test_parent_symmetry_rotation_precedes_the_child_basis_change():
    cell=[5,5,13,90,90,120];basis=np.array([[1,1,0],[0,1,0],[0,0,2]],dtype=float)
    rotation=np.array([[0,-1,0],[1,-1,0],[0,0,1]],dtype=float)
    g=metric(cell);child=cell_from_metric(basis@g@basis.T)
    u=[.004,.008,.012,.001,.0005,.0007]
    out=transform_uij(u,cell,child,basis,rotation)
    d=np.sqrt(np.diag(np.linalg.inv(g)));new_d=np.sqrt(np.diag(np.linalg.inv(metric(child))))
    parent_cov=matrix(u)*np.outer(d,d);child_cov=matrix(out)*np.outer(new_d,new_d)
    for h in np.random.default_rng(182).normal(size=(25,3)):
        old_h=rotation.T@np.linalg.inv(basis)@h
        assert h@child_cov@h==pytest.approx(old_h@parent_cov@old_h,abs=1e-13)


def phases():
    cell=[5,5,13,90,90,120]
    parent=dict(General=dict(AtomPtrs=[3,1,7,9],Cell=[False,*cell,1.],SGData=dict(SGOps=[(np.eye(3),np.zeros(3))])),
                Atoms=[['Fe','Fe',' ',0,0,.35,1.,'3',2,'A',0.,.003,.003,.005,.0015,0.,0.,123]])
    child=copy.deepcopy(parent);child['General']['AtomPtrs']=[3,1,10,12]
    child['Atoms']=[parent['Atoms'][0][:7]+[1.,0.,0.]+parent['Atoms'][0][7:]]
    child['Atoms'][0][2]='M';child['Atoms'][0][-1]=456
    return parent,child


def test_fixed_anisotropy_keeps_its_model_and_is_rechecked_against_the_original_parent():
    parent,child=phases();original=copy.deepcopy(parent)
    receipt=transfer_fixed_displacements(parent,child,['0:1+0,0,0'],np.eye(3))
    assert child['Atoms'][0][12]=='A'
    assert child['Atoms'][0][14:20]==pytest.approx(parent['Atoms'][0][11:17])
    assert parent['Atoms']==original['Atoms'] and parent['General']['Cell']==original['General']['Cell']
    assert verify_fixed_displacements(parent,child,receipt,np.eye(3))
    child['Atoms'][0][14]+=.001
    with pytest.raises(ValueError,match='ellipsoid changed'):verify_fixed_displacements(parent,child,receipt,np.eye(3))


def test_fixed_isotropic_model_remains_isotropic_and_audited():
    parent,child=phases();parent['Atoms'][0][9]='I';parent['Atoms'][0][10]=.012
    child['Atoms'][0][12]='I'
    receipt=transfer_fixed_displacements(parent,child,['0:1'],np.eye(3))
    assert child['Atoms'][0][12:14]==['I',.012]
    assert verify_fixed_displacements(parent,child,receipt,np.eye(3))
    child['Atoms'][0][13]=.025
    with pytest.raises(ValueError,match='ellipsoid changed'):verify_fixed_displacements(parent,child,receipt,np.eye(3))


def test_topology_expansion_does_not_change_the_original_or_use_an_isotropic_physical_model():
    parent,child=phases();before=copy.deepcopy(parent['Atoms'])
    def transformer(topology,new,*args):
        assert topology['Atoms'][0][9]=='I'
        assert parent['Atoms'][0][9]=='A'
        new['Atoms'][0][12]='I'
        return new,['0:1']
    expanded,codes=expand_magnetic_phase(parent,child,np.eye(3),[0,0,0],[0,0,0],transformer)
    assert expanded['Atoms'][0][12]=='A'
    receipt=transfer_fixed_displacements(parent,expanded,codes,np.eye(3))
    assert verify_fixed_displacements(parent,expanded,receipt,np.eye(3))
    assert parent['Atoms']==before


def test_fixed_tensor_transfer_rejects_releases_bad_mapping_and_nonphysical_tensors():
    parent,child=phases();child['Atoms'][0][2]='MU'
    with pytest.raises(ValueError,match='held'):transfer_fixed_displacements(parent,child,['0:1'],np.eye(3))
    child['Atoms'][0][2]='M'
    with pytest.raises(ValueError,match='mapping'):transfer_fixed_displacements(parent,child,['-1:1'],np.eye(3))
    parent['Atoms'][0][11]=-.01
    with pytest.raises(ValueError,match='nonphysical'):transfer_fixed_displacements(parent,child,['0:1'],np.eye(3))


def test_installed_gsas_expansion_with_anisotropic_atoms_preserves_physical_ellipsoids(tmp_path):
    G=pytest.importorskip('GSASII.GSASIIscriptable')
    from GSASII import GSASIIlattice as L,GSASIIspc as S
    project=G.G2Project(newgpx=str(tmp_path/'synthetic.gpx'))
    phase=project.add_phase(phasefile=None,phasename='Synthetic parent',spacegroup='R -3 c',cell=[5,5,13,90,90,120])
    p,_=phases();phase.data['Atoms']=p['Atoms'];G.SetupGeneral(phase.data,None)
    child=copy.deepcopy(phase.data);basis=np.diag([1.,1.,2.])
    child['General']['SGData']=S.SpcGroup('P 1')[1]
    child['General']['Cell'][1:]=list(L.TransformCell(phase.data['General']['Cell'][1:7],basis))
    child,codes=expand_magnetic_phase(phase.data,child,basis,np.zeros(3),np.zeros(3),L.TransformPhase)
    for atom in child['Atoms']:atom[2]='M'
    receipt=transfer_fixed_displacements(phase.data,child,codes,basis)
    assert len(receipt['atoms'])>1
    assert verify_fixed_displacements(phase.data,child,receipt,basis)
    cia=child['General']['AtomPtrs'][3]
    gs=L.cell2GS(child['General']['Cell'][1:7]);amat=L.cell2AB(child['General']['Cell'][1:7])[0]
    for atom in child['Atoms']:
        ueq,nonpositive=L.Uij2Ueqv(atom[cia+2:cia+8],gs,amat)
        assert not nonpositive
        assert ueq==pytest.approx((.003+.003+.005)/3,abs=1e-12)
