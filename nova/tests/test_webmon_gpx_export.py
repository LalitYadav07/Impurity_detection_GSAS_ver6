import copy
from types import SimpleNamespace
import numpy as np
import pytest
from radar_pd_nova.webmon.gpx_export import apply_magnetic_parameters, apply_snapshot_observations
from radar_pd_nova.webmon.contracts import file_hash, seal_posterior, verify_posterior


def test_mapped_moments_exported_and_seed_covariance_removed():
    phase = {'General': {'Type':'magnetic', 'AtomPtrs':[3,1,10,12]}, 'pId':3, 'Atoms':[[0]*14]}
    project = SimpleNamespace(data={'Covariance':{'data':{'stale':True}}}, phase=lambda name:SimpleNamespace(data=phase))
    parms = {'3::AMx:0':1., '3::AMy:0':0., '3::AMz:0':0., '3::Ax:0':.2}
    native = SimpleNamespace(x0=np.array([1.]), vary=['::constr0'], parms=parms, histo_phases=[{}, {'mag':copy.deepcopy(phase)}])
    def evaluate(values):
        native.parms.update({'::constr0':values[0], '3::AMx:0':values[0], '3::AMz:0':-values[0]})
    native.profile = evaluate
    mapped=apply_magnetic_parameters(project,native,np.array([2.7]))
    assert phase['Atoms'][0][7:10] == [2.7,0.,-2.7]
    assert mapped[0]['components'] == [2.7,0.,-2.7]
    assert project.data['Covariance']['data'] == {}


def test_export_rejects_hidden_nuclear_coupling():
    phase = {'General': {'Type':'magnetic', 'AtomPtrs':[3,1,10,12]}, 'pId':3, 'Atoms':[[0]*14]}
    native = SimpleNamespace(x0=np.array([1.]),vary=['::constr0'],parms={'3::AMx:0':1.,'3::Ax:0':.2},histo_phases=[{}, {'mag':phase}])
    def evaluate(_):native.parms['3::Ax:0']=.3
    native.profile=evaluate
    with pytest.raises(ValueError,match='fixed native parameter'):
        apply_magnetic_parameters(None,native,[2.])


def test_current_snapshot_preserves_mask_and_replaces_observations(tmp_path):
    x=np.array([1.,2.,3.]);a=[x.copy(),np.array([1.,1.,1.]),np.array([1.,0.,1.]),np.zeros(3),np.zeros(3),np.zeros(3)]
    hist=SimpleNamespace(data={'data':[{'Dummy':True},a]});project=SimpleNamespace(histograms=lambda:[hist])
    path=tmp_path/'observation.npz';np.savez(path,x=x,y=[3.,8.,5.],sigma=[2.,1.,4.],mask=[False,False,False])
    apply_snapshot_observations(project,{'observation':str(path),'observation_sha256':file_hash(path)})
    np.testing.assert_array_equal(a[1],[3.,8.,5.]);np.testing.assert_array_equal(a[2],[.25,0.,.0625])
    assert hist.data['data'][0]['Dummy'] is False


def test_posterior_export_added_to_artifact_seal(tmp_path):
    for name in ('kernel.json','kernel.npz','posterior.json','posterior.npz','inference-request.json','posterior-maximum.gpx','posterior-gpx.json'):
        (tmp_path/name).write_text(name)
    sealed=seal_posterior(tmp_path);assert 'posterior-maximum.gpx' in sealed
    assert verify_posterior(tmp_path/'posterior.json',sealed)
    (tmp_path/'posterior-maximum.gpx').write_text('changed')
    with pytest.raises(ValueError,match='changed'):verify_posterior(tmp_path/'posterior.json',sealed)
