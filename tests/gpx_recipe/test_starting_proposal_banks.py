import json
import numpy as np
import pytest

from test_structural_live_offline_start_bank import source
from gpx_recipe.structural_live.optimization import offline_native_start_bank as banks
from gpx_recipe.structural_live.optimization import starting_proposals as proposals


def generated(tmp_path,monkeypatch):
    s,_,o,_=source(monkeypatch)
    path=tmp_path/'initial.npz'
    receipt=banks._construct(s,path,policy='joint_cell_v1',
        proposal_renderer=lambda point:s.native(point)+1e-12*np.sin(o.x*7))
    return s,path,receipt


def test_complete_proposals_cannot_be_read_or_remembered_as_native(tmp_path,monkeypatch):
    s,path,receipt=generated(tmp_path,monkeypatch)
    assert receipt['native_curves']==0 and receipt['starting_curves']==185
    assert receipt['interpolation'] is True
    assert receipt['starting_curves']+receipt['chemically_infeasible_points']==receipt['proposed_points']
    with pytest.raises(ValueError,match='roles cannot be interchanged'):
        banks.OfflineNativeStartBank(s,path)
    with pytest.raises(ValueError,match='native-bank authority'):
        banks._remember_native(s,path)
    assert not banks.native_in_this_process(s,path)


def test_saved_flags_do_not_establish_owned_proposal_origin(tmp_path,monkeypatch):
    s,path,_=generated(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='owned construction'):
        proposals.StartingProposalBank(s,path)
    proposals._remember(s,path)
    bank=proposals.StartingProposalBank(s,path)
    assert len(bank.points)==185
    assert bank.diagnostics()['curve_role']=='starting_proposals'
    assert not bank.curves.flags.writeable


def test_changed_files_and_forked_process_do_not_inherit_proposal_origin(tmp_path,monkeypatch):
    s,path,_=generated(tmp_path,monkeypatch)
    proposals._remember(s,path)
    assert proposals._owned(s,path)
    parent_pid=proposals.os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(proposals.os,'getpid',lambda:parent_pid+1)
        assert not proposals._owned(s,path)
    proposals._remember(s,path)
    metadata=path.with_suffix('.json')
    receipt=json.loads(metadata.read_text());receipt['seconds']+=1.
    metadata.write_text(json.dumps(receipt))
    assert not proposals._owned(s,path)


def test_proposal_origin_cannot_be_registered_for_an_exact_bank(tmp_path,monkeypatch):
    s,_,_,_=source(monkeypatch);path=tmp_path/'native.npz';banks.build(s,path)
    with pytest.raises(ValueError,match='explicitly generated starting proposals'):
        proposals._remember(s,path)
