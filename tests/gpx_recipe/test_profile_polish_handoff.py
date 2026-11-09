"""The magnetic wrapper must keep its reviewed handoff after engine extraction."""
import json
import pytest
from gpx_recipe.live.contracts import file_hash
from gpx_recipe.live.profile_polish import polish


def inputs(tmp_path):
    source=tmp_path/'source.gpx';source.write_bytes(b'private synthetic placeholder')
    receipt=tmp_path/'request.json'
    receipt.write_text(json.dumps({'base_option':{'source_path':str(source),'source_sha256':file_hash(source)}}))
    return source,receipt


def test_selected_width_checkpoint_reaches_existing_magnetic_transfer(tmp_path,monkeypatch):
    from gpx_recipe.live import sample_profile_trials,prepare_reference
    source,receipt=inputs(tmp_path);selected=tmp_path/'strain.gpx';selected.write_bytes(b'selected synthetic checkpoint')
    calls=[]
    def trials(path,out):
        assert path==source and out==tmp_path
        return {'selected_for_review':'strain','trials':[{'name':'strain','path':str(selected)}]}
    def transfer(path):
        calls.append(json.loads(path.read_text()))
        return {'title':'Existing magnetic reference'}
    monkeypatch.setattr(sample_profile_trials,'sample_width_trials',trials)
    monkeypatch.setattr(prepare_reference,'prepare',transfer)
    result=polish(receipt)
    assert calls[0]['source_path']==str(selected) and calls[0]['source_sha256']==file_hash(selected)
    assert calls[0]['base_option']['source_path']==str(source)
    assert result['title']=='Profile diagnostic · Existing magnetic reference'
    assert result['profile_diagnostic']==str(tmp_path/'profile-trials.json')


def test_no_qualified_trial_cannot_start_a_magnetic_transfer(tmp_path,monkeypatch):
    from gpx_recipe.live import sample_profile_trials,prepare_reference
    source,receipt=inputs(tmp_path)
    monkeypatch.setattr(sample_profile_trials,'sample_width_trials',lambda *a:{'selected_for_review':None,'trials':[]})
    def forbidden(*a):raise AssertionError('An unqualified diagnostic cannot be transferred')
    monkeypatch.setattr(prepare_reference,'prepare',forbidden)
    with pytest.raises(ValueError,match='No physically bounded'):polish(receipt)


def test_changed_source_is_rejected_before_any_trial(tmp_path,monkeypatch):
    from gpx_recipe.live import sample_profile_trials
    source,receipt=inputs(tmp_path);source.write_bytes(b'changed source')
    def forbidden(*a):raise AssertionError('Changed source cannot enter a refinement')
    monkeypatch.setattr(sample_profile_trials,'sample_width_trials',forbidden)
    with pytest.raises(ValueError,match='Reviewed GPX changed'):polish(receipt)
