"""Recovery has one owner and cannot silently adopt unrelated working files."""
import json

import pytest

from gpx_recipe.preparation.structural_training_checkpoint import owned_run


def test_training_destination_retains_its_exact_identity_after_interruption(tmp_path):
    out=tmp_path/'training'
    with pytest.raises(InterruptedError):
        with owned_run(out,'a'*64):raise InterruptedError('stopped')
    with owned_run(out,'a'*64):pass
    before=(out/'training-identity.json').read_bytes()
    with pytest.raises(ValueError,match='changed'):
        with owned_run(out,'b'*64):pytest.fail('foreign work adopted')
    assert (out/'training-identity.json').read_bytes()==before


def test_a_second_writer_cannot_resume_an_active_training_destination(tmp_path):
    with owned_run(tmp_path/'training','a'*64):
        with pytest.raises(ValueError,match='active owner'):
            with owned_run(tmp_path/'training','a'*64):pytest.fail('second owner entered')


def test_old_unowned_outputs_are_not_overwritten_as_a_recovery(tmp_path):
    out=tmp_path/'training';out.mkdir();(out/'model.json').write_text('prior work')
    with pytest.raises(ValueError,match='no matching recovery identity'):
        with owned_run(out,'a'*64):pytest.fail('old result adopted')
    assert (out/'model.json').read_text()=='prior work'


def test_symlinked_recovery_directory_cannot_write_into_its_target(tmp_path):
    target=tmp_path/'target';target.mkdir();out=tmp_path/'link';out.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='owned regular directory'):
        with owned_run(out,'a'*64):pytest.fail('symlink followed')
    assert not list(target.iterdir())


def test_owned_recovery_anchor_is_data_only_and_bounded(tmp_path):
    with pytest.raises(ValueError,match='identity'):
        with owned_run(tmp_path/'invalid','not-a-digest'):pytest.fail('invalid identity')
    with owned_run(tmp_path/'training','a'*64):pass
    assert json.loads((tmp_path/'training/training-identity.json').read_text())==dict(
        schema='radar-owned-attention-training-run/v1',identity='a'*64)
