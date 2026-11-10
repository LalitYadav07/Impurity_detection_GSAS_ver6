import io
import json
import numpy as np
import pytest
pytest.importorskip('torch')
from gpx_recipe.preparation import structural_history as history
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation import structural_training_transfer as transfer
from gpx_recipe.preparation.structural_ranker import Ranker
from test_structural_joint_modes_native import cohort

def test_compressed_joint_history_resumes_and_imports_without_changing_any_weight(tmp_path,monkeypatch):
    _,data=cohort(tmp_path/'source')
    reference=training.train(data,tmp_path/'reference',epochs=2,threads=1)
    monkeypatch.setattr(history,'INLINE_BYTES',1)
    original=training.write_json
    def interrupt(path,value,**kw):
        original(path,value,**kw)
        if value.get('state')=='training' and value.get('epoch')==1:raise InterruptedError('complete epoch saved')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError):training.train(data,tmp_path/'resumed',epochs=2,threads=1)
    checkpoint=tmp_path/'resumed/training-checkpoint.npz'
    with np.load(checkpoint,allow_pickle=False) as arrays:metadata=json.loads(arrays['metadata'].tobytes())
    assert metadata['history']['schema']==history.SCHEMA
    owner=json.loads((tmp_path/'resumed/training-identity.json').read_text())
    transfer.snapshot(tmp_path/'resumed',tmp_path/'export',identity=owner['identity'],epochs=2)
    _,inspected=transfer.inspect(checkpoint,identity=owner['identity'],epochs=2)
    assert isinstance(inspected['history'],list)
    monkeypatch.setattr(training,'write_json',original)
    resumed=training.train(data,tmp_path/'resumed',epochs=2,threads=1)
    assert resumed['weights_sha256']==reference['weights_sha256']
    assert resumed['history']==reference['history']
    assert resumed['test']==reference['test']
    stored=json.loads((tmp_path/'resumed/model.json').read_text())
    assert stored['history']['schema']==history.SCHEMA
    assert Ranker(tmp_path/'resumed').metadata['history']==reference['history']
