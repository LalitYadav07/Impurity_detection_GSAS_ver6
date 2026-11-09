import json
import numpy as np
import pytest
pytest.importorskip('torch')
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation.structural_ranker import Ranker
from test_structural_joint_modes_native import cohort

def test_actual_label_diagnostics_do_not_change_training_weights_or_checkpoint_selection(tmp_path,monkeypatch):
    _,value=cohort(tmp_path/'source');original=training.evaluate
    def without_extra_diagnostics(*args,**kwargs):
        result=original(*args,**kwargs);result.pop('label_retention')
        for part in result.get('observation_metrics',{}).get('partitions',{}).values():part.pop('label_retention',None)
        return result
    monkeypatch.setattr(training,'evaluate',without_extra_diagnostics)
    reference=training.train(value,tmp_path/'reference',epochs=2,threads=1)
    monkeypatch.setattr(training,'evaluate',original)
    measured=training.train(value,tmp_path/'measured',epochs=2,threads=1)
    assert measured['weights_sha256']==reference['weights_sha256']
    assert measured['selected_epoch']==reference['selected_epoch']
    assert measured['mode_readout']==reference['mode_readout']
    assert measured['test']['loss']==reference['test']['loss']
    assert measured['test']['mode_validation']==reference['test']['mode_validation']
    imported=Ranker(tmp_path/'measured')
    assert imported.metadata['test']['label_retention']==measured['test']['label_retention']

@pytest.mark.parametrize('change',['class','denominator','depth','row_rate','authority'])
def test_portable_model_cannot_misreport_retention_of_native_labels(tmp_path,change):
    _,value=cohort(tmp_path/'source');training.train(value,tmp_path/'model',epochs=1,threads=1)
    path=tmp_path/'model/model.json';document=json.loads(path.read_text())
    row=document['test']['label_retention']['top5'];first=str(document['classes'][0])
    if change=='class':row['per_class'].pop(first)
    elif change=='denominator':row['per_class'][first]['target_rows']+=1
    elif change=='depth':row['requested_top_k']=1
    elif change=='row_rate':document['test']['top5']='bad'
    else:row['experimental_coverage_qualified']=True
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):Ranker(tmp_path/'model')
