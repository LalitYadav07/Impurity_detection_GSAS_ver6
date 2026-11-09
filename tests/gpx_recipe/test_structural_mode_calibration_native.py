"""Shared training and portable handoffs on authored contract curves.

These tests exercise >5000 fitting pairs and real Torch optimization, not
scientific diffraction accuracy or experimental mode-detectability claims.
"""
import copy
import json

import numpy as np
import pytest

pytest.importorskip('torch')

from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation import structural_mode_calibration as intervals
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_family_targets import build
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_ranker import Ranker
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_calibration_training import packet_fixture
from test_structural_mode_projection import attach, rewrite_projection


def mapped_cohort(root):
    packet_fixture(root)
    paths=sorted(root.glob('cohort-*'))
    for path in paths:
        attach(path,names='examples')
        rewrite_projection(path,change=lambda d:d.update(native_mode_labels=[
            'R-3m[0,0,0]GM1+(a)[Ba1:c:dsp]A1(a)','unknown origin label']))
        document=json.loads((path/'examples.json').read_text())
        document['mode_projection']['metadata_sha256']=file_sha(path/'mode-projection.json')
        (path/'examples.json').write_text(json.dumps(document))
    def blocks():return (read(p,names='examples') for p in paths)
    value=assemble_pairs(list(blocks()),bins=64)
    table=build(blocks,value)
    value.update(mode_family_targets=table,mode_family_target_identity=table.metadata['identity'],
        mode_family_supervision=table.metadata)
    return value


def test_frozen_joint_model_calibrates_once_and_imports_all_mode_interval_denominators(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_calibration_training as scoring
    original_scoring=scoring.calibrate;calls=[]
    def counted(model,rows,targets,split,*args,**kwargs):
        forwards=[];handle=model.local.register_forward_hook(lambda *unused:forwards.append(1))
        try:result=original_scoring(model,rows,targets,split,*args,**kwargs)
        finally:handle.remove()
        assert len(forwards)==(int(np.count_nonzero(split==3))+63)//64
        calls.append(len(forwards));return result
    monkeypatch.setattr(scoring,'calibrate',counted)
    cohort=mapped_cohort(tmp_path/'source')
    result=training.train(cohort,tmp_path/'model',epochs=1,threads=1)
    assert len(calls)==1
    cal=result['mode_interval_calibration'];test=result['mode_interval_test']
    assert result['calibration_partition']['available']
    assert result['calibration_partition']['fitting_clean_pairs']>=5001
    assert cal['groups']==result['calibration_partition']['calibration_groups']
    assert cal['rows']==result['dataset']['partitions']['calibration']['rows']
    assert len(cal['groups'])==20 and cal['identities']['weights_sha256']==result['weights_sha256']
    assert any(row['expansion'] is not None for row in cal['conditions'])
    assert test['known_queries']==result['test']['mode_validation']['known_queries']
    assert test['rows']==result['dataset']['partitions']['test']['rows']
    imported=Ranker(tmp_path/'model')
    assert imported.metadata['mode_interval_calibration']==cal
    assert not result['mode_readout']['intervals_calibrated']
    assert not result['mode_readout']['experimental_mode_absence_qualified']
    metadata=tmp_path/'model/model.json';original=metadata.read_bytes()
    for change in ['weights','group','used_groups','denominator','radius','authority','missing_partner','test_queries']:
        changed=json.loads(original)
        if change=='weights':changed['mode_interval_calibration']['identities']['weights_sha256']='0'*64
        elif change=='group':changed['mode_interval_calibration']['groups'][0]='different-trajectory'
        elif change=='used_groups':changed['mode_interval_calibration']['used_groups_sha256']='0'*64
        elif change=='denominator':changed['mode_interval_calibration']['rows']-=1
        elif change=='radius':changed['mode_interval_calibration']['conditions'][0]['expansion']=0.
        elif change=='authority':changed['mode_interval_test']['experimental_mode_absence_qualified']=True
        elif change=='missing_partner':changed.pop('mode_interval_test')
        else:changed['mode_interval_test']['conditions'][0]['queries']-=1
        metadata.write_text(json.dumps(changed))
        with pytest.raises(ValueError):Ranker(metadata.parent)
        metadata.write_bytes(original)
    # Earlier joint models, including the already running frozen benchmark,
    # stay readable without acquiring a new calibration claim.
    old=json.loads(original);old.pop('mode_interval_calibration');old.pop('mode_interval_test')
    metadata.write_text(json.dumps(old));Ranker(metadata.parent);metadata.write_bytes(original)


def test_postfreeze_interval_policy_cannot_change_weights_or_checkpoint_selection(tmp_path,monkeypatch):
    cohort=mapped_cohort(tmp_path/'source')
    first=training.train(cohort,tmp_path/'first',epochs=1,threads=1)
    original=intervals.Calibration.__init__
    def other_rate(self,*args,**kwargs):
        kwargs['error_rate']=.25
        original(self,*args,**kwargs)
    monkeypatch.setattr(intervals.Calibration,'__init__',other_rate)
    second=training.train(cohort,tmp_path/'second',epochs=1,threads=1)
    assert first['weights_sha256']==second['weights_sha256']
    assert first['selected_epoch']==second['selected_epoch']
    assert first['test']==second['test']
    assert first['mode_interval_calibration']['error_rate']!=second['mode_interval_calibration']['error_rate']
    Ranker(tmp_path/'second')
