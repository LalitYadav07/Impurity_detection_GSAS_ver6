import numpy as np
import pytest

pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention_metrics import evaluate
from gpx_recipe.preparation.structural_training_data import feature_rows


def test_prior_baseline_reports_rare_family_failures_and_set_valued_hits():
    rows=feature_rows(np.zeros((4,3,64),np.float32));targets=np.zeros((4,7),bool)
    targets[0,6]=True;targets[1,0]=True;targets[2,5:]=True;targets[3]=True
    result=evaluate(None,rows,targets,np.arange(4),np.zeros((3,1),np.float32),
        np.ones((3,1),np.float32),np.arange(1,8),prior=np.arange(7,0,-1))
    assert result['top1']==result['top5']==.5 and result['loss'] is None
    assert result['per_class']['7']['top5']==pytest.approx(1/3)
    assert result['macro_top5']==pytest.approx((5+.5+1/3)/7)
    assert result['worst_class_top5']==pytest.approx(1/3)
    with pytest.raises(ValueError,match='Every scored family'):
        evaluate(None,rows,targets,np.arange(3),np.zeros((3,1),np.float32),
            np.ones((3,1),np.float32),np.arange(1,8),prior=np.arange(7,0,-1))
