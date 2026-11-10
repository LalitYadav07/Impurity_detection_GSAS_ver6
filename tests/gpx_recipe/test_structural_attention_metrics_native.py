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
    actual=result['label_retention']['top5']
    assert actual['any_compatible_hit_rate']==result['top5']
    assert actual['per_class']['7']['retention']==0.
    assert actual['all_compatible_retention']==.25
    with pytest.raises(ValueError,match='Every scored family'):
        evaluate(None,rows,targets,np.arange(3),np.zeros((3,1),np.float32),
            np.ones((3,1),np.float32),np.arange(1,8),prior=np.arange(7,0,-1))


def test_clean_and_noisy_diagnostics_share_exactly_one_forward_pass():
    import torch
    values=np.zeros((6,3,64),np.float32);values[:,0,0]=np.arange(6)
    rows=feature_rows(values);targets=np.eye(6,dtype=bool)
    logits=np.zeros((6,6),np.float32);logits[np.arange(6),np.arange(6)]=10
    logits[0,0]=logits[1,1]=-10

    class Fixed(torch.nn.Module):
        calls=0
        def forward(self,values):
            self.calls+=1
            return torch.from_numpy(logits)[values[:,0,0].long()]

    model=Fixed();args=(model,rows,targets,np.arange(6),np.zeros((3,1),np.float32),
                       np.ones((3,1),np.float32),np.arange(1,7))
    result=evaluate(*args,clean=np.array([True,True,False,False,False,False]))
    assert model.calls==1
    parts=result.pop('observation_metrics')['partitions']
    assert parts['clean_native']['rows']==2 and parts['clean_native']['top5']==0.
    assert parts['simulated_observation']['rows']==4 and parts['simulated_observation']['top5']==1.
    assert parts['clean_native']['label_retention']['top5']['all_compatible_retention']==0.
    assert parts['simulated_observation']['label_retention']['top5']['all_compatible_retention']==1.
    assert result==evaluate(*args)
    assert model.calls==2


def test_prior_baseline_uses_no_feature_reads_and_keeps_all_metrics():
    """A baseline must not rebuild native data that cannot affect its ranking."""
    count=137;classes=np.arange(1,8)
    targets=np.eye(7,dtype=bool)[np.arange(count)%7]
    targets[::9]=True
    indices=np.r_[np.arange(count-1,-1,-1),[0,13,13]]
    clean=np.arange(count)%4==0
    values=np.zeros((count,3,64),np.float32)
    arguments=(targets,indices,np.zeros((3,1),np.float32),np.ones((3,1),np.float32),classes)
    expected=evaluate(None,feature_rows(values),*arguments,prior=np.array([7,6,6,4,3,2,1]),clean=clean)

    class LabelsOnly:
        def __len__(self):return count
        def take(self,indices):
            raise AssertionError('A fixed class prior requested diffraction features')

    actual=evaluate(None,LabelsOnly(),*arguments,prior=np.array([7,6,6,4,3,2,1]),clean=clean)
    assert actual==expected
    assert actual['label_retention']['top5']['rows']==len(indices)
    assert actual['top5']==np.mean(targets[indices,:5].any(1))
    assert actual['label_retention']['top5']['all_compatible_retention']==np.mean(
        ~targets[indices,5:].any(1))


@pytest.mark.parametrize('indices',[
    np.array([-1]),np.array([7]),np.array([0.]),np.array([True]),np.array([[0,1]]),
    np.array([2**64-1],dtype=np.uint64)])
def test_label_only_baseline_preserves_index_validation(indices):
    values=feature_rows(np.zeros((7,3,64),np.float32))
    with pytest.raises(ValueError,match='valid structural feature rows'):
        evaluate(None,values,np.eye(7,dtype=bool),indices,np.zeros((3,1),np.float32),
            np.ones((3,1),np.float32),np.arange(1,8),prior=np.ones(7))


@pytest.mark.parametrize('prior',[np.ones(6),np.ones((1,7)),np.full(7,np.nan),
    np.full(7,np.inf),np.array(['a']*7)])
def test_label_only_baseline_rejects_invalid_class_prior(prior):
    values=feature_rows(np.zeros((7,3,64),np.float32))
    with pytest.raises(ValueError,match='finite class-prior'):
        evaluate(None,values,np.eye(7,dtype=bool),np.arange(7),np.zeros((3,1),np.float32),
            np.ones((3,1),np.float32),np.arange(1,8),prior=prior)
