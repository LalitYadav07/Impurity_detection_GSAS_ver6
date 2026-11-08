"""All compatible labels receive gradients; ranking accuracy cannot pick epochs."""
import copy
import math

import pytest
import numpy as np

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_compatible_objective import OBJECTIVE,SELECTION,terms,checkpoint_key,validate_policy


def test_single_label_matches_cross_entropy_and_ambiguous_aliases_all_receive_gradients():
    logits=torch.tensor([[8.,-2.,-3.]],requires_grad=True);allowed=torch.tensor([[True,True,False]])
    loss,_=terms(logits,allowed)
    expected=-torch.nn.functional.log_softmax(logits,dim=1)[0,:2].mean()
    assert torch.equal(loss[0],expected)
    loss.mean().backward()
    assert logits.grad[0,1]<-.49 and logits.grad[0,0]>.49 and logits.grad[0,2]>0
    single=torch.tensor([[False,True,False]])
    assert torch.equal(terms(logits,single)[0],torch.nn.functional.cross_entropy(logits,torch.tensor([1]),reduction='none'))


def test_collapse_onto_one_alias_is_worse_than_retaining_both():
    labels=torch.tensor([[True,True,False]])
    collapsed=terms(torch.tensor([[10.,-10.,-10.]]),labels)[0].item()
    balanced=terms(torch.tensor([[5.,5.,-10.]]),labels)[0].item()
    assert collapsed>9. and balanced==pytest.approx(math.log(2),abs=1e-6)
    all_allowed=torch.ones((1,3),dtype=torch.bool)
    assert terms(torch.zeros((1,3)),all_allowed)[0].item()==pytest.approx(math.log(3))


def test_loss_is_label_permutation_and_logit_offset_invariant():
    values=torch.tensor([[2.,.5,-1.],[.2,4.,3.]])
    labels=torch.tensor([[True,False,True],[False,True,True]])
    baseline=terms(values,labels)[0]
    assert torch.allclose(terms(values+100,labels)[0],baseline,rtol=0,atol=2e-6)
    assert torch.equal(terms(values[:,[2,0,1]],labels[:,[2,0,1]])[0],baseline)


@pytest.mark.parametrize('kind',['missing','wrong_shape','nonboolean','nonfinite'])
def test_invalid_targets_or_scores_cannot_generate_a_loss(kind):
    values=torch.zeros((2,3));allowed=torch.ones((2,3),dtype=torch.bool)
    if kind=='missing':allowed[0]=False
    elif kind=='wrong_shape':allowed=allowed[:1]
    elif kind=='nonboolean':allowed=allowed.float()
    else:values[0,0]=float('nan')
    with pytest.raises(ValueError):terms(values,allowed)


def metric(macro,loss,top5=1.):
    return dict(loss_method=OBJECTIVE,macro_compatible_nll=macro,loss=loss,macro_top5=top5,top5=top5)


def test_epoch_selection_rejects_alias_collapse_even_when_top5_is_better():
    collapsed=metric(8.,4.,top5=1.)
    balanced=metric(1.,.8,top5=.8)
    assert checkpoint_key(balanced)<checkpoint_key(collapsed)
    assert checkpoint_key(dict(macro_top5=.9,top5=.8,loss=.2))==(-.9,-.8,.2)
    with pytest.raises(ValueError):checkpoint_key(dict(collapsed,loss_method='invented'))


def test_macro_nll_uses_every_compatible_label_in_one_forward_pass():
    from gpx_recipe.preparation.structural_attention_metrics import evaluate
    from gpx_recipe.preparation.structural_training_data import feature_rows
    values=np.zeros((2,3,64),np.float32);values[1,0,0]=1
    logits=torch.tensor([[3.,1.,-2.],[0.,1.,2.]])
    labels=np.array([[True,True,False],[False,True,True]])
    class Fixed(torch.nn.Module):
        calls=0
        def forward(self,x):
            self.calls+=1;return logits[x[:,0,0].long()]
    model=Fixed()
    result=evaluate(model,feature_rows(values),labels,np.arange(2),np.zeros((3,1),np.float32),
        np.ones((3,1),np.float32),np.array([1,2,3],np.int32))
    expected=-torch.log_softmax(logits,dim=1).numpy()
    per_class=[expected[0,0],expected[:,1].mean(),expected[1,2]]
    assert model.calls==1 and result['macro_compatible_nll']==pytest.approx(np.mean(per_class))
    assert result['loss']==pytest.approx(np.mean([expected[0,:2].mean(),expected[1,1:].mean()]))
    assert result['per_class']['2']['compatible_nll']==pytest.approx(per_class[1])


def test_saved_policy_cannot_change_objective_or_choose_test_metrics():
    history=[dict(epoch=1,checkpoint_eligible=False,validation=metric(.1,.1)),
        dict(epoch=2,checkpoint_eligible=True,validation=metric(.8,.7)),
        dict(epoch=3,checkpoint_eligible=True,validation=metric(.6,.7))]
    d=dict(training_objective=OBJECTIVE,checkpoint_selection=SELECTION,epochs=3,selected_epoch=3,
        history=history,test=metric(.01,.01))
    validate_policy(d)
    for key,value in [('training_objective','invented'),('checkpoint_selection','top5'),('selected_epoch',1)]:
        changed=copy.deepcopy(d);changed[key]=value
        with pytest.raises(ValueError):validate_policy(changed)
    validate_policy(dict(architecture='legacy'))
