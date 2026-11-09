"""Real shared optimization/transport on authored arrays, not scientific accuracy."""
import copy
import json

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from gpx_recipe.preparation import structural_joint_modes as joint
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation import structural_attention_weights as weights
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_family_targets import build
from gpx_recipe.preparation.structural_ranker import Ranker
from gpx_recipe.preparation.structural_training_data import feature_rows
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_family_targets import mapped


def cohort(root):
    path=mapped(root);block=read(path);a={k:v.copy() for k,v in block.arrays.items()};d=copy.deepcopy(block.manifest)
    for i in np.flatnonzero(a['role']==2):
        amplitude=.01+.007*int(a['trajectory'][i]);a['parameters'][i,0]=amplitude
        a['theory_intensity'][i]=a['parent_intensity'][i]+np.exp(-((a['q']-2.2)/.05)**2)*amplitude**2*10
    np.savez_compressed(path/'pairs.npz',**a);d['pairs_sha256']=file_sha(path/'pairs.npz')
    (path/'pairs.json').write_text(json.dumps(d))
    value=assemble_pairs([read(path)],bins=128)
    table=build(lambda:iter([read(path)]),value)
    value.update(mode_family_targets=table,mode_family_target_identity=table.metadata['identity'],mode_family_supervision=table.metadata)
    return path,value


def test_shared_joint_training_publishes_bound_mode_weights_and_round_trips_inference(tmp_path):
    path,value=cohort(tmp_path/'source')
    result=training.train(value,tmp_path/'model',epochs=2,threads=1)
    assert result['architecture']==weights.MODE_ARCHITECTURE and result['mode_head_trained']
    assert result['mode_readout']['target_scale_method']==joint.readout.SCALE_METHOD
    assert result['training_objective']==joint.OBJECTIVE and result['checkpoint_selection']==joint.SELECTION
    assert all(row['validation']['mode_validation']['available'] for row in result['history'])
    assert result['test']['mode_validation']['known_rows']>0
    with np.load(tmp_path/'model/weights.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)=={'parameters','q_edges','normalization','classes','reference_features','support_prototypes',joint.ARRAY}
        assert joint.decode(arrays[joint.ARRAY],result['mode_readout'])==result['mode_readout']
    ranker=Ranker(tmp_path/'model');a=read(path).arrays
    counts=[]
    handle=ranker.model.local.register_forward_hook(lambda *args:counts.append(1))
    try:
        output=ranker.rank(a['q'],a['theory_intensity'][2],background=a['background'],
            reference=dict(q=a['q'],intensity=a['parent_intensity'][2],background=a['parent_background'][2]),
            parent_sha256='a'*64,instrument_sha256='b'*64)
    finally:handle.remove()
    assert counts==[1] and output['ranking_available']
    evidence=output['mode_estimates'];assert evidence['available'] and len(evidence['rows'])==2
    assert all(row['status']=='unresolved' and row['lower_estimate'] is not None for row in evidence['rows'])
    assert not evidence['experimental_mode_absence_qualified'] and not evidence['intervals_calibrated']
    assert not output['candidate_pruning_authorized'] and output['complete_native_search_required']


def test_joint_evaluation_shares_one_encoder_pass_and_mode_gradients_reach_the_encoder(tmp_path):
    _,value=cohort(tmp_path/'source');context=joint.prepare(value,value['split'])
    model=network(len(value['classes']),architecture=weights.MODE_ARCHITECTURE,family_count=1)
    selected=np.flatnonzero(value['split']==0)
    logits,embedding=joint.shared(model,torch.from_numpy(value['features'][selected]))
    loss=context.objective(model,embedding,selected);loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.local.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mode_head.parameters())
    from gpx_recipe.preparation.structural_attention_metrics import evaluate
    calls=[];handle=model.local.register_forward_hook(lambda *args:calls.append(1));model.eval()
    try:
        result=evaluate(model,feature_rows(value['features']),
            value['compatible_labels'],np.flatnonzero(value['split']==1),np.zeros((3,1),np.float32),np.ones((3,1),np.float32),
            value['classes'],_mode_context=context)
    finally:handle.remove()
    assert calls==[1] and result['mode_validation']['available']


def test_missing_fitting_conditions_stay_unknown_and_never_become_absent(tmp_path):
    _,value=cohort(tmp_path/'source');context=joint.prepare(value,value['split'])
    spec=copy.deepcopy(context.spec);spec['known_intervals'][1][0]=0
    model=network(2,architecture=weights.MODE_ARCHITECTURE,family_count=1);model.eval()
    with torch.no_grad():
        output=joint.infer(model,model.embedding(torch.from_numpy(value['features'][:1])),spec,space_groups=spec['classes'])
    assert output['rows'][0]['lower_estimate'] is not None
    assert output['rows'][1]['lower_estimate'] is None and output['rows'][1]['upper_estimate'] is None
    assert all(row['status']=='unresolved' for row in output['rows'])


def test_target_scale_method_is_bound_and_legacy_specs_keep_exact_bytes(tmp_path):
    _,value=cohort(tmp_path/'source');context=joint.prepare(value,value['split'])
    spec=context.spec
    assert spec['target_scale_method']==joint.readout.SCALE_METHOD
    legacy=copy.deepcopy(spec);legacy.pop('target_scale_method')
    assert joint.validate(legacy)==legacy
    encoded=joint.encode(legacy)
    assert joint.decode(encoded,legacy)==legacy
    assert joint.encode(joint.decode(encoded,legacy)).tobytes()==encoded.tobytes()
    for invalid in [None,'positive-median/v1','unknown']:
        with pytest.raises(ValueError,match='scaling method'):
            joint.validate(dict(spec,target_scale_method=invalid))


def test_joint_checkpoint_selection_accounts_for_mode_validation_without_using_test_scores(tmp_path):
    _,value=cohort(tmp_path/'source');context=joint.prepare(value,value['split']);model=network(2,architecture=weights.MODE_ARCHITECTURE,family_count=1)
    selected=np.flatnonzero(value['split']==1);model.eval()
    state=context.metrics()
    with torch.no_grad():state.add(model,model.embedding(torch.from_numpy(value['features'][selected])),selected)
    modes=state.finish()
    from gpx_recipe.preparation.structural_compatible_objective import checkpoint_key,OBJECTIVE
    first=dict(loss_method=OBJECTIVE,macro_compatible_nll=1.,loss=1.,mode_validation=modes)
    second=copy.deepcopy(first)
    for condition in second['mode_validation']['conditions']:condition['pinball_loss']+=1.
    second['mode_validation']['loss']+=1.
    assert checkpoint_key(first)<checkpoint_key(second)
    first['top5']=0.;second['top5']=1.
    assert checkpoint_key(first)<checkpoint_key(second)


def test_joint_training_resumes_the_complete_head_optimizer_without_changing_the_selected_result(tmp_path):
    _,value=cohort(tmp_path/'source');reference=training.train(value,tmp_path/'reference',epochs=2,threads=1)
    from gpx_recipe.preparation import structural_training_transfer as transfer
    resume=None;partial=[]
    for index in range(20):
        destination=tmp_path/('worker-'+str(index))
        result=training.train(value,destination,epochs=2,threads=1,_segment_seconds=1e-9,_resume=resume)
        if result.get('status')!='checkpointed':break
        partial.append(result);resume=tmp_path/('transfer-'+str(index))
        transfer.snapshot(destination,resume,identity=result['identity'],epochs=2)
    assert partial and result['mode_head_trained']
    for key in ['weights_sha256','history','test','selected_epoch','mode_readout']:
        assert result[key]==reference[key]


@pytest.mark.parametrize('change',['metadata','weights_spec','unqualified_claim','missing_tensor'])
def test_portable_mode_descriptors_cannot_be_relabelled_or_grant_absence_authority(tmp_path,change):
    _,value=cohort(tmp_path/'source');training.train(value,tmp_path/'model',epochs=1,threads=1)
    path=tmp_path/'model/model.json';d=json.loads(path.read_text())
    if change=='metadata':d['mode_readout']['scales'][0]*=1.01
    elif change=='unqualified_claim':d['mode_readout']['experimental_mode_absence_qualified']=True
    else:
        with np.load(tmp_path/'model/weights.npz',allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
        if change=='weights_spec':arrays[joint.ARRAY]=joint.encode(dict(d['mode_readout'],scales=[d['mode_readout']['scales'][0]*1.01]))
        else:arrays.pop(joint.ARRAY)
        np.savez_compressed(tmp_path/'model/weights.npz',**arrays);d['weights_sha256']=file_sha(tmp_path/'model/weights.npz')
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):Ranker(tmp_path/'model')
