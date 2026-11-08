"""Conditional-head contracts; authored curves do not establish accuracy."""
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_readout as modes
from gpx_recipe.preparation.symmetry_mode_normalization import transform,physical
from gpx_recipe.preparation.structural_family_targets import build
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_family_targets import mapped,prepared


def test_normalization_uses_only_original_fitting_aliases_and_preserves_tiny_amplitudes(tmp_path):
    path=mapped(tmp_path/'tiny',tiny=True);blocks,cohort=prepared([path],noise=True)
    table=build(blocks,cohort);normal=modes.fit(table,cohort)
    assert normal.scales.dtype==np.float64 and normal.scales[0]==pytest.approx(2e-200,rel=1e-12,abs=0)
    values=normal.targets(table,np.arange(len(table),dtype=np.int64))
    assert values['upper'].max()==pytest.approx(np.arcsinh(1.))
    assert np.count_nonzero(values['upper']==0)>0
    assert normal.metadata['mode_head_trained'] is False
    assert not normal.metadata['experimental_mode_absence_qualified']
    assert normal.metadata['fitting_alias_groups']<normal.metadata['fitting_rows']
    with pytest.raises(ValueError):normal.scales.flags.writeable=True
    with pytest.raises(ValueError):normal.known.flags.writeable=True


def test_changed_holdout_amplitudes_cannot_change_fitting_scales(tmp_path):
    path=mapped(tmp_path/'source');blocks,cohort=prepared([path]);table=build(blocks,cohort)
    before=modes.fit(table,cohort)
    heldout={i for row,refs in enumerate(cohort['references']) if cohort['split'][row]!=0 for _,i in refs}
    with np.load(path/'pairs.npz',allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
    for index in heldout:
        if a['role'][index]==2:a['parameters'][index,0]=100.
    np.savez_compressed(path/'pairs.npz',**a)
    d=json.loads((path/'pairs.json').read_text());d['pairs_sha256']=file_sha(path/'pairs.npz')
    (path/'pairs.json').write_text(json.dumps(d))
    blocks,changed=prepared([path]);other=build(blocks,changed);after=modes.fit(other,changed)
    np.testing.assert_array_equal(before.scales,after.scales)
    np.testing.assert_array_equal(before.known,after.known)
    assert other.arrays['upper'].max()>100*table.arrays['upper'].max()
    with pytest.raises(ValueError):before.targets(other,np.array([0]))


@pytest.mark.parametrize('scale,amplitude',[(1.,0.),(2e-200,1e-200),(1e-300,1e200),(1e300,1e200)])
def test_amplitude_transform_retains_finite_scale_cancellation(scale,amplitude):
    encoded=transform(np.array([amplitude]),scale)
    actual=physical(encoded,scale)
    assert actual[0]==pytest.approx(amplitude,rel=5e-13,abs=0)
    assert bool(encoded[0]>0)==bool(amplitude>0)


def test_amplitude_transform_never_turns_a_numerical_failure_into_zero():
    with pytest.raises(ValueError,match='erase'):transform(1e-300,1e300)
    with pytest.raises(ValueError):physical(1000.,1.)
    with pytest.raises(ValueError):transform(-1.,1.)
    with pytest.raises(ValueError):physical(1.,0.)


def test_unmapped_families_cannot_create_a_mode_predictor(tmp_path):
    path=mapped(tmp_path/'legacy',legacy=True);blocks,cohort=prepared([path])
    with pytest.raises(ValueError):modes.fit(build(blocks,cohort),cohort)


def test_shared_embedding_preserves_class_forward_and_trains_conditional_modes():
    torch=pytest.importorskip('torch')
    from gpx_recipe.preparation.structural_attention import network
    torch.set_num_threads(1);torch.manual_seed(871)
    encoder=network(1);head=modes.network(1,3)
    values=torch.randn(4,3,128)
    keys=list(encoder.state_dict());encoder.eval()
    with torch.no_grad():
        embedding=encoder.embedding(values)
        assert torch.equal(encoder(values),encoder.head[-1](embedding))
    assert list(encoder.state_dict())==keys
    encoder.train()
    rows=np.repeat(np.arange(4),3).astype(np.int64);families=np.tile(np.arange(3),4).astype(np.int64)
    queries=[torch.from_numpy(v) for v in (rows,np.zeros(12,np.int64),families)]
    target=dict(row=rows,space_group=np.zeros(12,np.int64),
        lower=np.tile([0.,.3,.8],4).astype(np.float64),upper=np.tile([0.,.5,1.1],4).astype(np.float64))
    prediction=head(encoder.embedding(values),*queries)
    assert prediction.shape==(12,2) and bool((prediction[:,1]>=prediction[:,0]).all())
    loss=modes.loss(prediction,target,rows=4,class_count=1);loss.backward()
    assert float(encoder.local[0].weight.grad.abs().sum())>0
    assert float(encoder.attention.qkv.weight.grad.abs().sum())>0
    assert float(head.family.weight.grad.abs().sum())>0
    assert torch.isfinite(loss) and loss.dtype==torch.float64


def test_interval_loss_balances_classes_and_ignores_rows_with_unknown_targets():
    torch=pytest.importorskip('torch')
    prediction=torch.zeros((3,2),requires_grad=True)
    targets=dict(row=np.zeros(3,np.int64),space_group=np.array([0,0,1],np.int64),
        lower=np.array([1.,1.,3.]),upper=np.array([1.,1.,3.]))
    actual=modes.loss(prediction,targets,rows=2,class_count=2)
    # Class 0 has two known families; it does not receive double class mass.
    assert actual.item()==pytest.approx(1.)
    actual.backward();assert torch.isfinite(prediction.grad).all()
    targets['lower']=targets['lower'].astype(np.float32)
    with pytest.raises(ValueError):modes.loss(prediction,targets,rows=2,class_count=2)


def test_sparse_queries_are_complete_and_invalid_indices_are_rejected():
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    head=modes.network(2,4);embedding=torch.randn(2,64)
    row=torch.arange(4097)%2;sg=row.clone();family=torch.arange(4097)%4
    whole=head(embedding,row,sg,family)
    parts=torch.cat([head(embedding,row[:1000],sg[:1000],family[:1000]),
        head(embedding,row[1000:],sg[1000:],family[1000:])])
    assert whole.shape==(4097,2) and torch.allclose(whole,parts,rtol=1e-6,atol=1e-6)
    assert head(embedding,row[:0],sg[:0],family[:0]).shape==(0,2)
    with pytest.raises(ValueError):head(embedding,row,sg,family+4)
    with pytest.raises(ValueError):head(embedding,row.float(),sg,family)
    with pytest.raises(ValueError):head(torch.randn(2,63),row,sg,family)
