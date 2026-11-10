import numpy as np
import pytest

from gpx_recipe.preparation.structural_domain import envelope,relative_metric_norm,validate,parent_reference_norms,maximum_parent_metric,PARENT_SCHEMA
from gpx_recipe.preparation.structural_cohort import assemble,read_block
from test_structural_ranker_cohort import block


def test_metric_envelope_accounts_for_anisotropic_lattice_changes():
    cell=[4.,5.,6.,90.,90.,90.]
    assert relative_metric_norm(cell,cell)==0
    changed=[4.02,4.99,6.01,90.,90.,90.]
    expected=np.linalg.norm((np.array(cell[:3])/np.array(changed[:3]))**2-1)
    assert relative_metric_norm(cell,changed)==pytest.approx(expected)
    assert validate(envelope(expected))['maximum_relative_reciprocal_metric_norm']==expected


@pytest.mark.parametrize('bad',[-.1,float('nan'),float('inf'),True,[],None])
def test_training_domain_cannot_claim_invalid_coverage(bad):
    value=envelope(.01);value['maximum_relative_reciprocal_metric_norm']=bad
    with pytest.raises(ValueError):validate(value)


def test_only_training_trajectories_define_the_admitted_lattice_range(tmp_path):
    def strains(a):a['relative_metric_norm'][:]=np.arange(len(a['trajectory']))*.001
    directories=[block(tmp_path,change=strains),block(tmp_path,'second',label=65,offset=.01,change=strains)]
    result=assemble([read_block(p) for p in directories],bins=128)
    expected=max(read_block(directories[b]).arrays['relative_metric_norm'][i]
        for j,refs in enumerate(result['references']) if result['split'][j]==0 for b,i in refs)
    assert result['training_domain']['maximum_relative_reciprocal_metric_norm']==expected
    assert result['training_domain']['source']=='native_training_examples_only'


def test_parent_reference_norms_retain_anisotropy_and_off_diagonal_metric_changes():
    a=np.array([[.04,.03,.02,.001,.002,.003],[.0401,.0299,.02005,.0011,.0022,.0029]])
    actual=parent_reference_norms(a)
    def matrix(v):return np.array([[v[0],v[3]/2,v[4]/2],[v[3]/2,v[1],v[5]/2],[v[4]/2,v[5]/2,v[2]]])
    delta=np.linalg.solve(matrix(a[0]),matrix(a[1])-matrix(a[0]))
    assert actual[0]==0 and actual[1]==pytest.approx(np.sqrt(np.trace(delta@delta)))


def test_paired_envelope_uses_training_parent_cells_without_child_strain_or_holdout_leakage(tmp_path):
    from test_structural_paired_blocks import block as paired_block
    from gpx_recipe.preparation.structural_paired_blocks import read
    from gpx_recipe.preparation.structural_cohort import assemble_pairs
    def changes(a):
        for trajectory in sorted(set(a['trajectory'])-{-1}):
            selected=a['trajectory']==trajectory
            a['parent_a'][selected]*=1.+.0001*(int(trajectory)+1)
        a['relative_metric_norm'][:]=.08  # This child's strain is not parent-reference coverage.
    path=paired_block(tmp_path/'paired',change=changes);b=read(path)
    cohort=assemble_pairs([b],bins=128);norms=parent_reference_norms(b.arrays['parent_a'])
    expected=max(norms[i] for row,refs in enumerate(cohort['references']) if cohort['split'][row]==0 for _,i in refs)
    assert cohort['training_domain']['schema']==PARENT_SCHEMA
    assert cohort['training_domain']['maximum_relative_reciprocal_metric_norm']==pytest.approx(expected)
    assert expected<.01


def test_old_paired_models_do_not_claim_parent_shift_coverage_from_a_child_strain_envelope():
    from gpx_recipe.preparation.structural_attention_weights import SCHEMA as THREE_CURVE_SCHEMA
    assert maximum_parent_metric(dict(schema=THREE_CURVE_SCHEMA,training_domain=envelope(.02)))==0.
    assert maximum_parent_metric(dict(schema=THREE_CURVE_SCHEMA,training_domain=envelope(.003,parent_references=True)))==.003
    assert maximum_parent_metric(dict(schema='radar-structural-ranking-model/v4',training_domain=envelope(.02)))==.02


@pytest.mark.parametrize('bad',[np.zeros((2,6)),np.ones((2,5)),np.full((2,6),np.nan)])
def test_invalid_parent_reference_metrics_cannot_expand_training_coverage(bad):
    with pytest.raises(ValueError):parent_reference_norms(bad)
