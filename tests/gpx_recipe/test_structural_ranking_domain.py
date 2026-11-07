import numpy as np
import pytest

from gpx_recipe.preparation.structural_domain import envelope,relative_metric_norm,validate
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
