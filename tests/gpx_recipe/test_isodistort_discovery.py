import pytest
from gpx_recipe.symmetry.isodistort_discovery import choose_tier,discover_structural
from gpx_recipe.symmetry.isodistort_native import first_valid_form
from gpx_recipe.symmetry.isodistort_forms import IsodistortFormClient,ExternalServiceError


def candidate(uid,irrep,index):
    return dict(candidate_id=uid,irrep=irrep,irrep_label=irrep,label=uid,catalog_subgroup_index=index,catalog_lattice_index=2,catalog_basis=[[1,0,0],[0,2,0],[0,0,1]],catalog_origin=[0,0,0])


def test_highest_symmetry_uses_subgroup_index_and_retains_each_irrep():
    rows=[candidate('a','X1+',12),candidate('b','X1+',4),candidate('c','X2-',8)]
    assert [r['candidate_id'] for r in choose_tier(rows,'highest-symmetry-per-irrep',2)]==['b','c']
    with pytest.raises(ValueError,match='budget'):choose_tier(rows,'highest-symmetry-per-irrep',1)


def test_false_fixed_k_form_does_not_displace_a_valid_native_form():
    forms=[('wrong-fixed',1),('valid-parameterized',2)]
    def solve(remaining):return (0,{'match':False}) if remaining[0][0]=='wrong-fixed' else (0,{'a':'.5'})
    selected,parameters=first_valid_form(forms,solve)
    assert selected[0]=='valid-parameterized';assert parameters=={'a':'.5'}


def test_failed_child_and_deferred_opd_stay_in_discovery_coverage(tmp_path):
    rows=[candidate('iso:a','X1+',4),candidate('iso:b','X1+',8),candidate('iso:c','X2-',4)]
    class Client:
        def method2_k_vector_search(self,*args,**kwargs):return dict(candidates=rows,irrep_failures=[],session_state={},raw_responses={'parent':'public form'},standardized_parent_cif='data_parent')
        def method2_generate(self,_,uid):
            if uid=='iso:c':raise ValueError('Authoritative child unavailable')
            return dict(cif='data_child')
    parent=tmp_path/'parent.cif';parent.write_text('data_parent')
    c=discover_structural(parent,dict(setting_sha256='a'*64),['0','1/2','0'],tmp_path/'discovery',matcher=None,inventory=lambda _:dict(modes=['X1_mode']),client=Client())
    assert len(c['candidates'])==1
    assert [r['state'] for r in c['coverage']['candidates']]==['constructed','deferred','construction_failed']
    assert c['coverage']['discovered_count']==3


def test_form_adapter_cannot_post_structures_to_an_unrelated_endpoint():
    called=[];client=IsodistortFormClient(post=lambda *a,**k:called.append(a))
    with pytest.raises(ExternalServiceError,match='allowlist'):client._post('https://example.com/upload',step='test',data={})
    assert not called
