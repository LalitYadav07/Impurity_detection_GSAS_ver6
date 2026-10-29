import copy,json
import pytest
from test_symmetry_contracts import make_protocol
from gpx_recipe.symmetry.plans import build_plan
from gpx_recipe.symmetry.compiler import verify_inputs
from gpx_recipe.symmetry.contracts import iter_trials,digest


def inputs(tmp_path):
    p=make_protocol(tmp_path);catalog=json.loads((tmp_path/'catalog.json').read_text())
    context=dict(phases=[dict(name='nuclear',setting_sha256='a'*64,histograms=['PWDR sample'],atoms=[dict(label='Mn1',type='Mn',occupancy=1.)])],histograms=[dict(name='PWDR sample',instrument={'Type':['PNC','PNC',False]})],runtime={'magnetic_function_sha256':'b'*64})
    settings=dict(kind='magnetic',parent='nuclear',histogram='PWDR sample',k=['0','0','0'],candidates=['mag:fixture'],sites=p['sites'],directions=['x'],maximum_moment=6.,maximum_mode=.5,adequate_rwp=12.,cycles=12)
    return catalog,context,settings


def test_wrong_bank_or_xray_magnetic_source_needs_review(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    with pytest.raises(ValueError,match='not linked'):build_plan(context,catalog,tmp_path,dict(settings,histogram='another bank'))
    context['histograms'][0]['instrument']['Type'][0]='PXC'
    with pytest.raises(ValueError,match='neutron'):build_plan(context,catalog,tmp_path,settings)


def test_selected_displacement_modes_and_scoped_signs_preserve_full_catalog(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    (tmp_path/'child.cif').write_text('data_authoritative')
    from gpx_recipe.symmetry.compiler import file_sha
    c=catalog['candidates'][0];c.update(magnetic_space_group=None,child=dict(path='child.cif',sha256=file_sha(tmp_path/'child.cif')),modes=['X1_primary','Gamma_secondary'],irrep='X1+')
    (tmp_path/'catalog.json').write_text(json.dumps(catalog))
    settings.update(kind='structural',sites=[],modes=['mag:fixture|X1_primary'],start_mode=.1)
    p=build_plan(context,catalog,tmp_path,settings);verify_inputs(p,tmp_path)
    assert catalog['candidates'][0]['modes']==['X1_primary','Gamma_secondary']
    assert p['candidates'][0]['modes']==['X1_primary']
    assert [s['mode_values']['X1_primary'] for _,s in iter_trials(p)]==[-.1,.1]
    assert all(s['candidate_ids']==['mag:fixture'] for s in p['seeds'])


def test_explicit_chemical_site_validation_happens_before_compilation(tmp_path):
    catalog,context,settings=inputs(tmp_path);settings['sites']=[dict(label='Mn1',form_factor='Fe+3',lande_g=2.)]
    with pytest.raises(ValueError,match='chemical element'):build_plan(context,catalog,tmp_path,settings)


def test_full_fitted_magnetic_domain_is_explicit_and_bound_to_approval(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    full=build_plan(context,catalog,tmp_path,dict(settings,magnetic_d_min=0))
    limited=build_plan(context,catalog,tmp_path,dict(settings,magnetic_d_min=1))
    assert full['policy']['magnetic_d_min']==0
    assert digest(full)!=digest(limited)
    with pytest.raises(ValueError):build_plan(context,catalog,tmp_path,dict(settings,magnetic_d_min=-1))


def test_identical_project_bytes_from_different_acquisitions_do_not_collapse(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    first=build_plan(context,catalog,tmp_path,dict(settings,acquisition_id='run-1-bank-2'))
    second=build_plan(context,catalog,tmp_path,dict(settings,acquisition_id='run-2-bank-2'))
    assert first['source']==second['source']
    assert digest(first)!=digest(second)
