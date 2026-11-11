"""Cross-scan approval and publication must keep the native model boundary."""
import hashlib

import pytest

from gpx_recipe import compatibility as compat


REVISION = 'd' * 64


def context(sha='a' * 64, *, aluminium=False):
    phases = [dict(key='p0', name='Er oxide', space_group='P 1',
                   atoms=[dict(label='Er1', element='Er', adp_type='I', multiplicity=1, site_symmetry='1')])]
    if aluminium:
        phases.append(dict(key='p1', name='Al can', space_group='F m -3 m', atoms=[]))
    binding = dict(phases=phases, histograms=[dict(key='h0', instrument={'Type':'PNC','Lam':2.41},
        background=dict(function='chebyschev-1', coefficients=6))], source_constraints=[], source_bounds={})
    return dict(source=dict(sha256=sha), model_signature=compat._digest(binding), binding=binding)


def source(uid, ctx, *, cached=False):
    value = dict(id=uid, scan_label=uid, source_sha256=ctx['source']['sha256'])
    if cached:
        value.update(context=ctx, inspection_revision=REVISION)
    else:
        value['project'] = dict(recipe_model=compat.descriptor(ctx, revision=REVISION))
    return value


def test_new_publications_and_cached_inspections_distinguish_aluminium_model():
    ref = context()
    rows = [source('cold',ref,cached=True), source('warm',context('b'*64)),
            source('with-can',context('c'*64,aluminium=True)),dict(id='legacy',source_sha256='f'*64)]
    report = compat.preview(rows,ref,revision=REVISION)
    assert (report['compatible'],report['incompatible'],report['unverified']) == (2,1,1)
    assert 'Al can' in report['rows'][2]['reason']
    assert 'native worker' in report['rows'][3]['reason']


@pytest.mark.parametrize('change', ['source','revision','binding','signature','schema'])
def test_stale_or_tampered_publication_does_not_prove_compatibility(change):
    ref = context(); value = source('scan',ref); record=value['project']['recipe_model']
    if change=='source':value['source_sha256']='b'*64
    elif change=='revision':record['implementation_sha256']='e'*64
    elif change=='binding':record['binding']['histograms'][0]['instrument']['Lam']=1.54
    elif change=='signature':record['model_signature']='f'*64
    else:record['schema']='future/v2'
    assert compat.assess(value,ref,revision=REVISION)['status']=='unverified'


@pytest.mark.parametrize('field,phrase', [('instrument','calibration'),('background','Background')])
def test_same_phase_but_different_profile_or_background_is_incompatible(field,phrase):
    ref=context(); other=context('b'*64)
    other['binding']['histograms'][0][field]={}
    other['model_signature']=compat._digest(other['binding'])
    result=compat.assess(source('scan',other),ref,revision=REVISION)
    assert result['status']=='incompatible' and phrase in result['reason']


def test_variable_cell_values_do_not_split_the_recipe_model():
    ref=context(); other=context('b'*64)
    ref['cell']=[4,4,4];other['cell']=[4.2,4.2,4.2]
    assert compat.assess(source('scan',other),ref,revision=REVISION)['status']=='compatible'


def test_same_phase_name_with_different_atom_labels_and_profile_has_both_reasons():
    ref=context();other=context('b'*64)
    other['binding']['phases'][0]['atoms'][0].update(label='Tb1',element='Tb')
    other['binding']['histograms'][0]['instrument']['Lam']=2.42
    other['model_signature']=compat._digest(other['binding'])
    report=compat.assess(source('scan',other),ref,revision=REVISION)
    assert report['status']=='incompatible'
    assert 'recipe-only Er1; scan-only Tb1' in report['reason']
    assert 'calibration differs for histogram 1' in report['reason']


def test_capture_rejects_files_changed_before_or_during_inspection(tmp_path,monkeypatch):
    from gpx_recipe import snapshot
    file=tmp_path/'source.gpx';file.write_bytes(b'original')
    sha=hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(snapshot,'snapshot',lambda _:pytest.fail('Do not inspect changed bytes'))
    with pytest.raises(ValueError,match='before'):compat.capture(file,expected_sha256='f'*64)
    def changed(_):
        file.write_bytes(b'replaced');return context(sha)
    monkeypatch.setattr(snapshot,'snapshot',changed)
    with pytest.raises(ValueError,match='during'):compat.capture(file,expected_sha256=sha)


def _workspace(tmp_path,monkeypatch):
    from test_refinement_integration import create, source as base_source
    monkeypatch.setattr(compat,'inspection_revision',lambda:REVISION)
    workspace,runtime=create(tmp_path)
    base=base_source('s1',checksum='a'*64);workspace.observe([base]);workspace.inspect('s1')
    ref=context()
    # The recipe fixture needs a native-like histogram definition in addition
    # to the canonical identity being tested here.
    ref['histograms']=[dict(key='h0',name='h',background=dict(function='chebyschev-1',coefficients=6))]
    ref['phases']=[dict(key='p0',name='oxide',histograms=['h'],atoms=[])]
    workspace.state['sources']['s1'].update(context=ref,inspection_revision=REVISION)
    draft=workspace.prepare('exp','s1','background only')
    # prepare() uses the fake inspect revision; bind the completed inspection
    # exactly as a collected native result would be bound.
    draft=workspace.state['drafts']['exp'];draft['context']=ref
    draft['plan']['model_signature']=ref['model_signature'];draft['preview']['model_signature']=ref['model_signature']
    from gpx_recipe.contracts import plan_hash
    draft['preview']['plan_sha256']=plan_hash(draft['plan'])
    return workspace,runtime,draft,base_source


def test_approval_refuses_known_incompatible_scan_without_submitting(tmp_path,monkeypatch):
    from radar_pd_nova.refinement_sources import RefinementScope
    w,rt,draft,base_source=_workspace(tmp_path,monkeypatch)
    other=base_source('s2',checksum='b'*64);other.update(source('s2',context('b'*64,aluminium=True)))
    w.observe([other]);scope=RefinementScope('exp');_,token=w.selection(scope)
    before=len(rt.calls)
    with pytest.raises(ValueError,match='separate recipe.*Al can'):
        w.approve('exp',draft['generation'],scope,existing=True,future=False,selection_token=token,approver='user')
    assert not w.state['recipes'] and len(rt.calls)==before


def test_known_incompatible_future_scan_is_held_without_delaying_matching_scans(tmp_path,monkeypatch):
    from radar_pd_nova.refinement_sources import RefinementScope
    w,rt,draft,base_source=_workspace(tmp_path,monkeypatch)
    scope=RefinementScope('exp');_,token=w.selection(scope,future=True)
    recipe=w.approve('exp',draft['generation'],scope,existing=True,future=True,selection_token=token,approver='user')
    w.start(recipe['id'],token)
    other=base_source('s2',checksum='b'*64);other.update(source('s2',context('b'*64,aluminium=True)))
    w.observe([other]);job=next(j for j in w.state['jobs'].values() if j['source']['id']=='s2')
    assert job['status']=='needs_recipe' and job['attempts']==0 and 'Al can' in job['message']
    assert w.run_next()['source']['id']=='s1'
    assert not any(call[1]=='s2' for call in rt.calls)
    restored=type(w)(tmp_path,history_id='history',runtime=rt,planner=w._planner)
    assert next(j for j in restored.state['jobs'].values() if j['source']['id']=='s2')['status']=='needs_recipe'


def test_legacy_unknown_remains_native_checked_not_silently_rejected(tmp_path,monkeypatch):
    from radar_pd_nova.refinement_sources import RefinementScope
    w,rt,draft,base_source=_workspace(tmp_path,monkeypatch)
    w.observe([base_source('legacy',checksum='c'*64)])
    scope=RefinementScope('exp');_,token=w.selection(scope)
    recipe=w.approve('exp',draft['generation'],scope,existing=True,future=False,selection_token=token,approver='user')
    w.start(recipe['id'],token)
    assert all(j['status']=='queued' for j in w.state['jobs'].values())
