"""Source-bound advisory scheduling, with opaque scientific transport fixtures."""
import copy
import json
import shutil
import threading

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints
from gpx_recipe.preparation.structural_ranking_result import unavailable,validate
from gpx_recipe.symmetry import chunks
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_candidate,validate_protocol,iter_trials,trial_id
from test_preparation_structural_ranking_hints import prepared
from test_symmetry_contracts import make_protocol


def setup_inputs(root):
    library,thin=prepared(root/'library-inputs')
    inputs=root/'scientific-inputs';p=make_protocol(inputs)
    p.update(kind='structural',sites=[],candidates=[validate_candidate(c) for c in thin['candidates']],
        implementation_sha256=package_signature(),seeds=[dict(id='first'),dict(id='second')])
    p['policy'].update(maximum_trials=4,refine_cell=True,refine_background=True,
        refine_histogram_scale=True,structural_optimizer='prepared_nonlinear')
    row=next(r for r in library.document['entries'] if r['model']['kind']=='structural-examples')
    shutil.copyfile(library.root/row['model']['child'],inputs/'child.cif')
    write_json(inputs/'catalog.json',dict(schema='radar-symmetry-catalog/v1',
        parent_setting_sha256=p['parent_setting_sha256'],k=p['k'],candidates=p['candidates']))
    p['catalog']['sha256']=file_sha(inputs/'catalog.json');p=validate_protocol(p)
    write_json(inputs/'protocol.json',p)
    document=hints.capture(library,p,inputs)
    assert len(document['ranking_models']['entries'])==1
    return inputs,p,document


def receipt(p,document):
    entry=document['ranking_models']['entries'][0]
    result=unavailable(p,'Fixture priority; not a scientific classifier result',snapshot=document['snapshot_sha256'])
    result.update(ranking_available=True,candidate_order=['second','first'],
        families=[dict(space_group=65,model_score=.6,candidate_ids=['second']),
                  dict(space_group=63,model_score=.4,candidate_ids=['first'])],
        entry_id=entry['entry'],bank=entry['bank'],
        checks=[dict(entry=entry['entry'],status='compatible',reason='Native admission fixture')],
        support=dict(status='within_empirical_synthetic_support',nearest_training_rms=.1,
            validation_maximum_rms=.2,statistical_coverage_qualified=False))
    return validate(result,p)


@pytest.mark.parametrize('ranking_owner',[True,False])
@pytest.mark.parametrize('adaptive',[True,False])
def test_chunk_ranks_once_and_orders_exact_existing_work(tmp_path,monkeypatch,ranking_owner,adaptive):
    inputs,p,document=setup_inputs(tmp_path)
    ids=[trial_id(p,c['id'],s['id']) for c,s in iter_trials(p)]
    request=dict(schema='radar-model-chunk/v1',protocol=p,protocol_sha256=digest(p),
        trials=ids,workers=1,dispatch_key='d'*24,seconds=240,prepared_scattering=document,
        rank_structural=ranking_owner)
    calls=[];order=[]
    def rank(*args,**kwargs):
        calls.append(kwargs);value=receipt(p,document)
        if adaptive:
            from gpx_recipe.preparation.structural_ranking_result import attach_adaptive
            from test_structural_ranking_result import work_set
            value=attach_adaptive(value,work_set([65,63]))
        return value
    monkeypatch.setattr(chunks,'run_structural_ranking',rank)
    monkeypatch.setattr(chunks,'allocation',lambda *a,**k:dict(workers=1))
    def runner(directory,*,cancel):
        order.append(json.loads((directory/'job.json').read_text())['candidate_id'])
        return dict(state='finished',status='technical_failure')
    result=chunks.execute_chunk(request,inputs,tmp_path/'worker',runner=runner)
    assert len(calls)==int(ranking_owner)
    assert order==(['second','second','first','first'] if ranking_owner else ['first','first','second','second'])
    assert len(result['trials'])==len(ids) and not result['pending']
    assert {r['id'] for r in result['trials']}==set(ids)
    assert ('structural_ranking' in result)==ranking_owner
    if ranking_owner:assert ('adaptive_work_set' in result['structural_ranking'])==adaptive


def test_optional_ranking_timeout_keeps_full_native_order(tmp_path,monkeypatch):
    _,p,document=setup_inputs(tmp_path)
    import subprocess
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],1)
    monkeypatch.setattr(chunks,'run_scientific',timeout)
    directory=tmp_path/'worker';directory.mkdir()
    result=chunks.run_structural_ranking(p,tmp_path/'captured',directory,
        snapshot=document['snapshot_sha256'],cancel=threading.Event(),timeout=1)
    assert not result['ranking_available'] and result['candidate_order']==['first','second']
    validate(result,p,snapshot=document['snapshot_sha256'])


@pytest.mark.parametrize('adaptive',[True,False])
def test_campaign_prioritizes_later_batches_and_preserves_the_finite_denominator(tmp_path,adaptive):
    inputs,p,document=setup_inputs(tmp_path)
    campaign=Campaign(tmp_path/'campaign',shared_inputs=True)
    ids=campaign.enroll(p,inputs,approved_hash=digest(p))
    evidence=receipt(p,document)
    if adaptive:
        from gpx_recipe.preparation.structural_ranking_result import attach_adaptive
        from test_structural_ranking_result import work_set
        evidence=attach_adaptive(evidence,work_set([65,63]))
    campaign.set_setting('structural_rankings',{digest(p):evidence})
    class Transport:
        max_active_jobs=2;chunk_size=1
        def __init__(self):self.keys=[]
        def submit_many(self,entries,*,workers):
            self.keys.extend(uid for uid,_ in entries)
            return {uid:dict(key=uid,dispatch_key=uid) for uid,_ in entries}
        def poll(self,handle,directory):return dict(state='running',handle=handle)
    transport=Transport();campaign.tick(transport,slots=1)
    indexed={row['id']:row for row in campaign.trials()}
    assert indexed[transport.keys[0]]['candidate']=='second'
    assert len(indexed)==len(ids)==4
    # Corrupt saved advice changes no trial identity and cannot remove work.
    bad=copy.deepcopy(receipt(p,document));bad['candidate_order']=['second']
    campaign.set_setting('structural_rankings',{digest(p):bad})
    campaign.tick(transport,slots=1)
    assert 'complete native order retained' in campaign.setting('structural_ranking_notice')
    assert len(campaign.trials())==4
