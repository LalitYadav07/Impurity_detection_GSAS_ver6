from copy import deepcopy
import json
import pytest
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.model_checkpoint import portable_checkpoint, compact_transport, verify_archived_receipts


def checkpoint():
    receipt=dict(schema='radar-symmetry-trial/v1',id='trial',protocol_sha256='p',status='unsupported',reason='control')
    chunk=dict(key='dispatch',protocol='p',members=['trial'],complete=True,
        request={'scientific_choices':['unchanged']},
        job=dict(galaxy_job_id='job',inputs={'model_request_b64':'duplicated'},outputs={'result_bundle':'result-hda'}),
        result=dict(schema='radar-model-chunk-result/v1',dispatch_key='dispatch',protocol_sha256='p',
                    trials=[dict(id='trial',state='finished')],pending=[]))
    row=dict(id='trial',state='terminal',protocol='p',handle=json.dumps(dict(key='trial',dispatch_key='dispatch')),
             receipt=json.dumps(receipt))
    return dict(ledger=dict(trials=[row]),transport=dict(chunks={'dispatch':chunk},members={'trial':'dispatch'})),receipt


def test_compact_checkpoint_keeps_exact_receipt_witness_and_all_finite_starts():
    document,receipt=checkpoint();before=deepcopy(document)
    result=portable_checkpoint(document)
    assert document==before
    assert result['ledger']['trials'][0]['receipt'] is None
    assert result['transport']['receipt_hashes']=={'trial':digest(receipt)}
    archived=result['transport']['chunks']['dispatch']
    assert 'request' not in archived and 'inputs' not in archived['job']
    assert archived['job']['outputs']=={'result_bundle':'result-hda'}
    assert result['ledger']['trials'][0]['handle']==before['ledger']['trials'][0]['handle']
    assert portable_checkpoint(result)==result


@pytest.mark.parametrize('change',[
    dict(complete=False),dict(job=None),dict(job={'galaxy_job_id':'job','outputs':{}}),
    dict(result={'schema':'wrong'}),dict(result={'schema':'radar-model-chunk-result/v1','dispatch_key':'wrong','protocol_sha256':'p'}),
])
def test_ambiguous_or_unarchived_intents_keep_their_request_and_receipt(change):
    document,_=checkpoint();document['transport']['chunks']['dispatch'].update(change)
    result=portable_checkpoint(document)
    assert result['transport']['chunks']['dispatch']['request']=={'scientific_choices':['unchanged']}
    assert result['ledger']['trials'][0]['receipt']==document['ledger']['trials'][0]['receipt']


def test_prepared_and_terminal_without_matching_membership_are_preserved():
    document,_=checkpoint();document['ledger']['trials'][0]['state']='prepared'
    assert portable_checkpoint(document)['ledger']['trials'][0]['receipt'] is not None
    document['ledger']['trials'][0]['state']='terminal';document['transport']['members']['trial']='different'
    assert portable_checkpoint(document)['ledger']['trials'][0]['receipt'] is not None


def test_completed_receipt_cache_cannot_change_its_existing_witness():
    document,_=checkpoint();document['transport']['receipt_hashes']={'trial':'0'*64}
    with pytest.raises(ValueError,match='receipt changed'):portable_checkpoint(document)


def test_recovery_checks_original_witness_before_a_result_is_published(tmp_path):
    document,receipt=checkpoint();state=portable_checkpoint(document)['transport']
    path=tmp_path/'trials/trial/result.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(receipt))
    terminal={'trial':dict(state='finished')}
    verify_archived_receipts(state,terminal,tmp_path)
    path.write_text(json.dumps(dict(receipt,reason='changed after checkpoint')))
    with pytest.raises(ValueError,match='differs from its saved checkpoint'):
        verify_archived_receipts(state,terminal,tmp_path)


def test_unacknowledged_chunk_is_never_compacted():
    state={'chunks':{'pending':dict(request={'finite':['a','b']},pending=True,job=None)}}
    original=deepcopy(state);compact_transport(state)
    assert state==original
