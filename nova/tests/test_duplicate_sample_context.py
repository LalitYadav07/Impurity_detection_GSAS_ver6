"""Recorded title context must help selection without becoming identity."""
from copy import deepcopy
import time

from radar_pd_nova.sample_labels import with_recorded_run_context
from radar_pd_nova.live_assistant_language import ReplyContext, SampleOption, interpret_reply
from radar_pd_nova import powgen_samples, sample_catalog
from radar_pd_nova.facility_metadata import ExperimentMetadataIndex
from radar_pd_nova.powgen_samples import PowgenSampleIndex
from radar_pd_nova.powgen_watch import WatchedRun


def option(value, name='NMC'):
    return {'value':value,'title':f'{value} — {name} (2 scans)',
            'sample_names':[name],'scan_count':2}


def test_duplicate_names_get_their_own_recorded_context_without_changing_identity():
    options = [option('sample-A'),option('sample-B')]
    runs = [
        {'sample_id':'sample-A','value':11,'run_id':'NOM_11','run_title':'oxide-A first scan'},
        {'sample_id':'sample-B','value':12,'run_id':'NOM_12','run_title':'oxide-B first scan'},
        {'sample_id':'unrelated','value':13,'run_title':'must not leak'},
    ]
    before = deepcopy((options,runs))
    displayed = with_recorded_run_context(options,runs)
    assert (options,runs) == before
    for original, current in zip(options,displayed):
        assert current['value'] == original['value']
        assert current['title'] == original['title']
    assert displayed[0]['props']['subtitle'] == 'Recorded title examples · NOM_11: oxide-A first scan'
    assert displayed[1]['props']['subtitle'] == 'Recorded title examples · NOM_12: oxide-B first scan'
    assert 'must not leak' not in str(displayed)
    context = ReplyContext(step='selection',samples=tuple(SampleOption(x['value'],x['title']) for x in displayed))
    assert interpret_reply('NMC',context).action == 'clarify'
    exact = interpret_reply('sample ID sample-B',context)
    assert exact.value['powgen_processing_sample_ids'] == ['sample-B']


def test_unique_names_and_missing_titles_do_not_gain_invented_context():
    unique = [option('A','Nickel'),option('B','Copper')]
    assert with_recorded_run_context(unique,[]) is unique
    duplicate = [option('A'),option('B')]
    assert with_recorded_run_context(duplicate,[{'sample_id':'A','value':1}]) == duplicate


def test_repeated_names_are_case_insensitive_and_keep_distinct_first_last_labels():
    opts = [option('A','NMC'),option('B',' nmc ')]
    runs = [{'sample_id':'A','value':i,'run_title':title}
            for i,title in enumerate(['start','start','middle','last'],1)]
    out = with_recorded_run_context(opts,runs)
    assert [r['run_title'] for r in out[0]['run_title_examples']] == ['start','last']
    assert out[0]['props']['subtitle'].endswith('Run 1: start; Run 4: last')
    assert 'props' not in out[1]


def test_long_or_markup_titles_remain_bounded_plain_strings():
    out = with_recorded_run_context([option('A'),option('B')],
          [{'sample_id':'A','value':1,'run_title':'<b>literal label</b> ' + 'x'*1000}])
    subtitle = out[0]['props']['subtitle']
    assert len(subtitle) <= 465
    assert '<b>literal label</b>' in subtitle
    assert subtitle.endswith('…')
    assert out[0]['props']['lines'] == 'three'


def test_catalog_context_uses_existing_titles_without_extra_raw_reads(tmp_path,monkeypatch):
    rows = {n:{'sample_id':sid,'sample_name':'Shared name',
               'run_title':title,'identity_status':'resolved'}
            for n,sid,title in [(1,'A','A cold'),(2,'A','A warm'),(3,'B','B cold')]}
    monkeypatch.setattr(powgen_samples,'load_sample_catalog',lambda *a:rows)
    monkeypatch.setattr(powgen_samples,'catalog_created_at',lambda *a:time.monotonic())
    monkeypatch.setattr(powgen_samples,'run_role',lambda *a: (_ for _ in ()).throw(AssertionError('extra file read')))
    index = PowgenSampleIndex('IPTS-1',metadata_index=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path))
    monkeypatch.setattr(index,'identity',lambda *a: (_ for _ in ()).throw(AssertionError('extra identity read')))
    runs = [WatchedRun(n,str(tmp_path/f'PG3_{n}.gsa'),str(n)) for n in rows]
    found = index.inventory(runs,catalog=True)
    assert found['catalog_preview_count'] == 3
    assert [x['value'] for x in found['options']] == ['A','B']
    assert 'A cold' in found['options'][0]['props']['subtitle']
    assert 'B cold' in found['options'][1]['props']['subtitle']
    assert index._cache == {}
