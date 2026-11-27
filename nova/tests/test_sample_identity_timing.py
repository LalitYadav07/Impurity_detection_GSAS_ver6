"""Acquisition-clock tests: real-shaped warm metadata must remain a conflict."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time

import h5py
import numpy as np
import pytest

from radar_pd_nova import powgen_metadata, powgen_samples, sample_catalog
from radar_pd_nova.facility_metadata import ExperimentMetadataIndex
from radar_pd_nova.powgen_samples import PowgenSampleIndex, sample_identity_summary, select_runs
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.sample_identity_timing import scope_identity_log

START = '2020-01-01T12:00:00.755076667+00:00'
END = '2020-01-01T12:05:57.974721667+00:00'


def make_run(path, *, static='123', values=(456, 123), times=(0.0, 72956.399187466),
             start=START, end=END, units='second', log_start=None):
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('entry/sample/identifier', data=[static.encode()])
        handle.create_dataset('entry/sample/name', data=[b'chromium trichloride'])
        handle.create_dataset('entry/start_time', data=[start.encode()])
        handle.create_dataset('entry/end_time', data=[end.encode()])
        handle.create_dataset('entry/DASlogs/SampleId/value', data=values)
        if times is not None:
            d = handle.create_dataset('entry/DASlogs/SampleId/time', data=times)
            d.attrs['units'] = units
            d.attrs['start'] = log_start or start
        # Sparse and never read: a metadata reader must not visit these points.
        handle.create_dataset('entry/events/event_id', shape=(100_000_000,), dtype='i', chunks=True)


def test_real_shaped_warm_metadata_remains_a_detailed_static_log_conflict(tmp_path, monkeypatch):
    path = tmp_path/'warm.h5'; make_run(path)
    original = h5py.Dataset.__getitem__
    seen = []
    def bounded_read(dataset, key):
        assert '/events/' not in dataset.name
        seen.append(dataset.name)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', bounded_read)
    result = powgen_metadata.read_powgen_sample_identity('IPTS-1', 1, nexus_path=path)
    assert result['sample_id'] == '' and result['identity_status'] == 'conflict'
    assert '/entry/sample/identifier = 123' in result['identity_warning']
    assert '/entry/DASlogs/SampleId/value during acquisition = 456' in result['identity_warning']
    scope, = result['identity_time_scopes']
    assert scope['ignored_after_count'] == 1 and scope['kept_indices'] == [0]
    assert scope['recorded_observations'] == ['456', '123']
    assert any('out-of-acquisition' in notice for notice in result['metadata_notices'])
    assert '/entry/DASlogs/SampleId/time' in seen


@pytest.mark.parametrize('values,times,expected', [
    ((123, 456), (0.0, 400.0), 'resolved'),  # Future metadata is not an in-run change.
    ((123, 456), (0.0, 100.0), 'conflict'),  # A real change stays visible.
    ((456, 123), (-20.0, -1.0), 'resolved'), # Last pre-run value carries into acquisition.
    ((123, 456, 123), (0.0, 100.0, 200.0), 'conflict'),
    ((123, 456), None, 'conflict'),           # Missing timing cannot erase evidence.
    ((123, 456), (100.0, 0.0), 'conflict'),  # Invalid timing cannot erase evidence.
])
def test_actual_reader_keeps_in_run_changes_and_only_filters_qualified_clocks(tmp_path, values, times, expected):
    path=tmp_path/'run.h5';make_run(path, values=values, times=times)
    result=powgen_metadata.read_powgen_sample_identity('IPTS-1', 1, nexus_path=path)
    assert result['identity_status'] == expected
    assert result['sample_id'] == ('123' if expected == 'resolved' else '')


@pytest.mark.parametrize('times,attrs,start,end', [
    ([0,400], {'start':START, 'units':'ticks'}, START, END),
    ([0,400], {'start':START, 'units':'second', 'scaling_factor':float('nan')}, START, END),
    ([0], {'start':START, 'units':'second'}, START, END),
    ([0,400], {'start':START, 'units':'second'}, START, 'not a time'),
    ([0,400], {'start':'2020-01-01T12:00:00', 'units':'second'}, START, END),
])
def test_unqualified_clocks_retain_all_observations(times,attrs,start,end):
    kept,scope=scope_identity_log(['A','B'], times, attrs, start, end)
    assert kept == [0,1] and scope['status'] == 'unqualified'


def test_duplicate_timestamp_different_ids_is_not_resolved_by_order():
    kept,scope=scope_identity_log(['A','B'],[0,0],{'start':START,'units':'second'},START,END)
    assert kept == [0,1] and scope['status'] == 'acquisition_window'


def test_post_start_observation_cannot_erase_the_identity_at_acquisition_start():
    kept,scope=scope_identity_log(['A','B'],[-1,0.0000005],{'start':START,'units':'second'},START,END)
    assert kept == [0,1] and scope['status'] == 'acquisition_window'


def test_scaling_factor_and_offset_are_applied_before_windowing():
    kept,scope=scope_identity_log(['A','B'],[0,400_000],{'start':START,'units':'ms'},START,END)
    assert kept == [0] and scope['ignored_after_count'] == 1


def test_clock_qualified_identity_can_be_cached_despite_outside_history_notice(tmp_path):
    path=tmp_path/'run.h5';make_run(path,values=[123,456],times=[0,400])
    reads=[]
    def reader(*args):
        reads.append(args)
        return powgen_metadata.read_powgen_sample_identity(*args,nexus_path=path)
    index=PowgenSampleIndex('IPTS-1',nexus_path=lambda *a:path,identity_reader=reader,
                           metadata_index=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path))
    run=WatchedRun(1,str(tmp_path/'PG3_1.gsa'),'unchanged')
    assert index.identity(run)['sample_id']=='123'
    assert index.identity(run)['sample_id']=='123'
    assert len(reads)==1


def test_warm_runs_stay_visible_for_exact_run_inspection_without_sample_assignment(tmp_path,monkeypatch):
    runs=[];paths={};catalog=[]
    for run in [1,2,3,4,5]:
        path=tmp_path/f'{run}.h5';paths[run]=path
        make_run(path,values=[456,123] if run<3 else [123],times=[0,72956] if run<3 else [0])
        reduced=tmp_path/f'PG3_{run}.gsa';reduced.write_text('# CrCl3 recorded run\n')
        runs.append(WatchedRun(run,str(reduced),'unchanged'))
        catalog.append({'location':f'/SNS/PG3/IPTS-1/nexus/PG3_{run}.nxs.h5',
                        'indexed':{'run_number':run,'sample_id':123}})
    rows=sample_catalog.parse_sample_catalog(catalog,'PG3','IPTS-1')
    monkeypatch.setitem(sample_catalog._CACHE,('PG3','IPTS-1'),(time.monotonic(),rows))
    def reader(ipts,run):return powgen_metadata.read_powgen_sample_identity(ipts,run,nexus_path=paths[run])
    index=PowgenSampleIndex('IPTS-1',nexus_path=lambda ipts,run:paths[run],identity_reader=reader,
                           metadata_index=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path))
    view=index.inventory(runs,catalog=True,representative_names=True)
    assert view['options'][0]['scan_count']==3
    assert view['unidentified_count']==view['identity_conflict_count']==2
    assert len(view['run_options'])==5
    assert all('Conflicting sample IDs' in row['title'] for row in view['run_options'][:2])
    assert '2 scans have conflicting sample IDs' in sample_identity_summary(view)
    assert 'no readable sample ID' not in sample_identity_summary(view)
    assert 'Single pattern' in sample_identity_summary(view)
    assert 'Live sample binding remains blocked' in sample_identity_summary(view)
    chosen=select_runs(runs,index,None,excluded_run_numbers=[1,3,4,5])
    assert [r.run_number for r in chosen]==[2]
    assert index.identity(chosen[0])['identity_status']=='conflict' # inspection never auto-assigns an ID
    from radar_pd_nova.sample_models import single_sample_cohort
    with pytest.raises(ValueError, match='verified sample ID'):
        single_sample_cohort([index.identity(chosen[0])])
    assert [r.run_number for r in select_runs(runs,index,['123'])]==[3,4,5]


def test_cli_parser_keeps_clock_evidence_and_static_conflict(tmp_path,monkeypatch):
    path=tmp_path/'cli.h5';path.write_bytes(b'cli fixture')
    monkeypatch.setattr(powgen_metadata.shutil,'which',lambda _: '/available/h5dump')
    data='''DATASET "/entry/sample/identifier" { DATA { 123 } }
DATASET "/entry/DASlogs/SampleId/value" { DATA { 456, 123 } }
DATASET "/entry/DASlogs/SampleId/time" { DATA { 0, 72956.399187466 }
 ATTRIBUTE "start" { DATA { "2020-01-01T12:00:00.755076667+00:00" } }
 ATTRIBUTE "units" { DATA { "second" } } }
DATASET "/entry/start_time" { DATA { "2020-01-01T12:00:00.755076667+00:00" } }
DATASET "/entry/end_time" { DATA { "2020-01-01T12:05:57.974721667+00:00" } }
'''
    def run(command,**kwargs):
        assert all('/entry/events' not in arg for arg in command)
        return SimpleNamespace(returncode=0,stdout='DATASET "a" { DATASPACE SIMPLE { ( 2 ) / ( 2 ) } }' if '-H' in command else data)
    monkeypatch.setattr(powgen_metadata.subprocess,'run',run)
    result=powgen_metadata.read_powgen_sample_identity('IPTS-1',1,nexus_path=path,backend='h5dump')
    assert result['identity_status']=='conflict'
    assert result['identity_time_scopes'][0]['ignored_after_count']==1
    assert 'during acquisition = 456' in result['identity_warning']
