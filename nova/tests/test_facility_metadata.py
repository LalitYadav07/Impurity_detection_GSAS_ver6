"""Scientific metadata edge cases expressed as synthetic public fixtures."""
from pathlib import Path
import csv
import io

import pytest

from radar_pd_nova.facility_metadata import (
    ExperimentMetadataIndex, parse_scan_table, qualify_temperature,
    read_runinfo, resolve_identity, resolve_nexus_path, sample_identifier, title_role,
)


def table(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['RUN','IPTS#','ITEMS-ID','SAMPLE','WAVELENGTH','VAN RUN','EMPTY CAN RUN','GUIDE','TEMP'])
    writer.writerows(rows)
    return stream.getvalue().encode()


def parse(raw):
    return parse_scan_table(raw, source='synthetic.csv', instrument='PG3', ipts='IPTS-1')


def test_missing_middle_column_cannot_shift_temperature_or_identity():
    raw = table([[42,'IPTS-1','123','sample',1.5,10,-54,300]])
    result = parse(raw)
    assert result.rejected_count == 1 and not result.rows
    assert result.issues[0]['code'] == 'row_width'


def test_reference_conflict_keeps_consistent_identity_and_each_receipt():
    result = parse(table([[42,'IPTS-1','123','sample',1.5,10,20,-54,300],
                          [42,'IPTS-1','123','sample',1.5,11,20,-54,300]]))
    fields = result.fields(42)
    assert fields['fields']['sample_id'] == '123'
    assert fields['conflicts'] == ['vanadium_run']
    assert 'vanadium_run' not in fields['fields']
    assert {o['line'] for o in fields['sources']} == {2,3}


def test_conflicting_id_rows_never_choose_first_or_last():
    result = parse(table([[42,'IPTS-1','123','same name',1.5,10,20,-54,300],
                          [42,'IPTS-1','124','same name',1.5,10,20,-54,300]]))
    identity = resolve_identity(o for o in result.fields(42)['sources'] if o['field']=='sample_id')
    assert identity['identity_status']=='conflict' and identity['sample_id']==''


def test_foreign_proposal_and_scan_notes_are_not_sample_ids():
    result = parse(table([[42,'IPTS-2','123','sample',1.5,10,20,-54,300],
                          [43,'IPTS-1','parameters: x=3','sample',1.5,10,20,-54,300]]))
    assert result.foreign_count==1 and 42 not in result.rows
    assert 'sample_id' not in result.fields(43)['fields']


def test_nomad_scan_labels_do_not_become_sample_ids():
    result = parse_scan_table(b'#Scan,IPTS,title\n42,1,123\n',source='los.csv',instrument='NOM',ipts='IPTS-1')
    assert result.fields(42)['fields']=={'run_title':'123'}


@pytest.mark.parametrize('value',['None','NONE','-1','0','N/A','nan','(unset)','parameters: x=1'])
def test_missing_sentinels(value):
    assert sample_identifier(value)==''


def test_exact_ids_and_fallback_corroboration():
    assert sample_identifier('00123')=='00123'
    assert sample_identifier('A-123')=='A-123'
    assert resolve_identity([{'value':'None'},{'value':'123'}])['sample_id']=='123'


def test_legacy_paths_deduplicate_same_file_but_block_distinct_files(tmp_path):
    root=tmp_path/'PG3/IPTS-1';old=root/'data/PG3_42_event.nxs'
    old.parent.mkdir(parents=True);old.write_text('metadata fixture')
    alias=root/'0/42/NeXus/PG3_42_event.nxs';alias.parent.mkdir(parents=True);alias.symlink_to(old)
    assert resolve_nexus_path('PG3','IPTS-1',42,facility_root=tmp_path)==old
    modern=root/'nexus/PG3_42.nxs.h5';modern.parent.mkdir();modern.write_text('different source')
    with pytest.raises(ValueError,match='distinct raw'):
        resolve_nexus_path('PG3','IPTS-1',42,facility_root=tmp_path)


def xml(identifier='123',run=42):
    return f'<RunInfo instrument="PG3" runnumber="{run}"><SampleInfo Identifier="{identifier}" Name="sample"/></RunInfo>'


def test_xml_fills_gap_and_later_conflict_invalidates_index_cache(tmp_path):
    from radar_pd_nova.powgen_samples import PowgenSampleIndex
    from radar_pd_nova.powgen_watch import WatchedRun
    sources=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path)
    path=sources.xml_path(42);path.parent.mkdir(parents=True);path.write_text(xml())
    index=PowgenSampleIndex('IPTS-1',nexus_path=lambda *_:tmp_path/'absent.h5',metadata_index=sources)
    run=WatchedRun(42,'PG3_42.gsa','f')
    assert index.identity(run)['sample_id']=='123'
    csvpath=tmp_path/'shared/autoreduce/PG3_IPTS-1_runsummary.csv';csvpath.parent.mkdir(parents=True)
    csvpath.write_bytes(table([[42,'IPTS-1','124','sample',1.5,10,20,-54,300]]))
    sources.refresh(force=True)
    result=index.identity(run)
    assert result['sample_id']=='' and result['identity_status']=='conflict'


def test_wrong_run_or_unsafe_xml_is_not_a_fallback(tmp_path):
    path=tmp_path/'run.xml';path.write_text(xml(run=43))
    with pytest.raises(ValueError,match='disagrees'):read_runinfo(path,'PG3',42)
    path.write_text('<!DOCTYPE x [<!ENTITY x "123">]>'+xml())
    with pytest.raises(ValueError,match='Unsupported'):read_runinfo(path,'PG3',42)


@pytest.mark.parametrize('title',['sample_ramp_to_150C','sample_at_25C_cooling','sample-cool-to-0C'])
def test_separator_normalization_for_run_role(title):
    assert title_role(title)=='ramp'


def test_suspect_temperature_remains_unavailable_and_recorded_ramp_retains_range():
    assert qualify_temperature({'value':0,'unit':'K'})[0]is None
    assert qualify_temperature({'value':135,'unit':'K'},'at temperature 134 C.')[0]is None
    assert qualify_temperature({'value':25,'unit':''})[0]is None
    assert qualify_temperature({'value':300,'unit':'K','minimum':float('nan')})[0]is None
    metric,notices=qualify_temperature({'value':143,'unit':'K','minimum':134,'maximum':152})
    assert metric['quality']=='changing' and notices and metric['minimum']==134
    assert qualify_temperature({'value':300,'unit':'K','converted_from':'C'},'at temperature 26.85 C.')[0]


def test_table_size_and_schema_are_explicit(monkeypatch):
    from radar_pd_nova import facility_metadata as fm
    monkeypatch.setattr(fm,'MAX_TABLE_BYTES',10)
    with pytest.raises(ValueError,match='bounded'):parse(b'x'*11)


def test_bad_name_encoding_does_not_guess_text_or_erase_ascii_identity():
    raw=table([[42,'IPTS-1','123','sample',1.5,10,20,-54,300]]).replace(b'sample',b'sampl\xe9')
    result=parse(raw)
    assert result.fields(42)['fields']['sample_id']=='123'
    assert 'sample_name' not in result.fields(42)['fields']
    assert result.issues[0]['code']=='unqualified_text_encoding'


def test_duplicate_nonidentity_column_is_omitted_without_guessing():
    result=parse(b'RUN,IPTS#,ITEMS-ID,SAMPLE,TEMP,TEMP\n42,IPTS-1,123,name,300,400\n')
    assert result.fields(42)['fields']['sample_id']=='123'
    assert 'table_temperature' not in result.fields(42)['fields']
    with pytest.raises(ValueError,match='identity key'):
        parse(b'RUN,RUN,IPTS#,ITEMS-ID,SAMPLE\n42,43,IPTS-1,123,name\n')


def test_legacy_carriage_return_csv_is_parsed_without_shifting_fields():
    result=parse(b'RUN,IPTS#,ITEMS-ID,SAMPLE\r42,IPTS-1,123,"name, with comma"\r')
    assert result.fields(42)['fields']['sample_id']=='123'
    assert result.fields(42)['fields']['sample_name']=='name, with comma'


def test_transient_table_read_failure_retries_unchanged_file(tmp_path, monkeypatch):
    from radar_pd_nova import facility_metadata as fm
    path=tmp_path/'shared/autoreduce/PG3_IPTS-1_runsummary.csv'
    path.parent.mkdir(parents=True)
    path.write_bytes(table([[42,'IPTS-1','123','sample',1.5,10,20,-54,300]]))
    original=fm.read_scan_table
    def unavailable(*args):
        raise OSError('temporary filesystem read failure')
    monkeypatch.setattr(fm,'read_scan_table',unavailable)
    index=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path)
    index.refresh()
    assert not index.tables and index._retry_read
    monkeypatch.setattr(fm,'read_scan_table',original)
    index._last_refresh=-float('inf')
    index.refresh()
    assert index.tables[0].fields(42)['fields']['sample_id']=='123'


def test_unreadable_raw_identity_uses_independent_xml_with_notice(tmp_path):
    from radar_pd_nova.powgen_samples import PowgenSampleIndex
    from radar_pd_nova.powgen_watch import WatchedRun
    sources=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path)
    xmlpath=sources.xml_path(42);xmlpath.parent.mkdir(parents=True);xmlpath.write_text(xml())
    raw=tmp_path/'raw.h5';raw.write_bytes(b'being written')
    def unavailable(*args):raise OSError('raw metadata still being written')
    index=PowgenSampleIndex('IPTS-1',nexus_path=lambda *_:raw,identity_reader=unavailable,metadata_index=sources)
    result=index.identity(WatchedRun(42,'PG3_42.gsa','f'))
    assert result['sample_id']=='123'
    assert 'Raw sample identity is unavailable' in result['metadata_notices'][0]
    assert not index._cache


def test_recovered_table_invalidates_previously_resolved_identity(tmp_path, monkeypatch):
    from radar_pd_nova import facility_metadata as fm
    from radar_pd_nova.powgen_samples import PowgenSampleIndex
    from radar_pd_nova.powgen_watch import WatchedRun
    path=tmp_path/'shared/autoreduce/PG3_IPTS-1_runsummary.csv'
    path.parent.mkdir(parents=True)
    path.write_bytes(table([[42,'IPTS-1','124','sample',1.5,10,20,-54,300]]))
    raw=tmp_path/'raw.h5';raw.write_bytes(b'fixture')
    original=fm.read_scan_table
    def unavailable(*args):raise OSError('temporary read failure')
    monkeypatch.setattr(fm,'read_scan_table',unavailable)
    sources=ExperimentMetadataIndex('PG3','IPTS-1',root=tmp_path)
    index=PowgenSampleIndex('IPTS-1',nexus_path=lambda *_:raw,identity_reader=lambda *_:{'sample_id':'123'},metadata_index=sources)
    run=WatchedRun(42,'PG3_42.gsa','f')
    assert index.identity(run)['sample_id']=='123'
    monkeypatch.setattr(fm,'read_scan_table',original)
    sources._last_refresh=-float('inf')
    assert index.identity(run)['identity_status']=='conflict'
