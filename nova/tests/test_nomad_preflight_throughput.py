"""Bounded per-invocation NOMAD checks on synthetic files, never facility data."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import threading
import time

import pytest

from radar_pd_nova import nomad
from radar_pd_nova import nomad_controller as controller
from radar_pd_nova.powgen_watch import WatchDefinition
from test_nomad import profile


def inputs(tmp_path, monkeypatch, count=12):
    for run in range(1,count+1):
        text=f'Sample Run: {run}\n# Synthetic specimen\n# 2 Histograms\n'
        for bank,difc,angle in [(1,1000.,15.),(2,2861.51,31.2)]:
            text+=f'# Total flight path 20.5m, tth {angle}deg, DIFC {difc}\n'
            text+=f'BANK {bank} 2 2 SLOG 1000 1100 .0008 0 FXYE\n1000 -1 0.1\n1100 2 0.2\n'
        path=tmp_path/f'NOM_{run}.gsa';path.write_text(text)
        path.with_suffix('.json').write_text(json.dumps(dict(Instrument='NOMAD',
            Sample=dict(Runs=str(run),DummyInfo=False),Calibration=dict(Filename='synthetic'),Environment=dict(Name='air'))))
    supplied=tmp_path/'profile.instprm';supplied.write_text(profile(2,2861.51,31.2))
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(tmp_path)))
    monkeypatch.setattr(controller,'read_nomad_scan_metadata',lambda *args:dict(sample_id='A',sample_name='Synthetic'))
    return dict(selected_sample_ids=('A',),selected_banks=(2,),profile_path=supplied,profile_bank=2,
        profile_acknowledged=True,allow_exploratory=True,
        sample_index=SimpleNamespace(identity=lambda run:dict(sample_id='A',identity_status='resolved')))


def test_parallel_and_sequential_outcomes_match_including_ordered_errors(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch)
    # The last/first errors must be by run order, not by worker completion.
    for run in (2,5,9):
        path=tmp_path/f'NOM_{run}.gsa'
        path.write_text(path.read_text().replace('1100 2 0.2','1100 2 0'))
    original=nomad.inspect_reduced_setup
    def delayed(path,**kwargs):
        time.sleep(.002*(12-int(Path(path).stem.split('_')[1])))
        return original(path,**kwargs)
    monkeypatch.setattr(nomad,'inspect_reduced_setup',delayed)
    sequential=controller.preflight_nomad_experiment('IPTS-1',max_workers=1,**args)
    concurrent=controller.preflight_nomad_experiment('IPTS-1',max_workers=4,**args)
    assert concurrent==sequential
    assert not concurrent['ready'] and concurrent['scan_count']==12
    assert concurrent['profile_error'].index('NOM_2:')<concurrent['profile_error'].index('NOM_5:')<concurrent['profile_error'].index('NOM_9:')


def test_shared_profile_is_parsed_once_per_invocation_without_cross_check_cache(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch,64);calls=[];original=nomad._profile_blocks
    def parse(text):calls.append(1);return original(text)
    monkeypatch.setattr(nomad,'_profile_blocks',parse)
    assert controller.preflight_nomad_experiment('IPTS-1',**args)['ready']
    assert len(calls)==1
    args['profile_path'].write_text(profile(2,9999.99,31.2))
    assert not controller.preflight_nomad_experiment('IPTS-1',**args)['ready']
    assert len(calls)==2


def test_invalid_profile_domain_is_blocked_before_submission_and_valid_explicit_range_passes(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch,1)
    source=tmp_path/'NOM_1.gsa'
    source.write_text(source.read_text().replace('1100','10000'))
    supplied=args['profile_path']
    supplied.write_text(supplied.read_text().replace('sig-0:1','sig-0:-1')
                        .replace('sig-1:1','sig-1:4').replace('sig-2:1','sig-2:-1'))
    before_source=source.read_bytes();before_profile=supplied.read_bytes()
    result=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert not result['ready']
    assert 'Instrument Gaussian variance is non-positive' in result['profile_error']
    assert 'explicitly select a supported fitted range' in result['profile_error']
    result=controller.preflight_nomad_experiment('IPTS-1',native_limits=(2861.51*.8,2861.51*1.5),**args)
    assert result['ready'],result['profile_error']
    assert source.read_bytes()==before_source and supplied.read_bytes()==before_profile


def test_preflight_reports_existing_native_lower_tail_repair(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch,1)
    source=tmp_path/'NOM_1.gsa'
    source.write_text(source.read_text().replace('1100','6000'))
    supplied=args['profile_path']
    supplied.write_text(supplied.read_text().replace('sig-0:1','sig-0:-.25').replace('sig-2:1','sig-2:0'))
    result=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert result['ready'],result['profile_error']
    assert any('Native Gaussian-width safety will raise the lower fit limit' in s for s in result['quality_notices'])


def test_profile_byte_change_during_check_is_rejected_even_with_restored_stat(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch);supplied=args['profile_path'];stat=supplied.stat()
    original=nomad.validate_profile;changed=False;lock=threading.Lock()
    def change(*a,**kw):
        nonlocal changed
        result=original(*a,**kw)
        with lock:
            if not changed:
                raw=supplied.read_bytes();supplied.write_bytes(raw.replace(b'alpha:0.1',b'alpha:0.2'))
                os.utime(supplied,ns=(stat.st_atime_ns,stat.st_mtime_ns));changed=True
        return result
    monkeypatch.setattr(nomad,'validate_profile',change)
    result=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert not result['ready'] and 'profile changed during' in result['profile_error']
    assert supplied.stat().st_size==stat.st_size and supplied.stat().st_mtime_ns==stat.st_mtime_ns


def test_profile_change_during_metadata_read_is_rejected_before_ready(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch);supplied=args['profile_path']
    def metadata(*args):
        supplied.write_text(supplied.read_text().replace('alpha:0.1','alpha:0.2'))
        return {}
    monkeypatch.setattr(controller,'read_nomad_scan_metadata',metadata)
    result=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert not result['ready'] and 'profile changed during' in result['profile_error']


def test_invalid_shared_profile_is_parsed_once_and_keeps_bank_inventory(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch);args['profile_path'].write_text('Type:PNT\nBank:2\n')
    calls=[];original=nomad._profile_blocks
    def parse(text):calls.append(1);return original(text)
    monkeypatch.setattr(nomad,'_profile_blocks',parse)
    result=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert len(calls)==1 and not result['ready'] and result['available_banks']==[1,2]
    assert 'complete neutron TOF' in result['profile_error']


def test_concurrent_source_inspections_have_a_small_fixed_bound(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch,24);active=0;peak=0;lock=threading.Lock();original=nomad.inspect_reduced_setup
    def inspect(*a,**kw):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:
            time.sleep(.005)
            return original(*a,**kw)
        finally:
            with lock:active-=1
    monkeypatch.setattr(nomad,'inspect_reduced_setup',inspect)
    assert controller.preflight_nomad_experiment('IPTS-1',**args)['ready']
    assert 1<peak<=4


def test_unexpected_worker_failure_never_publishes_partial_readiness(tmp_path,monkeypatch):
    args=inputs(tmp_path,monkeypatch);original=nomad.inspect_reduced_setup
    def inspect(path,**kwargs):
        if Path(path).name=='NOM_3.gsa':raise RuntimeError('cancelled synthetic check')
        return original(path,**kwargs)
    monkeypatch.setattr(nomad,'inspect_reduced_setup',inspect)
    with pytest.raises(RuntimeError,match='cancelled synthetic check'):
        controller.preflight_nomad_experiment('IPTS-1',**args)


@pytest.mark.parametrize('workers',[0,5,True,'4'])
def test_invalid_worker_limits_fail_before_source_reads(monkeypatch,workers):
    monkeypatch.setattr(nomad,'directory_listing',lambda *a:pytest.fail('Invalid limit reached facility reads'))
    with pytest.raises(ValueError,match='from 1 to 4'):
        controller.preflight_nomad_experiment('IPTS-1',max_workers=workers)
