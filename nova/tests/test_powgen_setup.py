"""Run/bank contracts use real reduced text and packaged calibration files."""
from dataclasses import replace
from pathlib import Path
import json
import pytest

from radar_pd_nova.powgen_setup import (
    ReducedBank, ReducedSetup, inspect_reduced_setup, resolve_bank_profile,
    reduction_arguments, stage_bank, single_bank_profile, load_catalog,
    DATA_ROOT, text_hash,
)
from radar_pd_nova.powgen_controller import PowgenExperimentSettings, PowgenWatchController, preflight_powgen_experiment
from radar_pd_nova.powgen_watch import WatchDefinition, WatchState
from radar_pd_nova import powgen_controller
from test_powgen_controller import FakeGalaxyService


def source(tmp_path, run=43124, banks=(2,), wavelength=1.5, cycle='2019_1_11A_CAL', calibration='PG3_PAC.h5', frequency=60, mode='HighRes'):
    p=tmp_path/f'PG3_{run}.gsa'
    p.write_text(f'Sample Run: {run} Vanadium Run: 42839 Wavelength: {wavelength} A\n'
        f'# calibration test\n# {len(banks)} Histograms\n# Sample Temperature: 300 K Freq: {frequency} Hz Guide: -54 mm\n'
        + ''.join(f'BANK {b} 2 2 SLOG 10000 10020 .001 0 FXYE\n 10000.0 7.25 0.4\n 10010.0 -0.25 0.0\n' for b in banks))
    p.with_suffix('.py').write_text(f"SNSPowderReduction(CalibrationFile='/SNS/PG3/shared/CALIBRATION/{cycle}/{calibration}', CharacterizationRunsFile='/SNS/PG3/shared/CALIBRATION/{cycle}/PG3_char-{mode}.txt', SaveAs='gsas topas')\n")
    return p


def test_2019_case_uses_header_bank_not_profile_local_bank(tmp_path):
    p=source(tmp_path);setup=inspect_reduced_setup(p);profile=resolve_bank_profile(setup,2)
    assert profile.profile_filename=='2019A_HighRes_60HzB2_CWL1p5.instprm'
    assert profile.source_bank==2 and profile.profile_bank==1
    assert profile.sha256=='31438c892c89f6955a53461645e5f03c64fcc0bcd2194a62aa888bd8767e5941'
    data,inst=stage_bank(p,setup,2,profile,tmp_path/'staged')
    assert 'BANK 1 2 2' in data.read_text()
    assert ' 10000.0 7.25 0.4\n 10010.0 -0.25 0.0\n' in data.read_text()
    assert 'Bank:1.0' in inst.read_text()
    assert json.loads((tmp_path/'staged/PG3_43124_bank2.json').read_text())['source_bank']==2


def test_real_2017_two_bank_mapping_and_coefficients_are_distinct(tmp_path):
    p=source(tmp_path,37875,(3,4),cycle='2017_1_2_11A_CAL',calibration='PG3_PAC_2BANKS.h5')
    setup=inspect_reduced_setup(p);assert [b.bank for b in setup.banks]==[3,4]
    profiles=[resolve_bank_profile(setup,b) for b in (3,4)]
    assert [p.profile_bank for p in profiles]==[3,4]
    texts=[single_bank_profile(p) for p in profiles]
    assert texts[0]!=texts[1]
    assert all('INS   BANK      1' in t and 'INS  1PRCF' in t for t in texts)
    assert all('INS  2' not in t for t in texts)
    for b,profile in zip((3,4),profiles):
        data,_=stage_bank(p,setup,b,profile,tmp_path/f'b{b}')
        assert data.read_text().count('BANK ')==1
        assert '# 1 Histograms' in data.read_text()


@pytest.mark.parametrize('change,pattern',[
    ({'frequency':30},'No validated'),({'wavelength':1.333},'No validated'),
    ({'mode':'HI','guide':-54},'Guide position'),({'cycle':''},'cycle is missing'),
    ({'mode':''},'mode is missing'),({'cycle':'2023_2_11A_CAL'},'No validated'),
    ({'grouping_files':('/SNS/PG3/custom.xml',)},'Custom detector grouping'),
])
def test_no_nearest_or_unverified_fallback(tmp_path,change,pattern):
    setup=replace(inspect_reduced_setup(source(tmp_path)),**change)
    with pytest.raises(ValueError,match=pattern):resolve_bank_profile(setup,2)


def test_reduction_script_is_never_executed_and_non_gsas_second_call_is_ignored(tmp_path):
    marker=tmp_path/'MUST_NOT_EXIST'
    text=f"__import__('pathlib').Path({str(marker)!r}).touch()\nSNSPowderReduction(CalibrationFile='/right', SaveAs='gsas')\nSNSPowderReduction(CalibrationFile='/wrong', GroupingFile='/custom', SaveAs='topas')"
    args=reduction_arguments(text)
    assert args==[{'CalibrationFile':'/right','SaveAs':'gsas'}]
    assert not marker.exists()
    assert reduction_arguments("SNSPowderReduction(CalibrationFile=make_path())")==[{}]


def test_legacy_calfilename_keyword_and_multiple_cycle_conflict(tmp_path):
    p=source(tmp_path)
    p.with_suffix('.py').write_text("LoadCalFile(CalFilename='/SNS/PG3/2013_2_11A_CAL/x.cal')")
    assert inspect_reduced_setup(p).cycle=='2013_2_11A_CAL'
    with p.with_suffix('.py').open('a') as f:f.write("\nSNSPowderReduction(CalibrationFile='/SNS/PG3/2019_1_11A_CAL/y.h5')")
    with pytest.raises(ValueError,match='Conflicting'):inspect_reduced_setup(p)


def test_unsupported_sidecar_syntax_is_diagnostic(tmp_path):
    p=source(tmp_path);p.with_suffix('.py').write_text('print "Python 2 script"')
    with pytest.raises(ValueError,match='syntax'):inspect_reduced_setup(p)


def test_wrong_run_duplicate_banks_and_truncation_are_rejected(tmp_path):
    p=source(tmp_path)
    p.write_text(p.read_text().replace('Sample Run: 43124','Sample Run: 1'))
    with pytest.raises(ValueError,match='do not agree'):inspect_reduced_setup(p)
    p=source(tmp_path,banks=(2,2))
    with pytest.raises(ValueError,match='duplicate'):inspect_reduced_setup(p)
    p=source(tmp_path);setup=inspect_reduced_setup(p);profile=resolve_bank_profile(setup,2)
    p.write_text(p.read_text().rsplit('\n',2)[0]+'\n')
    with pytest.raises(ValueError,match='Incomplete'):
        inspect_reduced_setup(p)
    with pytest.raises(ValueError,match='changed after preflight'):stage_bank(p,setup,2,profile,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_single_declared_histogram_does_not_hide_trailing_or_duplicate_banks(tmp_path):
    for banks in [(2,3),(2,2)]:
        p=source(tmp_path,banks=banks)
        p.write_text(p.read_text().replace('# 2 Histograms','# 1 Histograms'))
        with pytest.raises(ValueError,match='duplicate|incomplete'):
            inspect_reduced_setup(p)


def test_saved_xml_can_supply_missing_receipt_and_cannot_override_conflicting_python(tmp_path):
    p=source(tmp_path)
    xml=p.with_suffix('.xml')
    xml.write_text('''<receipt><instrument_name>PG3</instrument_name><runnumber>43124</runnumber>
    <calibrationfile>/SNS/PG3/shared/CALIBRATION/2019_1_11A_CAL/PG3_PAC.h5</calibrationfile>
    <characterizationrunsfile>/SNS/PG3/shared/CALIBRATION/2019_1_11A_CAL/PG3_char-HighRes.txt</characterizationrunsfile>
    <saveas>gsas topas</saveas></receipt>''')
    assert inspect_reduced_setup(p).reduction_xml_sha256
    p.with_suffix('.py').unlink()
    setup=inspect_reduced_setup(p)
    assert setup.cycle=='2019_1_11A_CAL' and setup.mode=='HR'
    profile=resolve_bank_profile(setup,2)
    xml.write_text(xml.read_text()+'\n')
    with pytest.raises(ValueError,match='XML changed'):
        stage_bank(p,setup,2,profile,tmp_path/'out')
    p=source(tmp_path,calibration='different.h5')
    with pytest.raises(ValueError,match='receipts disagree'):
        inspect_reduced_setup(p)


def test_file_change_and_checksum_change_fail_before_staging(tmp_path):
    p=source(tmp_path);setup=inspect_reduced_setup(p);profile=resolve_bank_profile(setup,2)
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed after preflight'):stage_bank(p,setup,2,profile,tmp_path/'out')
    with pytest.raises(ValueError,match='changed after resolution'):single_bank_profile(replace(profile,sha256='0'*64))


def test_legacy_pab_tables_survive_bank_extraction(tmp_path):
    from radar_pd_nova.powgen_setup import BankProfile
    p=DATA_ROOT/'PG60HR_2011B.prm'
    raw=p.read_text();bank=1
    profile=BankProfile('legacy-test',p,text_hash(p.read_bytes()),bank,bank,'2011_2_11A_CAL',.533,60,'HR','facility')
    text=single_bank_profile(profile)
    tables=[l for l in raw.splitlines() if l.startswith('INS  1PAB')]
    assert tables and all(l in text for l in tables)


def test_all_packaged_profiles_are_hash_verified():
    catalog=load_catalog()
    for resource in catalog['resources']:
        assert text_hash((DATA_ROOT/resource['resource']).read_bytes())==resource['sha256']
    for rule in catalog['rules']:
        assert rule['resource'] in {r['resource'] for r in catalog['resources']}


def local_controller(monkeypatch,tmp_path,**settings):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda _:str(tmp_path)))
    monkeypatch.setattr(powgen_controller,'read_powgen_scan_metadata',lambda *_:{})
    return PowgenWatchController(FakeGalaxyService(),PowgenExperimentSettings('IPTS-2767','history-1','config-hda','',**settings))


def test_bank_selection_backfill_and_checkpoint_identity(monkeypatch,tmp_path):
    source(tmp_path,37874,(3,4),cycle='2017_1_2_11A_CAL',calibration='PG3_PAC_2BANKS.h5')
    source(tmp_path,37875,(3,4),cycle='2017_1_2_11A_CAL',calibration='PG3_PAC_2BANKS.h5')
    c=local_controller(monkeypatch,tmp_path,initial_backfill_limit=1)
    runs=c.discover();assert [r.run_id for r in runs]==['PG3_37875_BANK3','PG3_37875_BANK4']
    records=[c.submit(r) for r in runs]
    assert {r.inputs.bank for r in records}=={'3','4'}
    assert len({s.idempotency_token for s in c.service.snapshots})==2
    assert all(Path(r.inputs.data_path).read_text().count('BANK ')==1 for r in records)
    assert c.discover()==[]
    state=WatchState.from_dict(c.state.as_dict());assert len(state.submitted)==2
    c=local_controller(monkeypatch,tmp_path,selected_banks=(4,))
    assert [r.run_id for r in c.discover()]==['PG3_37874_BANK4','PG3_37875_BANK4']


def test_preflight_checks_earlier_selected_runs_and_wavelength_conflicts(monkeypatch,tmp_path):
    source(tmp_path,43124)
    source(tmp_path,63764,(2,),cycle='2026-2_11A_CAL')
    c=local_controller(monkeypatch,tmp_path)
    result=preflight_powgen_experiment('IPTS-2767')
    assert result['ready'] and result['analysis_count']==2
    assert result['profile_filename']=='2 calibrated profiles'
    result=preflight_powgen_experiment('IPTS-2767',requested_wavelength='0.8')
    assert not result['ready'] and result['blocked_count']==2
    p=tmp_path/'PG3_43124.py';p.unlink()
    result=preflight_powgen_experiment('IPTS-2767')
    assert not result['ready'] and 'PG3_43124' in result['profile_error']
    result=preflight_powgen_experiment('IPTS-2767',initial_backfill_limit=1)
    assert result['ready'] and result['analysis_count']==1


def test_manual_profile_cannot_cover_mixed_setups(monkeypatch,tmp_path):
    source(tmp_path,43124);source(tmp_path,63764,(2,),cycle='2026-2_11A_CAL')
    local_controller(monkeypatch,tmp_path)
    r=preflight_powgen_experiment('IPTS-2767',instrument_profile_name='fallback.instprm')
    assert not r['ready'] and 'mixed calibration' in r['profile_error']


def test_independent_bank_traces_and_heatmap_rows():
    from radar_pd_nova.results import experiment_phase_fraction_figure,experiment_phase_heatmap_figure,experiment_fit_quality_figure
    rows=[{'run_id':f'PG3_{n}_BANK{b}','run_number':n,'bank':b,'rwp':b+n/100,
           'phases':[{'label':'MgO','weight_percent':b*10}]} for n in [1,2] for b in [3,4]]
    fraction=experiment_phase_fraction_figure(rows)
    assert len(fraction.data)==2
    assert {t.name for t in fraction.data}=={'MgO · Bank 3','MgO · Bank 4'}
    assert set(experiment_phase_heatmap_figure(rows).data[0].y)=={'MgO · Bank 3','MgO · Bank 4'}
    assert len(experiment_fit_quality_figure(rows).data)==2


def test_archived_folder_choices_are_bounded_and_not_merged(monkeypatch):
    from types import SimpleNamespace
    from radar_pd_nova.powgen_controller import discover_powgen_samples
    from radar_pd_nova import powgen_controller as module
    class Entry:
        def __init__(self,name):self.name=name
        def is_dir(self,**kwargs):
            assert self.name in {'2016','2017-B','2019_B'}
            assert kwargs=={'follow_symlinks':False}
            return True
    class Listing:
        def __enter__(self):return iter([Entry(n) for n in ['2016','2017-B','2019_B','cache','nexus','PG3_1.gsa']])
        def __exit__(self,*args):pass
    paths=[]
    monkeypatch.setattr(module,'bounded_directory_listing',lambda p: paths.append(p) or [])
    monkeypatch.setattr(module,'_scandir',lambda _:Listing())
    monkeypatch.setattr(Path,'is_dir',lambda _:True)
    index=SimpleNamespace(inventory=lambda _: {'options':[],'scan_count':0,'unidentified_count':0})
    result=discover_powgen_samples('IPTS-2767',index,'shared/autoreduce/2017-B')
    assert paths==['/SNS/PG3/IPTS-2767/shared/autoreduce/2017-B']
    assert {x['value'] for x in result['source_options']}=={'shared/autoreduce','shared/autoreduce/2016','shared/autoreduce/2017-B','shared/autoreduce/2019_B'}


def test_checkpoint_retains_bank_profile_hash_and_original_source(monkeypatch,tmp_path):
    source(tmp_path)
    c=local_controller(monkeypatch,tmp_path)
    r=c.discover()[0];c.submit(r)
    p=c.service.uploaded[-1][0]['submitted']['PG3_43124']['scan_metadata']['powgen_bank_provenance']
    assert p['source_bank']==2 and p['staged_bank']==1
    assert p['source_path']==str(tmp_path/'PG3_43124.gsa')
    assert p['profile']['sha256']==text_hash((DATA_ROOT/p['profile']['path'].rsplit('/',1)[-1]).read_bytes())
    assert len(p['source_sha256'])==64


def test_changed_reduction_script_invalidates_resolved_setup(tmp_path):
    p=source(tmp_path);setup=inspect_reduced_setup(p);profile=resolve_bank_profile(setup,2)
    assert len(setup.reduction_script_sha256)==64
    p.with_suffix('.py').write_text(p.with_suffix('.py').read_text()+'# changed\n')
    with pytest.raises(ValueError,match='script changed after preflight'):
        stage_bank(p,setup,2,profile,tmp_path/'out')
    assert not (tmp_path/'out').exists()
