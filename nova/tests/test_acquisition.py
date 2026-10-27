"""Measured readbacks must retain their identity, interval and provenance."""
from copy import deepcopy
import hashlib
import json

from radar_pd_nova.acquisition import acquisition_context, stage_manual_acquisition, PREFIX


def metadata():
    window = {'start':'2021-08-04T21:41:12+00:00','end':'2021-08-04T21:45:38+00:00'}
    channel = {'quantity':'temperature','role':'measured','status':'recorded','value':322.5178,
        'minimum':322.509,'maximum':322.528,'unit':'K','source':'/entry/DASlogs/BL11A:SE:SampleTemp/value',
        'coverage_fraction':1.0,'statistic':'time_weighted_step_hold','acquisition_window':window,
        'requires_binding':False,'trajectory':{'seconds':list(range(5000))}}
    return {'nexus_path':'/SNS/PG3/IPTS-2767/nexus/PG3_50713.nxs.h5','sample_id':'18062','sample_name':'Al2O3+MgO',
        'temperature':channel,'conditions':{'$schema':'radar-pd-conditions/v1','registry':'SNS-PG3-NOM-conditions/2',
        'acquisition_window':window,'window_scope':'parent_run; verify reduction does not select a narrower time interval',
        'channels':[channel]}}


def test_verified_whole_run_compacts_the_actual_readback_without_trajectories(tmp_path):
    m=metadata();source=tmp_path/'PG3_50713.gsa'
    script=source.with_suffix('.py');script.write_text(f"SNSPowderReduction(Filename={m['nexus_path']!r}, FilterBadPulses=10)\n")
    packet=acquisition_context(m,instrument='PG3',run_number=50713,source=source,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    assert packet['product_window_verified']
    assert packet['channels'][0]['value']==322.5178
    assert 'trajectory' not in packet['channels'][0]
    assert len(json.dumps(packet)) < 1600


def test_filtered_dynamic_or_nonexecuted_reductions_never_qualify_parent_temperature(tmp_path):
    m=metadata();source=tmp_path/'PG3_50713.gsa';script=source.with_suffix('.py')
    for text in [
        f"SNSPowderReduction(Filename={m['nexus_path']!r}, FilterByTimeStart=60)",
        f"SNSPowderReduction(Filename={m['nexus_path']!r}, **settings)",
        f"SNSPowderReduction(Filename={m['nexus_path']!r}, InputWorkspace='filtered')",
        f"def unused():\n    SNSPowderReduction(Filename={m['nexus_path']!r})",
        "SNSPowderReduction(Filename='/SNS/PG3/IPTS-999/nexus/PG3_1.nxs.h5')",
        f"FilterEvents(InputWorkspace='x')\nSNSPowderReduction(Filename={m['nexus_path']!r})",
    ]:
        script.write_text(text)
        packet=acquisition_context(m,instrument='PG3',run_number=50713,source=source,
            script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
        assert not packet['product_window_verified']
    script.write_text(f"SNSPowderReduction(Filename={m['nexus_path']!r})")
    assert not acquisition_context(m,instrument='PG3',run_number=50713,source=source,script_sha256='a'*64)['product_window_verified']


def test_missing_interval_and_unbound_or_partial_channels_remain_unassigned(tmp_path):
    m=metadata();m['conditions']['acquisition_window']={}
    assert acquisition_context(m,instrument='PG3',run_number=50713,source=tmp_path/'x') is None
    for field,value in [('coverage_fraction',.7),('value',float('nan')),('status','needs_mapping'),('role','requested')]:
        m=metadata();m['conditions']['channels'][0][field]=value
        packet=acquisition_context(m,instrument='NOM',run_number=1,source=tmp_path/'x')
        assert packet['channels']==[]


def test_manual_staging_preserves_numeric_bytes_and_leaves_the_facility_source_unchanged(tmp_path):
    source=tmp_path/'original.gsa';raw=b'title\r\nBANK 1 2 2 CONST 1 1 0 0 FXYE\r\n1 2 3\r\n2 3 4\r\n';source.write_bytes(raw)
    packet=acquisition_context(metadata(),instrument='NOM',run_number=50713,source=source)
    staged=stage_manual_acquisition(source,tmp_path/'staged.gsa',packet,
        {'schema':'radar-pd-input-provenance/v1','warnings':[]})
    assert source.read_bytes()==raw
    assert b''.join(l for l in staged.read_bytes().splitlines(keepends=True) if not l.startswith(PREFIX))==raw
    receipt=json.loads(next(l[len(PREFIX):] for l in staged.read_bytes().splitlines() if l.startswith(PREFIX)))
    assert receipt['acquisition']['channels'][0]['value']==322.5178
