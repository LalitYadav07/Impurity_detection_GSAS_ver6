from copy import deepcopy
import json
import pickle

import pytest

from scripts.ndip_acquisition import annotate_gpx_records, validate_acquisition
from scripts.ndip_input_provenance import read_input_provenance, PREFIX
from scripts.ndip_outputs import collect_outputs


def packet():
    return {'$schema':'radar-pd-acquisition/v1','instrument':'PG3','run_number':50713,
        'sample_id':'18062','product_window_verified':True,'product_window_basis':'hashed whole-run recipe',
        'acquisition_window':{'start':'2021-08-04T21:41:12+00:00','end':'2021-08-04T21:45:38+00:00'},
        'channels':[{'quantity':'temperature','role':'measured','status':'recorded',
          'value':322.5178,'unit':'K','source':'/entry/DASlogs/BL11A:SE:SampleTemp/value',
          'minimum':322.509,'maximum':322.528,'coverage_fraction':1.0,'statistic':'time_weighted_step_hold'}]}


def records():
    return [[['PWDR run', [{'Rwp':2.71},[[1,2],[5,6],[1,1],[5,6]]]],
        ['Sample Parameters',{'Temperature':300.0,'Scale':[1,True]}], ['Comments',['original']]],
        [['Phases',{}],['phase',{'General':{'Cell':[True,4,4,4,90,90,90,64]}}]],
        [['Covariance',{'freshCOV':True,'Rvals':{'Rwp':2.71}}]]]


def test_temperature_handoff_changes_only_acquisition_metadata():
    original=records();changed=deepcopy(original)
    receipt=annotate_gpx_records(changed,packet())
    assert receipt['temperature_written'] and receipt['temperature_K']==322.5178
    assert changed[0][1][1]['Temperature']==322.5178
    assert changed[0][0]==original[0][0] and changed[1:]==original[1:]
    assert changed[0][1][1]['Scale']==original[0][1][1]['Scale']
    assert 'time_weighted_step_hold' in changed[0][2][1][-1]
    annotate_gpx_records(changed,packet())
    assert sum(item[0]=='RADAR-PD acquisition' for item in changed[0])==1
    assert len(changed[0][2][1])==2


def test_unqualified_interval_and_multihistograms_never_receive_a_temperature():
    p=packet();p['product_window_verified']=False
    data=records();receipt=annotate_gpx_records(data,p)
    assert not receipt['temperature_written'] and data[0][1][1]['Temperature']==300
    data=records();data.append(deepcopy(data[0]));data[-1][0][0]='PWDR other'
    assert annotate_gpx_records(data,packet())['status']=='multiple_histograms_need_mapping'
    assert data[0][1][1]['Temperature']==300 and data[-1][1][1]['Temperature']==300


@pytest.mark.parametrize('field,value',[('value',float('nan')),('coverage_fraction',.8),('role','requested'),('status','needs_mapping')])
def test_invalid_conditions_cannot_be_promoted_to_sample_parameters(field,value):
    p=packet();p['channels'][0][field]=value
    with pytest.raises(ValueError):validate_acquisition(p)


def test_publication_rewrites_a_copy_and_indexes_the_handoff(tmp_path):
    root=tmp_path/'run';root.mkdir();source=root/'seq_final_joint_polished.gpx'
    with source.open('wb') as f:
        for row in records():pickle.dump(row,f,protocol=1)
    before=source.read_bytes()
    (root/'input_provenance.json').write_text(json.dumps({'acquisition':packet(),'warnings':[]}))
    result=collect_outputs(root,tmp_path/'portal',mode='full',include_archive=False)
    assert source.read_bytes()==before
    artifact=result['artifacts']['gpx'][0]
    assert artifact['acquisition_handoff']['temperature_written']
    assert result['gpx_projects'][0]['acquisition_handoff']['temperature_K']==322.5178
    assert result['acquisition']['run_number']==50713


def test_existing_powder_provenance_is_backward_compatible_and_new_conditions_are_checked(tmp_path):
    path=tmp_path/'input.gsa';base={'schema':'radar-pd-input-provenance/v1','warnings':[]}
    path.write_bytes(b'title\n'+PREFIX+json.dumps(base).encode()+b'\nBANK 1\n')
    assert read_input_provenance(path)==base
    base['acquisition']=packet();path.write_bytes(b'title\n'+PREFIX+json.dumps(base).encode()+b'\nBANK 1\n')
    assert read_input_provenance(path)['acquisition']['channels'][0]['value']==322.5178
