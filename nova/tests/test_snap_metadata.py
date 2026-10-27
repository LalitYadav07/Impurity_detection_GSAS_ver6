from radar_pd_nova.facility_metadata import parse_scan_table
from radar_pd_nova.sample_catalog import parse_sample_catalog


def test_real_snap_table_shape_supplies_names_not_unqualified_pressure():
    raw=b'RUN,IPTS,TITLE,DURATION,pCHARGE,START TIME,END TIME,WAVELENGTH,TEMPERATURE,PRESSUREA,TOTAL COUNTS,USER,SAMPLE\n42,IPTS-1,"oxide, 50 bar",30,100,start,end,2.1,300,50,1000,user,oxide\n'
    table=parse_scan_table(raw,source='local-test.csv',instrument='SNAP',ipts='IPTS-1')
    assert table.schema=='snap_runsummary' and table.row_count==1
    fields=table.fields(42)['fields']
    assert fields['sample_name']=='oxide' and fields['run_title']=='oxide, 50 bar'
    assert 'sample_id' not in fields and 'pressure' not in fields and 'temperature' not in fields


def test_catalog_unsigned_unset_id_is_not_a_selectable_specimen():
    rows=[{'location':'/SNS/SNAP/IPTS-1/nexus/SNAP_42.nxs.h5','metadata':{'entry':{
        'run_number':42,'experiment_identifier':'IPTS-1','instrument':{'name':'SNAP'},
        'sample':{'identifier':4294967295,'name':'No sample'},'title':'NIST Si standard'}}}]
    result=parse_sample_catalog(rows,'SNAP','IPTS-1')
    assert result[42]['sample_id']=='' and result[42]['identity_status']=='unidentified'
    assert result[42]['run_title']=='NIST Si standard'
