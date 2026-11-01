import pytest
from radar_pd_nova.webmon.mapping import reviewed_mapping


def args():
    return dict(source={'id':'published','source_sha256':'a'*64,'bank':'2'},receipt={'content_sha256':'b'*64,'identity':{'instrument':'PG3','ipts':1,'run':2,'epoch':'live'}},bank=2,export_difc=20000.,normalization='producer distribution per d',producer_sha256='c'*64,evidence='Instrument producer configuration inspected',accepted=True)


def test_explicit_mapping_is_bound_to_source_snapshot_and_remains_unqualified():
    mapping=reviewed_mapping(**args())
    assert mapping['source_gpx_sha256']=='a'*64 and mapping['snapshot_sha256']=='b'*64
    assert mapping['scientifically_qualified'] is False and mapping['forecast_enabled'] is False
    assert mapping['native_preparation_required'] is True


@pytest.mark.parametrize('key,value',[('bank',3),('export_difc',float('nan')),('accepted',False),('producer_sha256','unknown'),('normalization',''),('evidence','')])
def test_mapping_rejects_unknown_or_inconsistent_inputs(key,value):
    payload=args();payload[key]=value
    with pytest.raises(ValueError):reviewed_mapping(**payload)
