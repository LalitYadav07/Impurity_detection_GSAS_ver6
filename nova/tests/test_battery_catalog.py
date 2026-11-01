from copy import deepcopy
import json

import pytest

from radar_pd_nova.battery import catalog as m

LOCATION='/SNS/NOM/IPTS-42/nexus/NOM_10.nxs.h5'


class Response:
    def __init__(self, data=None, status=200, headers=None, raw=None):
        self.status_code=status;self.headers=headers or {}
        self.raw=raw if raw is not None else json.dumps(data).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,size):yield self.raw


class Session:
    def __init__(self,*responses):self.responses=list(responses);self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs));response=self.responses.pop(0)
        if isinstance(response,Exception):raise response
        return response


def datafile(**changes):
    data=dict(location=LOCATION,facility='SNS',instrument='NOM',experiment='IPTS-42',indexed={'run_number':10},
        metadata={'entry':{'run_number':'10','experiment_identifier':'IPTS-42','title':'cell 1',
            'start_time':'2020-01-01T00:00:00.123456789Z','sample':{'identifier':123,'chemical_formula':'registered text'}}})
    data.update(changes);return data


def proposal():
    return dict(id='IPTS-42',title='Battery study',abstract='A declared scientific purpose.',
        instruments=[{'id':'NOM','contacts':['private person']}],investigators=['do not export'],contact='private email')


def test_exact_scoped_get_projects_fields_and_never_exports_people_or_credentials():
    session=Session(Response(proposal()),Response(datafile()))
    client=m.CatalogClient(session=session,token_loader=lambda:'synthetic-private-credential')
    p=client.proposal('NOM','IPTS-42');r=client.acquisition('NOM','IPTS-42',10,location=LOCATION)
    assert p['status']==r['status']=='available'
    assert p['instruments']==['NOM'] and r['identity']['sample_id']=='123'
    exported=json.dumps([p,r])
    assert all(word not in exported for word in ('private person','private email','do not export','synthetic-private-credential'))
    url,request=session.calls[1]
    assert url=='https://oncat.ornl.gov/api/datafiles/SNS/NOM/IPTS-42/nexus/NOM_10.nxs.h5'
    assert ('facility','SNS') in request['params'] and ('instrument','NOM') in request['params']
    assert r['request_scope']==dict(facility='SNS',instrument='NOM',experiment='IPTS-42')
    assert request['allow_redirects'] is False and request['stream'] is True
    assert ('projection','metadata.entry.sample.chemical_formula') in request['params']


def test_public_metadata_redaction_is_not_reported_as_a_missing_run():
    session=Session(Response({'location':LOCATION}))
    result=m.CatalogClient(session=session).acquisition('NOM','IPTS-42',10,location=LOCATION)
    assert result['status']=='partial' and result['fields']=={}
    assert 'entry exists' in result['message'] and session.calls[0][1]['headers']=={}


def test_rejected_credential_is_not_retried_until_it_changes():
    token=['expired-synthetic-token'];session=Session(Response(status=401),Response(proposal()))
    client=m.CatalogClient(session=session,token_loader=lambda:token[0])
    assert client.proposal('NOM','IPTS-42')['status']=='access_unavailable'
    client.begin_batch()
    assert client.acquisition('NOM','IPTS-42',10)['status']=='access_unavailable'
    assert len(session.calls)==1
    token[0]='replacement-synthetic-token'
    assert client.proposal('NOM','IPTS-42')['status']=='available'
    assert len(session.calls)==2


@pytest.mark.parametrize('response,status',[(Response(status=302),'unavailable'),
    (Response({},headers={'Link':'external next page'}),'unavailable'),
    (Response(raw=b'x'*(m.MAX_BYTES+1)),'partial'),(Response(raw=b'not json'),'unavailable')])
def test_unbounded_redirected_or_malformed_responses_cannot_be_adopted(response,status):
    result=m.CatalogClient(session=Session(response)).proposal('NOM','IPTS-42')
    assert result['status']==status and 'fields' not in result


def test_outage_and_request_budget_leave_explicit_partial_coverage(monkeypatch):
    import requests
    session=Session(requests.ConnectionError('sensitive exception details'))
    client=m.CatalogClient(session=session)
    first=client.proposal('NOM','IPTS-42');second=client.acquisition('NOM','IPTS-42',10)
    assert first['status']==second['status']=='unavailable' and len(session.calls)==1
    assert 'sensitive' not in json.dumps([first,second])
    monkeypatch.setattr(m,'MAX_REQUESTS',1)
    client=m.CatalogClient(session=Session(Response(proposal())))
    client.proposal('NOM','IPTS-42')
    assert client.acquisition('NOM','IPTS-42',10)['status']=='partial'


@pytest.mark.parametrize('change',[{'location':'/SNS/NOM/IPTS-99/nexus/NOM_10.nxs.h5'},
    {'indexed':{'run_number':11}}, {'experiment':'IPTS-99'}, {'instrument':'PG3'}])
def test_different_run_or_proposal_is_never_merged(change):
    result=m.CatalogClient(session=Session(Response(datafile(**change)))).acquisition('NOM','IPTS-42',10,location=LOCATION)
    assert result['status']=='conflict' and 'fields' not in result


def test_fallback_uses_only_three_known_exact_paths_and_stops_when_found():
    path='/SNS/NOM/IPTS-42/data/NOM_10_event.nxs'
    session=Session(Response(status=404),Response(datafile(location=path)))
    result=m.CatalogClient(session=session).acquisition('NOM','IPTS-42',10)
    assert result['location']==path and len(session.calls)==2
    assert all('/api/datafiles/' in url for url,_ in session.calls)


def test_comparison_preserves_raw_identity_and_normalizes_timestamp_offsets():
    client=m.CatalogClient(session=Session(Response(datafile())))
    remote=client.acquisition('NOM','IPTS-42',10,location=LOCATION)
    raw=dict(identity={'sample_id':'different','identity_status':'resolved'},fields={
        'start_time':{'value':'2019-12-31T19:00:00.123456789-05:00'},'run_title':{'value':'another label'}})
    before=deepcopy(raw)
    compared=m.compare_acquisition(raw,remote)
    assert set(compared['conflicts'])=={'sample_identity','run_title'}
    assert next(r for r in compared['fields'] if r['field']=='start_time')['status']=='agrees'
    assert raw==before
    assert m.compare_acquisition({},remote)['status']=='context_only'
    raw['fields']['start_time']['value']='2020-01-01T00:00:00'
    assert next(r for r in m.compare_acquisition(raw,remote)['fields'] if r['field']=='start_time')['status']=='unresolved'


@pytest.mark.parametrize('value,expected',[(300.0,'agrees'),('300.000','agrees'),('3e2','agrees'),
    (300.01,'conflict'),('300 seconds','unresolved'),(False,'unresolved'),('NaN','unresolved')])
def test_numeric_representation_does_not_create_false_disagreements_or_assume_units(value,expected):
    raw=dict(fields={'duration':{'value':300}})
    remote=dict(fields={'duration':{'value':value}})
    assert m.compare_acquisition(raw,remote)['fields'][0]['status']==expected


def test_unsupported_beamline_is_rejected_before_any_request():
    session=Session();client=m.CatalogClient(session=session)
    with pytest.raises(ValueError,match='NOMAD and POWGEN'):client.proposal('SNAP','IPTS-42')
    with pytest.raises(ValueError,match='NOMAD and POWGEN'):client.acquisition('SNAP','IPTS-42',10)
    assert session.calls==[]


def test_scope_conflict_explains_why_no_catalog_fields_were_adopted():
    remote=m.CatalogClient(session=Session(Response(datafile(instrument='PG3')))).acquisition('NOM','IPTS-42',10)
    comparison=m.compare_acquisition({},remote)
    assert comparison['status']=='conflict'
    assert 'conflicting or unsupported acquisition identity' in comparison['message']


def test_capture_reproduces_conflicts_and_binds_context_to_the_exact_preparation(tmp_path):
    from gpx_recipe.battery.catalog_evidence import capture_catalog,read_catalog,validate_payload
    client=m.CatalogClient(session=Session(Response(proposal()),Response(datafile())))
    p=client.proposal('NOM','IPTS-42');r=client.acquisition('NOM','IPTS-42',10,location=LOCATION)
    record=dict(fields={'run_title':{'value':'different file title'}},identity={'sample_id':'123','identity_status':'resolved'},catalog=r)
    record['catalog_comparison']=m.compare_acquisition(record,r,p)
    acquisition=dict(instrument='NOM',ipts='IPTS-42',run=10,sample_id='123',metadata_sha256='a'*64)
    reference=capture_catalog(p,record,acquisition,tmp_path)
    payload=read_catalog(reference,tmp_path,acquisition)
    assert payload['role']=='review_only' and payload['comparison']['conflicts']==['run_title']
    with pytest.raises(ValueError,match='another reviewed acquisition'):
        read_catalog(reference,tmp_path,dict(acquisition,run=11))
    altered=deepcopy(payload);altered['comparison']['conflicts']=[]
    with pytest.raises(ValueError,match='reproduced'):validate_payload(altered,acquisition)
    altered=deepcopy(payload);altered['proposal']['fields']['abstract']='changed context'
    with pytest.raises(ValueError,match='digest changed'):validate_payload(altered,acquisition)
    (tmp_path/reference['path']).write_text('{}')
    with pytest.raises(ValueError,match='source changed'):read_catalog(reference,tmp_path,acquisition)


def test_saved_receipt_cannot_hide_the_captured_catalog_disagreement(tmp_path,monkeypatch):
    from hashlib import sha256
    from gpx_recipe.battery.catalog_evidence import capture_catalog,read_catalog,catalog_summary
    from radar_pd_nova import battery_preparation as preparation
    client=m.CatalogClient(session=Session(Response(proposal()),Response(datafile())))
    p=client.proposal('NOM','IPTS-42');r=client.acquisition('NOM','IPTS-42',10,location=LOCATION)
    record=dict(fields={'run_title':{'value':'Different archive title'}},identity={'sample_id':'123','identity_status':'resolved'},catalog=r)
    record['catalog_comparison']=m.compare_acquisition(record,r,p)
    acquisition=dict(instrument='NOM',ipts='IPTS-42',run=10,sample_id='123',metadata_sha256='a'*64)
    inputs=tmp_path/'inputs';inputs.mkdir();model=tmp_path/'model';model.mkdir()
    reference=capture_catalog(p,record,acquisition,inputs)
    context=read_catalog(reference,inputs,acquisition)
    result=dict(schema='radar-battery-preparation-result/v1',status='prepared_needs_review',
        specification=dict(acquisition=acquisition,catalog_context=reference),
        project_sha256=sha256(b'unchanged project').hexdigest(),catalog_context=catalog_summary(context,reference))
    (model/'prepared.gpx').write_bytes(b'unchanged project')
    path=model/'preparation.json';path.write_text(json.dumps(result))
    # Native preparation is covered separately; exercise reopening's independent summary check.
    monkeypatch.setattr(preparation,'validate_preparation',lambda *args:None)
    assert preparation.verify_prepared(tmp_path)==result
    result['catalog_context']['comparison_status']='compared';result['catalog_context']['conflicts']=[]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='summary differs'):preparation.verify_prepared(tmp_path)
    result['specification'].pop('catalog_context');path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='no captured evidence'):preparation.verify_prepared(tmp_path)
