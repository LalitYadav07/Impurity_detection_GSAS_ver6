import hashlib
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.battery_transport import prepare_on_galaxy


def test_preparation_dispatch_publishes_exact_native_model_and_receipt(tmp_path):
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True)
    (job/'inputs/preparation.json').write_text('{}')
    calls=[]
    def call(action,source,directory,**kwargs):
        calls.append((action,source,kwargs))
        (directory/'model').mkdir(parents=True,exist_ok=True)
        (directory/'model/prepared.gpx').write_bytes(b'model')
        digest=hashlib.sha256(b'model').hexdigest()
        result={'project_sha256':digest}
        (directory/'model/preparation.json').write_text(json.dumps(result))
        return dict(status='model_prepared',model_dataset_id='model-dataset',preparation=result,
                    output_sha256=digest,galaxy_job_id='job-id',
                    implementation_sha256=kwargs['plan']['implementation_sha256'])
    runtime=SimpleNamespace(call=call,service=SimpleNamespace(upload_document=lambda *args,**kwargs:'input-dataset'))
    result=prepare_on_galaxy(runtime,job)
    assert calls[0][0]=='prepare' and calls[0][1]['dataset_id']=='input-dataset'
    assert len(calls[0][2]['dispatch_key'])==24
    assert (job/'model/prepared.gpx').read_bytes()==b'model'
    assert json.loads((job/'preparation-publication.json').read_text())['dataset_id']=='model-dataset'
    assert result['project_sha256']==hashlib.sha256(b'model').hexdigest()
    assert calls[0][2]['recover'] is False
    # The same completed dispatch recovers its original model, even when the
    # input file's timestamp changes after a session restore.
    import os
    os.utime(job/'inputs/preparation.json',(1700000000,1700000000))
    again=prepare_on_galaxy(runtime,job)
    assert again==result and calls[1][2]['recover'] is True
    assert calls[0][2]['dispatch_key']==calls[1][2]['dispatch_key']


def test_packet_identity_ignores_location_mode_and_mtime(tmp_path):
    import os
    from radar_pd_nova.battery_transport import write_preparation_packet
    for name,stamp,mode in [('first',1700000000,0o600),('second',1800000000,0o644)]:
        folder=tmp_path/name;folder.mkdir()
        for leaf,data in [('preparation.json',b'{}'),('bank.gsa',b'exact records\n')]:
            path=folder/leaf;path.write_bytes(data);os.chmod(path,mode);os.utime(path,(stamp,stamp))
        write_preparation_packet(folder,tmp_path/(name+'.zip'))
    assert (tmp_path/'first.zip').read_bytes()==(tmp_path/'second.zip').read_bytes()


def test_new_implementation_does_not_reinterpret_old_dispatch(tmp_path,monkeypatch):
    import gpx_recipe.battery.transport as native
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True)
    (job/'inputs/preparation.json').write_text('{}')
    from radar_pd_nova.battery_transport import write_preparation_packet
    write_preparation_packet(job/'inputs',job/'preparation-inputs.zip')
    digest=hashlib.sha256((job/'preparation-inputs.zip').read_bytes()).hexdigest()
    (job/'preparation-dispatch.json').write_text(json.dumps(dict(
        source=dict(source_sha256=digest),implementation_sha256='old')))
    monkeypatch.setattr(native,'implementation_signature',lambda:'new')
    with pytest.raises(ValueError,match='implementation changed'):
        prepare_on_galaxy(SimpleNamespace(),job)
