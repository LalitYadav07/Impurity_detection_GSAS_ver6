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
        (directory/'model').mkdir(parents=True)
        (directory/'model/prepared.gpx').write_bytes(b'model')
        digest=hashlib.sha256(b'model').hexdigest()
        result={'project_sha256':digest}
        (directory/'model/preparation.json').write_text(json.dumps(result))
        return dict(status='model_prepared',model_dataset_id='model-dataset',preparation=result,
                    output_sha256=digest,galaxy_job_id='job-id')
    runtime=SimpleNamespace(call=call,service=SimpleNamespace(upload_document=lambda *args,**kwargs:'input-dataset'))
    result=prepare_on_galaxy(runtime,job)
    assert calls[0][0]=='prepare' and calls[0][1]['dataset_id']=='input-dataset'
    assert len(calls[0][2]['dispatch_key'])==24
    assert (job/'model/prepared.gpx').read_bytes()==b'model'
    assert json.loads((job/'preparation-publication.json').read_text())['dataset_id']=='model-dataset'
    assert result['project_sha256']==hashlib.sha256(b'model').hexdigest()
