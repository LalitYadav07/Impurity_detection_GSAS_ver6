import copy
import hashlib
import json
import zlib
import base64
import pytest
from gpx_recipe.preparation import structural_history as history

def large():
    conditions=[dict(space_group_index=c,family_index=f,queries=531,
        pinball_loss=.123456789123,target_interval_containment=.812345678912)
        for c in range(16) for f in range(20)]
    return [dict(epoch=i+1,validation=dict(mode_validation=dict(conditions=conditions,loss=.123456789123))) for i in range(60)]

def test_full_sixty_epoch_mode_history_preserves_every_value_within_metadata_limit():
    original=large();assert len(json.dumps(original,indent=2).encode())>2*1024**2
    packed=history.pack(original)
    assert packed['schema']==history.SCHEMA
    assert len(json.dumps(packed).encode())<256*1024
    assert history.unpack(packed)==original

@pytest.mark.parametrize('change',['hash','size','trailing','truncated','encoding','overexpanded'])
def test_damaged_or_unbounded_history_cannot_bypass_existing_validation(change,monkeypatch):
    packed=history.pack(large())
    if change=='hash':packed['sha256']='a'*64
    elif change=='size':packed['bytes']+=1
    elif change=='trailing':packed['payload']=base64.b64encode(base64.b64decode(packed['payload'])+b'other').decode()
    elif change=='truncated':packed['payload']=packed['payload'][:-4]
    elif change=='encoding':packed['payload']='not base64!'
    else:monkeypatch.setattr(history,'MAX_RAW_BYTES',1000)
    with pytest.raises(ValueError):history.unpack(packed)

def test_legacy_inline_history_is_unchanged_and_nonfinite_values_cannot_be_archived():
    original=[dict(epoch=1,loss=.5)]
    assert history.pack(original) is original and history.unpack(original) is original
    with pytest.raises(ValueError):history.pack([dict(loss=float('nan'))])
