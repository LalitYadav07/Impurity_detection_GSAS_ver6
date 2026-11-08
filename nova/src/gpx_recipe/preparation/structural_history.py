"""Lossless bounded JSON history storage; decoded values keep existing audits."""
import base64
import hashlib
import json
import zlib

SCHEMA='radar-lossless-training-history/v1'
INLINE_BYTES=256*1024
MAX_RAW_BYTES=16*1024**2
MAX_PACKED_BYTES=2*1024**2

def pack(history):
    if not isinstance(history,list) or len(history)>1000:raise ValueError('Bounded training history required')
    raw=json.dumps(history,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_RAW_BYTES:raise ValueError('Complete training history exceeds its decoded bound')
    if len(raw)<=INLINE_BYTES:return history
    payload=base64.b64encode(zlib.compress(raw,level=1)).decode('ascii')
    if len(payload)>MAX_PACKED_BYTES:raise ValueError('Complete training history exceeds its stored bound')
    return dict(schema=SCHEMA,codec='zlib-json-base64/v1',bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),payload=payload)

def unpack(value):
    if isinstance(value,list):return value
    if (not isinstance(value,dict) or set(value)!={'schema','codec','bytes','sha256','payload'}
            or value['schema']!=SCHEMA or value['codec']!='zlib-json-base64/v1'
            or type(value['bytes'])is not int or not 1<=value['bytes']<=MAX_RAW_BYTES
            or not isinstance(value['sha256'],str) or len(value['sha256'])!=64
            or not isinstance(value['payload'],str) or len(value['payload'])>MAX_PACKED_BYTES):
        raise ValueError('Invalid bounded training history archive')
    try:
        packed=base64.b64decode(value['payload'],validate=True)
        decoder=zlib.decompressobj();raw=decoder.decompress(packed,MAX_RAW_BYTES+1)
        if (len(raw)!=value['bytes'] or len(raw)>MAX_RAW_BYTES or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail
                or hashlib.sha256(raw).hexdigest()!=value['sha256']):
            raise ValueError('Complete training history bytes changed or exceeded their bound')
        def invalid(_):raise ValueError('Nonfinite training history value')
        result=json.loads(raw,parse_constant=invalid)
    except (zlib.error,UnicodeError,TypeError) as error:
        raise ValueError('Invalid lossless training history data') from error
    if not isinstance(result,list) or len(result)>1000:raise ValueError('Bounded training history required')
    return result

def restore(metadata):
    if isinstance(metadata,dict) and 'history' in metadata:
        metadata=dict(metadata,history=unpack(metadata['history']))
    return metadata
