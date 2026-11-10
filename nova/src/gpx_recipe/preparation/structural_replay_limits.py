"""Closed dimensions for seekable native-pair training rows.

These limits concern an owned data representation. They do not enlarge dense
NPY, portable pattern-archive or legacy continuation limits, or qualify a model.
"""
SCHEMA = 'radar-seekable-native-pair-rows/v1'
MAX_ROWS = 1048576
INDEX_BYTES_PER_ROW = 76  # three int32 selectors and two SHA256 byte planes
MAX_INDEX_BYTES = 96*1024**2
MAX_LABEL_BYTES = 64*1024**2
MAX_NATIVE_BYTES = 8*1024**3


def validate(value, *, rows=None, bins=None):
    keys = {'schema', 'rows', 'bins', 'channels', 'native_bytes', 'index_bytes'}
    if (not isinstance(value, dict) or set(value) != keys or value['schema'] != SCHEMA
            or any(type(value[k]) is not int for k in keys-{'schema'})
            or value['channels'] != 3 or not 1 <= value['rows'] <= MAX_ROWS
            or not 64 <= value['bins'] <= 8192 or not 1 <= value['native_bytes'] <= MAX_NATIVE_BYTES
            or value['index_bytes'] != value['rows']*INDEX_BYTES_PER_ROW
            or value['index_bytes'] > MAX_INDEX_BYTES
            or (rows is not None and value['rows'] != rows)
            or (bins is not None and value['bins'] != bins)):
        raise ValueError('Invalid bounded seekable native-pair row layout')
    return value
