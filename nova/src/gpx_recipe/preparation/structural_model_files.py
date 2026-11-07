"""Small metadata inspection shared by model dispatch and numerical readers."""
import json
from pathlib import Path

from .tof_kernel_arrays import read_bounded

MAX_BYTES=32*1024**2
MAX_METADATA_BYTES=2*1024**2


def document(directory,metadata_name='model.json',weights_name='weights.npz'):
    if (any(not isinstance(name,str) or not name or Path(name).name!=name
            for name in (metadata_name,weights_name))
            or not metadata_name.endswith('.json') or not weights_name.endswith('.npz')):
        raise ValueError('Structural model assets must be data files within one captured directory')
    value=json.loads(read_bounded(Path(directory)/metadata_name,MAX_METADATA_BYTES))
    if not isinstance(value,dict):raise ValueError('A structural model metadata object is required')
    return value
