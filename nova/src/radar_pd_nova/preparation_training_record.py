"""Read small recorded training metadata on demand, without loading weights."""
import hashlib
import json
import math
from pathlib import Path

from gpx_recipe.preparation.library import confined
from gpx_recipe.preparation.structural_history import restore
from gpx_recipe.preparation.tof_kernel_arrays import read_bounded
from .preparation_labels import _entry_detail

MAX_METADATA=2*1024**2
DEVICES={'cpu':'CPU','mps':'Apple GPU (MPS)','cuda':'CUDA GPU'}


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def _duration(seconds):
    rounded=round(seconds)
    return f'{rounded} s' if rounded<60 else f'{rounded//60} min {rounded%60} s'


def describe(library,entry):
    if _entry_detail(library,entry,{'structural-training','structural-ranking-model'}) is None:return None
    files=[a for a in entry.get('files',[]) if Path(a['path']).name=='model.json']
    if not files:
        return dict(summary='Training details not recorded',rows=[],status='unrecorded',
            note='This older library does not retain a training metadata file. No device or completed epoch count is inferred.')
    if len(files)!=1:raise ValueError('The saved model has ambiguous training metadata')
    asset=files[0];path=confined(library.root,asset['path'])
    if not 0<path.stat().st_size<=MAX_METADATA:raise ValueError('Training record exceeds its metadata bound')
    library.verify_assets([asset['path']])
    raw=read_bounded(path,MAX_METADATA)
    if hashlib.sha256(raw).hexdigest()!=asset['sha256']:raise ValueError('Training metadata changed during inspection')
    def invalid(_):raise ValueError('Training metadata contains a nonfinite value')
    model=restore(json.loads(raw,parse_constant=invalid))
    if not isinstance(model,dict):raise ValueError('Training metadata must be an object')
    schema=model.get('schema')
    if not isinstance(schema,str) or schema not in {'radar-three-curve-structural-ranking-model/v1',
            'radar-structural-ranking-model/v3','radar-structural-ranking-model/v4'}:
        raise ValueError('This model format has no recognized training record')
    epochs=model.get('epochs');history=model.get('history');selected=model.get('selected_epoch')
    if epochs is not None and (type(epochs)is not int or not 1<=epochs<=1000):
        raise ValueError('Recorded training epochs are invalid')
    requested=entry['model'].get('settings',{}).get('epochs')
    if entry['model']['kind']=='structural-training' and requested is not None and epochs!=requested:
        raise ValueError('Recorded training differs from the requested epoch count')
    completed=None
    if history is not None:
        if (epochs is None or not isinstance(history,list) or not 0<=len(history)<=epochs
                or any(not isinstance(row,dict) or type(row.get('epoch'))is not int
                    or row['epoch']!=index for index,row in enumerate(history,1))):
            raise ValueError('Recorded completed epochs do not form the original training history')
        completed=len(history)
    if selected is not None and (type(selected)is not int or epochs is None
            or not 1<=selected<=(epochs if completed is None else completed)):
        raise ValueError('Selected checkpoint is outside the recorded training history')
    device='Not recorded';portable='Not recorded'
    transport=model.get('training_device_transport')
    if transport is not None:
        if (not isinstance(transport,dict) or not isinstance(transport.get('device'),str)
                or transport['device'] not in DEVICES
                or transport.get('portable_inference_device')!='cpu'
                or type(transport.get('representatives'))is not int or transport['representatives']<1
                or not _finite(transport.get('maximum_logit_absolute_error'))):
            raise ValueError('Recorded training device or CPU transport is invalid')
        device=DEVICES[transport['device']];portable='CPU (reported numerical transport check)'
    seconds=model.get('seconds')
    if seconds is not None and not _finite(seconds):raise ValueError('Recorded trainer time is invalid')
    count='Not recorded' if completed is None else f'{completed} of {epochs}'
    status='unrecorded' if completed is None else 'complete' if completed==epochs else 'partial'
    rows=[dict(label='Recorded training device',value=device),dict(label='Completed epochs',value=count),
        dict(label='Selected checkpoint',value='Not recorded' if selected is None else f'Epoch {selected}'),
        dict(label='Trainer time',value='Not recorded' if seconds is None else _duration(seconds)),
        dict(label='Portable inference',value=portable)]
    note='Values reported by the saved model. Trainer time excludes Galaxy queue, image startup and data transfer.'
    supervision=model.get('mode_family_supervision')
    if isinstance(supervision,dict) and 'unresolved_native_strain_rows' in supervision:
        unresolved=supervision['unresolved_native_strain_rows'];total=supervision.get('native_points')
        if (type(total)is not int or total<1 or type(unresolved)is not int or not 0<=unresolved<=total
                or supervision.get('strain_labels_from_out_of_basis_projection') is not False):
            raise ValueError('Recorded unresolved strain targets are invalid')
        rows.append(dict(label='Native strain labels',value=f'{unresolved} of {total} native curves unresolved'))
        if unresolved:
            note+=' Unresolved strain projections supplied no strain targets. Their native curves and valid displacement targets were retained; a missing strain label is not zero strain or evidence of absence.'
    if status=='partial':note+=' The retained history does not establish completed training.'
    return dict(summary=f'{device} · {count} completed epochs',rows=rows,status=status,note=note)
