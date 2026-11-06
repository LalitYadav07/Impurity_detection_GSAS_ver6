"""Bind structural preparation to current native geometry and owned outputs."""
from pathlib import Path
import time

import numpy as np

from ..structural_live.optimization.offline_native_start_bank import (
    OfflineNativeStartBank,build,native_in_this_process,
)
from ..symmetry.compiler import file_sha
from .hints import _copy
from .structural_hints import MAX_BYTES,MAX_METADATA


def prepare_starting_bank(source,destination,choices=()):
    """Preserve the full joint-cell portfolio and actual observation contract.

    A checksum is not native attestation. Imported banks with an exact current
    context are fully recomputed once unless this process itself constructed
    those exact bytes. The owned result contains freshly native curves and
    freshly selected proposal steps. Qualification cost is reported, never
    disguised as a fast cache hit. Later snapshots reuse the owned bank.
    """
    if not isinstance(choices,(list,tuple)) or len(choices)>5000:
        raise ValueError('A bounded structural preparation selection is required')
    destination=Path(destination);began=time.monotonic();discarded=[];selected=None;provided=None
    for choice in choices:
        try:
            files=choice['files'];path=Path(files['path']);receipt_path=Path(files['receipt_path'])
            if (path.is_symlink() or receipt_path.is_symlink() or path.stat().st_size>MAX_BYTES
                    or receipt_path.stat().st_size>MAX_METADATA
                    or file_sha(path)!=choice['sha256'] or file_sha(receipt_path)!=choice['receipt_sha256']):
                raise ValueError('Optional structural bank changed')
            bank=OfflineNativeStartBank(source,path,receipt_path=receipt_path)
            if bank.receipt['policy']!='joint_cell_v1':raise ValueError('Prepared start policy differs from the complete current portfolio')
            selected=choice;provided=bank;break
        except (OSError,ValueError,EOFError,KeyError,TypeError,IndexError) as exc:
            discarded.append(dict(entry=choice.get('entry') if isinstance(choice,dict) else None,
                reason=str(exc)[:300],status='rebuild_required'))
    if selected is not None and native_in_this_process(source,selected['files']['path'],selected['files']['receipt_path']):
        _copy(Path(selected['files']['path']),destination,selected['sha256'])
        _copy(Path(selected['files']['receipt_path']),destination.with_suffix('.json'),selected['receipt_sha256'])
        receipt=provided.receipt
        qualification=dict(status='reused_native_bank',curve_bytes_reused=True,
            native_curves_recomputed=0,native_origin='Same owned process, exact bank/receipt bytes and current physical context')
    else:
        receipt=build(source,destination,policy='joint_cell_v1',reuse_fixed_cell_profiles=True)
        qualification=dict(status='native_requalified' if selected is not None else 'rebuilt_current_context' if choices else 'not_selected',
            curve_bytes_reused=False,native_curves_recomputed=receipt['native_curves'],
            native_origin='Fresh native curves and proposal-step selection; uploaded values are not used as scientific predictions')
        if selected is not None:
            checked=OfflineNativeStartBank(source,destination)
            if np.array_equal(checked.points,provided.points):
                errors=np.linalg.norm(checked.curves-provided.curves,axis=1)/np.maximum(np.linalg.norm(checked.curves,axis=1),1e-30)
                qualification['supplied_maximum_relative_curve_error']=float(np.max(errors,initial=0.))
                qualification['supplied_curves_match_current']=bool(np.isfinite(errors).all() and np.max(errors,initial=0.)<1e-11)
            else:qualification['supplied_curves_match_current']=False
    qualification.update(selected_entry=selected['entry'] if selected is not None else None,
        selected_bank=selected['bank'] if selected is not None else None,
        attached_entries=len(choices),discarded_entries=discarded,
        seconds=time.monotonic()-began,observations_reused=False,scientific_candidates_removed=False,
        qualification_persisted_as_trust=False)
    return receipt,qualification
