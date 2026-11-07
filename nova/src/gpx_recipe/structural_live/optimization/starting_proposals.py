"""Current-context starting proposals with process-owned numeric provenance.

Portable native banks keep their existing admission path. These interpolated
initial curves are never native certificates, likelihoods or fitted results.
Fresh workers rebuild them from their current native geometry/profile.
"""
from collections import OrderedDict
import os
from pathlib import Path
import threading
import numpy as np

from . import offline_native_start_bank as banks
from .native_bank_reuse import NativeBankSource
from .native_cell_proposals import ProposalUnavailable
from .native_tof_scalars import ScalarBatchUnavailable
from ...execution import write_json
from ...live.native_peak_batch import BatchUnavailable
from ...symmetry.contracts import digest

_origins=OrderedDict()
_pid=os.getpid()
_lock=threading.Lock()


def _reset_fork():
    global _pid,_lock
    if _pid!=os.getpid():
        _origins.clear();_lock=threading.Lock();_pid=os.getpid()


def _key(source,path,receipt_path=None):
    path=Path(path)
    receipt=path.with_suffix('.json') if receipt_path is None else Path(receipt_path)
    return digest(dict(context=banks.context(source),arrays=banks.sha(path),receipt=banks.sha(receipt)))


def _remember(source,path):
    _reset_fork()
    receipt=banks.read_bank_receipt(path)
    if (receipt.get('curve_role')!='starting_proposals' or receipt.get('interpolation') is not True
            or receipt.get('native_curves')!=0):
        raise ValueError('Only explicitly generated starting proposals may be registered')
    key=_key(source,path)
    with _lock:
        _origins[key]=True;_origins.move_to_end(key)
        while len(_origins)>64:_origins.popitem(last=False)


def _owned(source,path,receipt_path=None):
    _reset_fork();key=_key(source,path,receipt_path)
    with _lock:
        if key not in _origins:return False
        _origins.move_to_end(key);return True


class StartingProposalBank(banks.OfflineNativeStartBank):
    CURVE_ROLE='starting_proposals'

    def __init__(self,source,path,*,receipt_path=None):
        super().__init__(source,path,receipt_path=receipt_path)
        if not _owned(source,path,receipt_path):
            raise ValueError('Starting proposals require this process owned construction')


class _Hybrid:
    def __init__(self,source,exact,proposals):
        self.source=source;self.exact=exact;self.proposals=proposals
        self.checked=False;self.disabled=False;self.anchor_checks=0
        self.fallbacks=[];self.proposal_calls=0

    def evaluate(self,point):
        source=self.source;point=np.asarray(point,float)
        if np.all(point[source.mode_dimensions:]==0) or self.disabled:
            return self.exact.native(point)
        try:
            curve=self.proposals.evaluate(point);self.proposal_calls+=1
        except (ProposalUnavailable,ScalarBatchUnavailable,BatchUnavailable) as error:
            self.disabled=True;self.fallbacks.append(str(error))
            return self.exact.native(point)
        if not self.checked:
            native=self.exact.native(point);self.anchor_checks+=1
            n=source.fast.native;engine=self.exact.forward
            _,_,_,_,_,_,axis,hist,_,_=engine._context()
            hfx=engine.collector.hfx
            background,_=engine.background_function(hfx,n.parms,n.controls[hfx+'bakType'],
                n.controls[hfx+'histType'],axis,hist['Background'][1].get('fixback'))
            delta=(curve-native)[source.fast.slice]
            signal=native[source.fast.slice]-background
            relative=float(np.linalg.norm(delta)/max(np.linalg.norm(signal),1e-30))
            weights=np.asarray(hist['Data'][2][source.fast.slice],float)
            valid=np.isfinite(weights)&(weights>0)
            sigma=float(np.max(abs(delta[valid])*np.sqrt(weights[valid]))) if np.any(valid) else 0.
            self.checked=True
            if not np.isfinite([relative,sigma]).all() or relative>1e-8 or sigma>.01:
                self.disabled=True;self.fallbacks.append('Current native proposal anchor disagrees')
            # The required native anchor is already available. Keep its exact
            # curve without promoting the other proposed curves to native.
            return native
        return curve

    def diagnostics(self):
        return dict(method='Exact fixed-cell reuse with current continuous TOF starting proposals',
            native_anchor_checks=self.anchor_checks,proposal_calls=self.proposal_calls,
            disabled=self.disabled,fallbacks=self.fallbacks,
            fixed_cell_cached_calls=self.exact.cached_calls,
            original_forward=self.exact.forward.diagnostics() if self.exact.forward else None,
            continuous_profile=self.proposals.diagnostics(),
            native_origin_authority=False,original_native_objective_required=True)


def build_if_useful(source,path):
    """Automatic execution policy, not a user scientific-accuracy option."""
    if getattr(getattr(source,'fast',None),'native',None) is None:
        return None,dict(status='original_native',reason='No supported native context')
    exact=NativeBankSource(source)
    if exact.forward is None:
        return None,dict(status='original_native',reason='Current context needs the original native renderer')
    from .tof_profile_proposals import for_source
    try:proposals=for_source(source,exact=exact.forward)
    except (ProposalUnavailable,ScalarBatchUnavailable,BatchUnavailable) as error:
        return None,dict(status='original_native',reason=str(error))
    if proposals is None:
        return None,dict(status='original_native',reason='Current profile is below the measured proposal-compilation work threshold')
    hybrid=_Hybrid(source,exact,proposals)
    receipt=banks._construct(source,path,policy='joint_cell_v1',proposal_renderer=hybrid.evaluate)
    receipt['construction_execution']=hybrid.diagnostics()
    write_json(Path(path).with_suffix('.json'),receipt)
    _remember(source,path)
    return receipt,dict(status='generated_current_proposals',curve_bytes_reused=False,
        native_curves_recomputed=0,starting_proposals=receipt['starting_curves'],
        native_origin=None,native_anchor_checks=hybrid.anchor_checks,
        original_native_objective_required=True,execution=hybrid.diagnostics())


def load_current_bank(source,path,output):
    """Only owned working-context proposals may enter this shared fitter."""
    receipt=banks.read_bank_receipt(path)
    role=receipt.get('curve_role','native')
    if role=='native':
        return banks.OfflineNativeStartBank(source,path),dict(curve_role='native',regenerated=False)
    if role!='starting_proposals':raise ValueError('Unknown starting-curve role')
    if _owned(source,path):
        return StartingProposalBank(source,path),dict(curve_role=role,regenerated=False,
            origin='Current owned process; no imported proof flag')
    # Stored proposal flags cannot authorize another worker. Rebuild into the
    # fresh output rather than modifying an immutable saved preparation.
    destination=Path(output)/'current-starting-curves.npz'
    rebuilt,execution=build_if_useful(source,destination)
    if rebuilt is None:
        rebuilt=banks.build(source,destination,policy='joint_cell_v1',reuse_fixed_cell_profiles=True)
        bank=banks.OfflineNativeStartBank(source,destination)
    else:bank=StartingProposalBank(source,destination)
    return bank,dict(curve_role=bank.CURVE_ROLE,regenerated=True,
        reason='Starting proposal construction is not owned by this process',execution=execution)
