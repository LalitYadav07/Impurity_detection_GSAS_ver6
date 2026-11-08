"""Bounded diagnostic receipts, never scientific results or native authority."""
import json
from pathlib import Path

from ..symmetry.compiler import package_signature
from ..symmetry.contracts import digest
from .tof_kernel_arrays import read_bounded

SCHEMA='radar-preparation-worker-failure/v1'
MAX_BYTES=8192
KINDS={'reference','magnetic','structural','kernel','batch','structural-examples','structural-training'}
REASONS={
    'parent_cell_domain_conflict':
        'The sampled parent-reference cells exceed this child model\u2019s reviewed cell bounds. '
        'Adjust the declared parent-cell range or child-cell bounds before retrying. '
        'The original shared parent cells and physical bounds were preserved.',
    'native_validation_failed':
        'Native preparation rejected the declared inputs or scientific scope. '
        'Inspect the native console for the specific check; no model or physical bound was removed.',
    'native_execution_failed':
        'Native preparation stopped before producing a complete result. '
        'Inspect the native console before retrying; no scientific result was accepted.'}


class PreparationScopeError(ValueError):
    def __init__(self,code):
        if code not in REASONS:raise ValueError('An installed preparation failure code is required')
        self.code=code
        super().__init__(REASONS[code])


def write(path,kind,request,error):
    """Fresh worker output only, without raw exception text or user strings."""
    if kind not in KINDS:raise ValueError('A complete preparation operation is required')
    code=error.code if isinstance(error,PreparationScopeError) else (
        'native_validation_failed' if isinstance(error,ValueError) else 'native_execution_failed')
    value=dict(schema=SCHEMA,kind=kind,request_sha256=digest(request),
        implementation_sha256=package_signature(),code=code,reason=REASONS[code])
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>MAX_BYTES:raise ValueError('Preparation failure receipt exceeds its bound')
    try:
        with Path(path).open('xb') as stream:stream.write(raw)
    except FileExistsError:return False
    return True


def read(path,kind,request):
    """Only the exact failed owned request can explain its queue disposition."""
    try:
        value=json.loads(read_bounded(path,MAX_BYTES))
        if (not isinstance(value,dict) or set(value)!=
                {'schema','kind','request_sha256','implementation_sha256','code','reason'}
                or value['schema']!=SCHEMA or value['kind']!=kind or kind not in KINDS
                or value['request_sha256']!=digest(request)
                or value['implementation_sha256']!=package_signature()
                or not isinstance(value['code'],str) or value['code'] not in REASONS
                or value['reason']!=REASONS[value['code']]):return None
        return value['reason']
    except (OSError,ValueError,TypeError):return None
