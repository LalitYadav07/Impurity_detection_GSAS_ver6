"""Working-only output reservations measured by an owned native task.

This file is produced after native dimensions and all selected trajectory
rows are known. It is never accepted as an uploaded library hint. The batch
still verifies every final artifact and keeps its original hard byte limits.
"""
import hashlib
import json
from pathlib import Path

from ..execution import write_json
from ..symmetry.compiler import package_signature
from ..symmetry.contracts import digest
from .structural_paired_blocks import MAX_BYTES
from .structural_pair_arrays import MAX_ROWS, MAX_BINS
from .structural_training_blocks import REPRESENTATION
from .tof_kernel_arrays import read_bounded

SCHEMA = 'radar-preparation-native-size/v1'
NAME = 'native-size.json'
MAX_RECORD_BYTES = 4096
EXAMPLE_RESERVE = 184*1024**2
# Two bounded manifests, NPZ/zlib/header overhead and the existing task log.
# The original artifact cap remains authoritative if this estimate is larger.
OVERHEAD = 24*1024**2


def paired_array_bytes(rows, bins, modes, cells):
    values = (rows, bins, modes, cells)
    if (any(type(v) is not int for v in values) or not 3 <= rows <= MAX_ROWS
            or not 3 <= bins <= MAX_BINS or modes < 0 or not 1 <= cells <= 6):
        raise ValueError('Complete native pair dimensions required')
    # Three curve planes; Q/background; parent metrics; native coordinates;
    # trajectory, role, two geometry labels, ambiguity and two physical norms.
    size = 8*(3*rows*bins+2*bins+6*rows+rows*(modes+cells))+34*rows
    if size > MAX_BYTES:
        raise ValueError('Native pair arrays exceed their existing numerical bound')
    return size


def _context(request):
    from .structural_jobs import validate_examples
    request = validate_examples(request)
    if request['settings'].get('representation') != REPRESENTATION:
        raise ValueError('Native sizing currently requires paired structural examples')
    return request


def _dimensions(request, rows, bins, mode_dimensions, cell_dimensions):
    settings = request['settings']; first, last = settings['trajectory_slice']
    if (rows != 1+(last-first)*(settings['steps']+1)
            or type(bins) is not int or bins > request['grid']['bins']+1):
        raise ValueError('Native dimensions differ from the complete declared trajectory/grid scope')
    size = paired_array_bytes(rows, bins, mode_dimensions, cell_dimensions)
    return size, min(EXAMPLE_RESERVE, size+OVERHEAD)


def publish(request, *, rows, bins, mode_dimensions, cell_dimensions, numeric_bytes):
    request = _context(request)
    size, reserve = _dimensions(request, rows, bins, mode_dimensions, cell_dimensions)
    if type(numeric_bytes) is not int or numeric_bytes != size:
        raise ValueError('The native array inventory differs from its dimensional reservation')
    path = Path(request['destination'])/NAME
    if path.exists() or path.is_symlink():
        raise ValueError('Native size records belong to one fresh task attempt')
    write_json(path, dict(schema=SCHEMA, request_sha256=digest(request),
        implementation_sha256=package_signature(), rows=rows, bins=bins,
        mode_dimensions=mode_dimensions, cell_dimensions=cell_dimensions,
        numeric_bytes=size, reservation_bytes=reserve))


def read(request, implementation_sha256):
    """Read only the task's fresh output namespace, never captured input files."""
    request = _context(request)
    path = Path(request['destination'])/NAME
    raw = read_bounded(path, MAX_RECORD_BYTES)
    value = json.loads(raw)
    fields = {'schema', 'request_sha256', 'implementation_sha256', 'rows', 'bins',
              'mode_dimensions', 'cell_dimensions', 'numeric_bytes', 'reservation_bytes'}
    if (not isinstance(value, dict) or set(value) != fields or value['schema'] != SCHEMA
            or value['request_sha256'] != digest(request)
            or value['implementation_sha256'] != implementation_sha256):
        raise ValueError('Native sizing record differs from its owned task or worker implementation')
    size, reserve = _dimensions(request, value['rows'], value['bins'],
                                value['mode_dimensions'], value['cell_dimensions'])
    if (type(value['numeric_bytes']) is not int or value['numeric_bytes'] != size
            or type(value['reservation_bytes']) is not int or value['reservation_bytes'] != reserve):
        raise ValueError('Native sizing byte counts differ from their dimensions')
    return dict(value, record_sha256=hashlib.sha256(raw).hexdigest())
