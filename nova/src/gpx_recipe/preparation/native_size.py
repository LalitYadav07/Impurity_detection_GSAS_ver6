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
MODE_SCHEMA = 'radar-preparation-native-size/v2'
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


def _dimensions(request, rows, bins, mode_dimensions, cell_dimensions,mode_coverage_rows=None):
    settings = request['settings']; first, last = settings['trajectory_slice']
    size = paired_array_bytes(rows, bins, mode_dimensions, cell_dimensions)
    expected=1+(last-first)*(settings['steps']+1)
    if 'mode_coverage' in settings:
        from .structural_mode_coverage import SCHEMA as COVERAGE,CONTEXTS_PER_MODE
        c=mode_coverage_rows
        fields={'schema','native_modes','gauge_modes','contexts','rays','selected_rays'}
        if (not isinstance(c,dict) or set(c)!=fields or c['schema']!=COVERAGE
                or any(type(c[k])is not int for k in fields-{'schema'})
                or not 0<=c['gauge_modes']<=c['native_modes']<=2048
                or c['native_modes']-c['gauge_modes']<mode_dimensions
                or c['contexts']!=min(CONTEXTS_PER_MODE,settings['trajectories'])
                or c['rays']!=2*c['contexts']*(c['native_modes']-c['gauge_modes'])
                or not 0<=c['selected_rays']<=min(c['rays'],2*(last-first)*(c['native_modes']-c['gauge_modes']))
                or c['selected_rays']%2):
            raise ValueError('Native sizing requires complete signed mode-ray counts')
        expected+=c['selected_rays']*settings['steps']
    elif mode_coverage_rows is not None:
        raise ValueError('Legacy paired sizing cannot claim additional mode rays')
    if (type(rows)is not int or rows != expected
            or type(bins) is not int or bins > request['grid']['bins']+1):
        raise ValueError('Native dimensions differ from the complete declared trajectory/grid scope')
    return size, min(EXAMPLE_RESERVE, size+OVERHEAD)


def publish(request, *, rows, bins, mode_dimensions, cell_dimensions, numeric_bytes,mode_coverage_rows=None):
    request = _context(request)
    size, reserve = _dimensions(request, rows, bins, mode_dimensions, cell_dimensions,mode_coverage_rows)
    if type(numeric_bytes) is not int or numeric_bytes != size:
        raise ValueError('The native array inventory differs from its dimensional reservation')
    path = Path(request['destination'])/NAME
    if path.exists() or path.is_symlink():
        raise ValueError('Native size records belong to one fresh task attempt')
    extra={} if mode_coverage_rows is None else dict(mode_coverage_rows=mode_coverage_rows)
    write_json(path, dict(schema=MODE_SCHEMA if extra else SCHEMA, request_sha256=digest(request),
        implementation_sha256=package_signature(), rows=rows, bins=bins,
        mode_dimensions=mode_dimensions, cell_dimensions=cell_dimensions,
        numeric_bytes=size, reservation_bytes=reserve,**extra))


def read(request, implementation_sha256):
    """Read only the task's fresh output namespace, never captured input files."""
    request = _context(request)
    path = Path(request['destination'])/NAME
    # A producer has not published its dimensions yet. Keep this distinct
    # from an existing invalid record in scheduler diagnostics. The bounded
    # no-follow reader below still checks the opened file against replacement.
    path.lstat()
    raw = read_bounded(path, MAX_RECORD_BYTES)
    value = json.loads(raw)
    fields = {'schema', 'request_sha256', 'implementation_sha256', 'rows', 'bins',
              'mode_dimensions', 'cell_dimensions', 'numeric_bytes', 'reservation_bytes'}
    named='mode_coverage' in request['settings']
    if named:fields.add('mode_coverage_rows')
    if (not isinstance(value, dict) or set(value) != fields or value['schema'] != (MODE_SCHEMA if named else SCHEMA)
            or value['request_sha256'] != digest(request)
            or value['implementation_sha256'] != implementation_sha256):
        raise ValueError('Native sizing record differs from its owned task or worker implementation')
    size, reserve = _dimensions(request, value['rows'], value['bins'],
                                value['mode_dimensions'], value['cell_dimensions'],value.get('mode_coverage_rows'))
    if (type(value['numeric_bytes']) is not int or value['numeric_bytes'] != size
            or type(value['reservation_bytes']) is not int or value['reservation_bytes'] != reserve):
        raise ValueError('Native sizing byte counts differ from their dimensions')
    return dict(value, record_sha256=hashlib.sha256(raw).hexdigest())
