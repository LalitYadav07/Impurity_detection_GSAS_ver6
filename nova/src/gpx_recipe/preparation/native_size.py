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
PROJECTED_SCHEMA = 'radar-preparation-native-size/v3'
PROJECTED_MODE_SCHEMA = 'radar-preparation-native-size/v4'
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


def _dimensions(request, rows, bins, mode_dimensions, cell_dimensions,mode_coverage_rows=None,native_mode_dimensions=None):
    settings = request['settings']; first, last = settings['trajectory_slice']
    size = paired_array_bytes(rows, bins, mode_dimensions, cell_dimensions)
    expected=1+(last-first)*(settings['steps']+1)
    if 'mode_coverage' in settings:
        from .structural_mode_coverage import SCHEMAS as COVERAGE,TARGETED_SCHEMA,CONTEXTS_PER_MODE
        c=mode_coverage_rows
        fields={'schema','native_modes','gauge_modes','contexts','rays','selected_rays'}
        if (not isinstance(c,dict) or set(c)!=fields or c['schema']!=settings['mode_coverage']
                or c['schema'] not in COVERAGE
                or any(type(c[k])is not int for k in fields-{'schema'})
                or not 0<=c['gauge_modes']<=c['native_modes']<=2048
                or c['native_modes']-c['gauge_modes']<mode_dimensions
                or c['contexts']!=min(CONTEXTS_PER_MODE,settings['trajectories'])
                or c['rays']!=2*c['contexts']*(c['native_modes']-c['gauge_modes'])
                or not 0<=c['selected_rays']<=min(c['rays'],2*(last-first)*(c['native_modes']-c['gauge_modes']))
                or c['selected_rays']%2):
            raise ValueError('Native sizing requires complete signed mode-ray counts')
        expected+=c['selected_rays']*(settings['steps']+int(c['schema']==TARGETED_SCHEMA))
    elif mode_coverage_rows is not None:
        raise ValueError('Legacy paired sizing cannot claim additional mode rays')
    if (type(rows)is not int or rows != expected
            or type(bins) is not int or bins > request['grid']['bins']+1):
        raise ValueError('Native dimensions differ from the complete declared trajectory/grid scope')
    projection_bytes=0
    if native_mode_dimensions is not None:
        from .structural_mode_projection import MAX_NATIVE,MAX_BYTES as MAX_PROJECTION_BYTES
        n=native_mode_dimensions;m=mode_dimensions
        if type(n)is not int or not m<=n<=MAX_NATIVE:
            raise ValueError('A projected reservation needs every native and physical coordinate')
        projection_bytes=8*(n*n+m*m)
        if projection_bytes>MAX_PROJECTION_BYTES:
            raise ValueError('Native coordinate arrays exceed their portable byte bound')
    # Keep the existing aggregate artifact cap. Projection bytes join the
    # reservation; NPZ/header/JSON overhead is already covered by OVERHEAD.
    return size, min(EXAMPLE_RESERVE, size+projection_bytes+OVERHEAD)


def publish(request, *, rows, bins, mode_dimensions, cell_dimensions, numeric_bytes,mode_coverage_rows=None,native_mode_dimensions=None):
    request = _context(request)
    size, reserve = _dimensions(request, rows, bins, mode_dimensions, cell_dimensions,mode_coverage_rows,native_mode_dimensions)
    if type(numeric_bytes) is not int or numeric_bytes != size:
        raise ValueError('The native array inventory differs from its dimensional reservation')
    path = Path(request['destination'])/NAME
    if path.exists() or path.is_symlink():
        raise ValueError('Native size records belong to one fresh task attempt')
    extra={} if mode_coverage_rows is None else dict(mode_coverage_rows=mode_coverage_rows)
    schema=MODE_SCHEMA if extra else SCHEMA
    if native_mode_dimensions is not None:
        extra['native_mode_dimensions']=native_mode_dimensions
        schema=PROJECTED_MODE_SCHEMA if mode_coverage_rows is not None else PROJECTED_SCHEMA
    write_json(path, dict(schema=schema, request_sha256=digest(request),
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
    projected=isinstance(value,dict) and 'native_mode_dimensions' in value
    if projected:fields.add('native_mode_dimensions')
    schema=(PROJECTED_MODE_SCHEMA if named else PROJECTED_SCHEMA) if projected else (MODE_SCHEMA if named else SCHEMA)
    if (not isinstance(value, dict) or set(value) != fields or value['schema'] != schema
            or value['request_sha256'] != digest(request)
            or value['implementation_sha256'] != implementation_sha256):
        raise ValueError('Native sizing record differs from its owned task or worker implementation')
    size, reserve = _dimensions(request, value['rows'], value['bins'],
                                value['mode_dimensions'], value['cell_dimensions'],value.get('mode_coverage_rows'),value.get('native_mode_dimensions'))
    if (type(value['numeric_bytes']) is not int or value['numeric_bytes'] != size
            or type(value['reservation_bytes']) is not int or value['reservation_bytes'] != reserve):
        raise ValueError('Native sizing byte counts differ from their dimensions')
    return dict(value, record_sha256=hashlib.sha256(raw).hexdigest())
