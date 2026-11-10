"""One closed payload contract for NOVA dispatch and the Galaxy native worker."""
import math

FIELDS = {
    'inspect': ({'source'}, set()),
    'export': ({'source', 'phase', 'destination'}, set()),
    'match': ({'cif', 'k', 'options'}, set()),
    'modes': ({'child', 'temporary'}, set()),
    'modes-many': ({'children', 'workers'}, set()),
    'k-prepare': ({'source', 'source_sha256', 'parent', 'histogram'}, set()),
    'k-inventory': ({'source', 'source_sha256', 'parent', 'parent_setting_sha256',
                     'sites', 'directions', 'catalogs'}, {'moment_model'}),
    'k-template': ({'directory'}, set()),
    'k-portfolio': ({'directories', 'workers', 'seconds'}, set()),
    'preparation-reference': ({'source', 'instrument', 'grid', 'settings', 'destination'}, set()),
    'preparation-magnetic': ({'source', 'instrument', 'grid', 'settings', 'destination'}, set()),
    'preparation-structural': ({'source', 'instrument', 'child', 'grid', 'settings', 'destination'}, set()),
    'preparation-kernel': ({'source', 'instrument', 'child', 'grid', 'settings', 'destination'}, set()),
    'preparation-batch': ({'source', 'plan', 'destination', 'workers', 'seconds'}, set()),
    'preparation-structural-examples': ({'source', 'instrument', 'child', 'grid', 'settings', 'destination'}, set()),
    'preparation-structural-training': ({'source', 'instrument', 'grid', 'plan', 'destination', 'settings'}, set()),
    'structural-snapshot': ({'source','directory','anchor_snapshot','snapshot','receipt','mapping',
        'source_scope','revision','source_revision','reference_revision','refine_transfer','destination'}, {'prepared_scattering','rank_structural'}),
}
STRUCTURAL_SNAPSHOT_OPERATIONS=frozenset({'structural-snapshot','structural-snapshot-ranking'})
FIELDS['structural-snapshot-ranking']=(FIELDS['structural-snapshot'][0],{'prepared_scattering'})
from ..planning.managed import OPERATIONS as PLANNING_OPERATIONS
FIELDS.update({name: ({'source','plan','children','destination'}, set()) for name in PLANNING_OPERATIONS})


def native_timeout_limit(operation):
    if operation not in FIELDS:raise ValueError('Unsupported native operation')
    # Complete signed-mode simulation units can take longer than a short
    # multi-model batch. They receive independent jobs, retaining all curves.
    # These offline ceilings never widen fit, discovery or batch deadlines.
    return {'preparation-structural-training':7200,
            'preparation-structural-examples':3600}.get(operation,600)


def validate_native_timeout(operation,value):
    if (type(value) not in {int,float} or not math.isfinite(value)
            or not 1<=value<=native_timeout_limit(operation)):
        raise ValueError('Invalid native deadline for this operation')
    return float(value)


def validate_native_payload(operation, payload):
    if operation not in FIELDS or not isinstance(payload, dict):
        raise ValueError('Unsupported native operation')
    required, optional = FIELDS[operation]
    if set(payload) - required - optional or required - set(payload):
        raise ValueError('Unexpected native operation fields for ' + operation)
    if operation == 'k-inventory' and payload.get('moment_model', 'collinear') not in {
        'collinear', 'symmetry_allowed',
    }:
        raise ValueError('Choose a supported magnetic moment model')
    if operation.startswith('preparation-'):
        from ..preparation.worker import validate_request
        validate_request(operation.removeprefix('preparation-'), payload)
    if operation in STRUCTURAL_SNAPSHOT_OPERATIONS:
        from ..structural_live.job import validate_payload
        validate_payload(payload)
    if operation in PLANNING_OPERATIONS:
        from ..planning.managed import validate_payload
        validate_payload(operation,payload)
