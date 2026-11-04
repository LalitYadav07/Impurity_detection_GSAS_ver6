"""One closed payload contract for NOVA dispatch and the Galaxy native worker."""

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
}


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
