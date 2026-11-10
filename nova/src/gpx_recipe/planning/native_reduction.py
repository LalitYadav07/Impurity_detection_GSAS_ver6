"""Content-bound, isolated native operations for POWGEN validation.

Every operation owns a fresh process, copied inputs and an empty private cache.
Native cache filenames are not accepted as complete scientific identities.
This adapter records processing provenance; it does not qualify beam physics.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Lock

SCHEMA = 'powgen-native-reduction-request/v1'
SOURCE_MODULES = ('SNSPowderReduction', 'AlignAndFocusPowderFromFiles',
                  'CreateCacheFilename', 'mantid.utils.absorptioncorrutils')
FILE_PROPERTIES = {
    'SNSPowderReduction': {'Filename', 'CalibrationFile', 'CharacterizationRunsFile',
                          'GroupingFile', 'ExpIniFilename'},
    'AbsorptionCorrection': {'InputWorkspace'},
}
MANAGED = {'CacheDir', 'CleanCache', 'OutputDirectory', 'OutputWorkspace'}
_PROCESS_CLAIMED = False
_PROCESS_LOCK = Lock()


def _finite_json(value, depth=0):
    if depth > 16:
        raise ValueError('Native operation metadata exceeds the nesting budget')
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_finite_json(item, depth + 1) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _finite_json(item, depth + 1) for key, item in value.items()}
    raise ValueError('Native operation metadata must be finite JSON values')


def _canonical(value):
    result = json.dumps(_finite_json(value), sort_keys=True, separators=(',', ':'),
                        ensure_ascii=True, allow_nan=False).encode('ascii')
    if len(result) > 2 * 1024**2:
        raise ValueError('Native operation metadata exceeds 2 MiB')
    return result


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file() or before.st_size <= 0:
        raise ValueError('A nonempty native input file is required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('A native input changed while being hashed')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': digest.hexdigest()}


def _validate_request(request):
    request = json.loads(_canonical(request))
    if set(request) != {'schema', 'algorithm', 'parameters', 'inputs', 'file_bindings', 'runtime', 'context'} or request['schema'] != SCHEMA:
        raise ValueError('An explicit POWGEN native reduction request is required')
    algorithm = request['algorithm']
    if not isinstance(algorithm, str) or algorithm not in FILE_PROPERTIES:
        raise ValueError('Choose a supported native POWGEN operation')
    parameters = request['parameters']
    if not isinstance(parameters, dict) or set(parameters) & (MANAGED | FILE_PROPERTIES[algorithm]):
        raise ValueError('File and managed output/cache properties must use explicit bindings')
    if any('Workspace' in key for key in parameters):
        raise ValueError('Uncaptured in-memory workspace inputs are not accepted')
    prefix = parameters.get('OutputFilePrefix', '')
    if not isinstance(prefix, str) or (prefix and not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', prefix)):
        raise ValueError('An output prefix must be a simple filename prefix')
    inputs = request['inputs']
    if not isinstance(inputs, dict) or not 1 <= len(inputs) <= 256:
        raise ValueError('Declare 1–256 native input files')
    basenames = {}
    for role, item in inputs.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', role):
            raise ValueError('Use bounded semantic input roles')
        if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Every native input requires a path, byte count and SHA256')
        if (type(item['path']) is not str or not Path(item['path']).is_absolute()
                or type(item['bytes']) is not int or item['bytes'] <= 0
                or type(item['sha256']) is not str or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
            raise ValueError('Invalid native file identity')
        basename = Path(item['path']).name
        if basename in basenames and basenames[basename] != (item['bytes'], item['sha256']):
            raise ValueError('Different native inputs share a basename; reference lookup would be ambiguous')
        basenames[basename] = (item['bytes'], item['sha256'])
    bindings = request['file_bindings']
    if not isinstance(bindings, dict) or set(bindings) - FILE_PROPERTIES[algorithm]:
        raise ValueError('Unsupported native input property binding')
    for prop, roles in bindings.items():
        if (not isinstance(roles, list) or not roles
                or any(type(role) is not str or role not in inputs for role in roles)
                or len(set(roles)) != len(roles)):
            raise ValueError('Each native file property requires an ordered list of captured input roles')
    required = ({'Filename', 'CalibrationFile', 'CharacterizationRunsFile'}
                if algorithm == 'SNSPowderReduction' else {'InputWorkspace'})
    if not required <= set(bindings):
        raise ValueError('Missing required native input file bindings')
    for prop, roles in bindings.items():
        if prop not in {'Filename', 'CharacterizationRunsFile'} and len(roles) != 1:
            raise ValueError('A singleton native file property has multiple bindings')
    runtime = request['runtime']
    if not isinstance(runtime, dict) or set(runtime) != {'mantid_version', 'python_executable_sha256', 'source_sha256'}:
        raise ValueError('Declare the expected native version, executable and reduction sources')
    if not isinstance(runtime['mantid_version'], str) or not runtime['mantid_version']:
        raise ValueError('An explicit native Mantid version is required')
    hashes = runtime['source_sha256']
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_MODULES):
        raise ValueError('All native reduction source identities are required')
    if any(type(value) is not str or not re.fullmatch(r'[a-f0-9]{64}', value)
           for value in [runtime['python_executable_sha256'], *hashes.values()]):
        raise ValueError('Native runtime identities must be SHA256 values')
    if not isinstance(request['context'], dict):
        raise ValueError('Declare native operation context, including any model assumptions')
    return request


def operation_identity(request):
    """Hash declared full-file identities and physics/settings, without directories.

    A changed detector TZERO/DIFA, geometry file, model parameter, integration
    mesh or native source invalidates this identity. It does not prove that the
    caller has supplied every physical dependency; cache reuse is disabled.
    """
    request = _validate_request(request)
    payload = dict(request)
    payload['inputs'] = {role: dict({key: value for key, value in item.items() if key != 'path'},
                                   filename=Path(item['path']).name)
                         for role, item in request['inputs'].items()}
    payload['adapter_module_sha256'] = file_identity(__file__)['sha256']
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_request(algorithm, parameters, inputs, runtime, *, context):
    """Capture declared files. Inputs map semantic roles to path/property pairs."""
    captured, bindings = {}, {}
    for role, item in inputs.items():
        if not isinstance(item, dict) or set(item) != {'path', 'property'}:
            raise ValueError('Bind each native input path explicitly')
        captured[role] = file_identity(item['path'])
        if item['property'] is not None:
            if type(item['property']) is not str:
                raise ValueError('Native file property names must be strings')
            bindings.setdefault(item['property'], []).append(role)
    return _validate_request({'schema': SCHEMA, 'algorithm': algorithm,
                              'parameters': parameters, 'inputs': captured, 'file_bindings': bindings,
                              'runtime': runtime, 'context': context})


def claim_fresh_native_process(ads_names):
    """Reject existing workspaces and a second operation, even after ADS.clear()."""
    global _PROCESS_CLAIMED
    with _PROCESS_LOCK:
        if _PROCESS_CLAIMED:
            raise RuntimeError('Each native operation requires a fresh process')
        if list(ads_names):
            raise RuntimeError('Native operation must start with an empty workspace store')
        _PROCESS_CLAIMED = True


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _readonly_input_directory(request):
    """Require one exact flat bundle on a kernel-enforced read-only mount.

    File permissions alone are insufficient. Check each file as well as its
    directory, so a writable file bind mounted below a read-only directory
    cannot bypass the guard. Full content checks still occur before/after use.
    """
    paths = [Path(item['path']) for item in request['inputs'].values()]
    if any(path.is_symlink() for path in paths):
        raise ValueError('Read-only native inputs cannot be symbolic links')
    roots = {path.resolve(strict=True).parent for path in paths}
    if len(roots) != 1:
        raise ValueError('Read-only native inputs require one flat input directory')
    root = roots.pop()
    entries = list(root.iterdir())
    if (any(path.is_symlink() or not path.is_file() for path in entries)
            or {path.name for path in entries} != {path.name for path in paths}):
        raise ValueError('Read-only native directory must contain exactly the declared input files')
    checked = []
    for path in [root, *sorted(set(path.resolve(strict=True) for path in paths))]:
        flags = os.statvfs(path).f_flag
        if not flags & os.ST_RDONLY:
            raise ValueError('Native inputs require a kernel read-only mount; permissions alone are insufficient')
        checked.append({'path': str(path), 'statvfs_f_flag': flags, 'ST_RDONLY': True})
    return root, {'kernel_readonly_verified': True, 'checked': checked}


def execute_native_operation(request, output, *, input_mode='copy'):
    """Execute in the caller's new, offline Linux process using pinned Mantid.

    Run this entry point in a container/process with no facility cache mounted.
    The caller supplies all reference files. Missing references fail rather than
    authorizing archive discovery. No saved archive Python is executed.
    """
    request = _validate_request(request)
    if input_mode not in {'copy', 'read_only_mount'}:
        raise ValueError('Choose copied inputs or a verified read-only input mount')
    identity = operation_identity(request)
    output = Path(output).resolve()
    if (str(output).startswith(('/SNS/', '/HFIR/', '/gpfs/neutronsfs/instruments/',
                                '/gpfs/neutronsfs/instruments-hfir/'))
            or str(output) in {'/SNS', '/HFIR'}):
        raise ValueError('Native operation outputs must be outside facility archives')
    if output.exists():
        raise FileExistsError('Choose a new native operation output directory')
    if sys.platform != 'linux':
        raise RuntimeError('Use the pinned Linux Mantid runtime for native operations')
    from mantid.kernel import ConfigService
    ConfigService['UpdateInstrumentDefinitions.OnStartup'] = '0'
    ConfigService['CheckMantidVersion.OnStartup'] = '0'
    ConfigService['usagereports.enabled'] = '0'
    ConfigService['datasearch.searcharchive'] = 'Off'
    import mantid
    from mantid import simpleapi as api
    from mantid.api import mtd
    observed = {'mantid_version': str(mantid.__version__),
                'python_executable_sha256': file_identity(sys.executable)['sha256'],
                'source_sha256': {name: file_identity(importlib.import_module(name).__file__)['sha256']
                                  for name in SOURCE_MODULES}}
    if observed != request['runtime']:
        raise RuntimeError('Native runtime differs from the captured version, executable or sources')
    readonly_proof = None
    if input_mode == 'read_only_mount':
        inputs, readonly_proof = _readonly_input_directory(request)
        if output.is_relative_to(inputs):
            raise ValueError('Native output must be outside the read-only input directory')
    claim_fresh_native_process(mtd.getObjectNames())
    output.mkdir(parents=True)
    if input_mode == 'copy':
        inputs = output / 'inputs'; inputs.mkdir()
    products = output / 'products'; products.mkdir()
    cache = output / ('private-cache-' + identity); cache.mkdir()
    _write(output / 'request.json', request)
    _write(output / 'processing.json', {
        'schema': 'powgen-isolated-native-processing/v1', 'operation_identity': identity,
        'adapter_module_sha256': file_identity(__file__)['sha256'], 'input_capture_mode': input_mode,
        'kernel_readonly_mount_verified': readonly_proof is not None,
        'native_runtime_observed': observed, 'fresh_process_claimed': True,
        'private_cache': cache.name, 'native_cache_imported_or_reused_between_jobs': False,
        'complete_runtime_binary_tree_independently_verified': False,
        'physical_dependency_completeness_certified': False,
        'experimental_accuracy_certified': False})
    if readonly_proof is not None:
        _write(output / 'readonly-inputs-before.json', readonly_proof)
    try:
        staged = {}
        for role, item in request['inputs'].items():
            source = Path(item['path'])
            target = source if input_mode == 'read_only_mount' else inputs / source.name
            if input_mode == 'copy' and not target.exists():
                before = source.stat()
                digest, size = hashlib.sha256(), 0
                with source.open('rb') as src, target.open('xb') as dst:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        dst.write(block); digest.update(block); size += len(block)
                after = source.stat()
                if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                        or (size, digest.hexdigest()) != (item['bytes'], item['sha256'])):
                    raise ValueError('Native source changed or differs from its captured identity: ' + role)
                target.chmod(0o444)
            found = file_identity(target)
            if (found['bytes'], found['sha256']) != (item['bytes'], item['sha256']):
                raise ValueError('Staged native input differs from its captured identity: ' + role)
            staged[role] = found
        if input_mode == 'copy':
            inputs.chmod(0o555)
        ConfigService['datasearch.directories'] = str(inputs)
        parameters = dict(request['parameters'])
        for prop, roles in request['file_bindings'].items():
            parameters[prop] = ','.join(staged[role]['path'] for role in roles)
        if request['algorithm'] == 'SNSPowderReduction':
            ConfigService['default.facility'] = 'SNS'
            ConfigService['default.instrument'] = 'POWGEN'
            parameters.update(CacheDir=str(cache), CleanCache=False, OutputDirectory=str(products))
            api.SNSPowderReduction(**parameters)
        else:
            path = parameters.pop('InputWorkspace')
            api.LoadNexusProcessed(Filename=path, OutputWorkspace='__powgen_input')
            parameters.update(InputWorkspace='__powgen_input', OutputWorkspace='__powgen_correction')
            api.AbsorptionCorrection(**parameters)
            api.SaveNexusProcessed(InputWorkspace='__powgen_correction', Filename=str(products / 'correction.nxs'))
        for role, found in staged.items():
            if file_identity(found['path']) != found:
                raise ValueError('Native input changed during processing: ' + role)
        if readonly_proof is not None:
            _, after = _readonly_input_directory(request)
            if after != readonly_proof:
                raise ValueError('Native read-only input mount changed during processing')
            _write(output / 'readonly-inputs-after.json', after)
        result = {'schema': 'powgen-isolated-native-result/v1', 'operation_identity': identity,
                  'algorithm': request['algorithm'], 'staged_input_identities': staged,
                  'products': [file_identity(path) for path in sorted(products.iterdir()) if path.is_file()],
                  'workspace_names': list(mtd.getObjectNames()),
                  'archive_python_executed': False, 'experimental_accuracy_certified': False}
        if not result['products']:
            raise RuntimeError('Native operation returned without a saved numeric product')
        _write(output / 'result.json', result)
        return result
    except Exception as error:
        _write(output / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                                        'experimental_accuracy_certified': False})
        raise
