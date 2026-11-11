"""Trusted-launcher provenance, separate from all uploaded scientific data.

Galaxy renders the envelope from selected receipt datasets and their creating
jobs. The installed command, not a request field, supplies its path. Numerical
children receive only an ephemeral, hash-bound capsule in their environment.
No API key, signing secret or executable object is required or serialized.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading

from ..symmetry.compiler import file_sha,package_signature
from .certificates import read,validate,MARKER,checksum

SCHEMA='radar-galaxy-preparation-evidence/v1'
CAPSULE='radar-native-preparation-authority/v1'
ENV_PATH='RADAR_NATIVE_PREPARATION_AUTHORITY'
ENV_HASH='RADAR_NATIVE_PREPARATION_AUTHORITY_SHA256'
MAX_BYTES=4*1024*1024
TOOLS={'neutrons_radar_pd_model_campaign_prototype','neutrons_radar_pd_model_campaign'}
# Only wrappers reviewed with the separate, server-rendered provenance input
# and the installed certificate producer belong here. Versions 0.3.0–0.3.3
# omitted that contract; do not authorize them simply because they are newer.
VERSIONS={'0.2.17','0.2.18','0.3.4','0.3.5','0.3.6'}
_lock=threading.Lock()
_owned=None


def verify_envelope(path):
    """Optional invalid origins are denied, not permission to bypass science."""
    accepted=[];rejected=[]
    if path is None:return accepted,rejected
    path=Path(path)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_BYTES:
            raise ValueError('Server preparation envelope is absent or oversized')
        data=json.loads(path.read_text())
        if (not isinstance(data,dict) or set(data)!={'schema','records'} or data['schema']!=SCHEMA
                or not isinstance(data['records'],list) or len(data['records'])>256):
            raise ValueError('Invalid server preparation envelope')
    except (OSError,ValueError):return [],['server_envelope_unavailable']
    for row in data['records']:
        try:
            if (not isinstance(row,dict) or set(row)!={'certificate_path','tool_id','tool_version','job_state',
                    'dataset_state','dataset_deleted','dataset_purged','job_stdout'}
                    or row['tool_id'] not in TOOLS or row['tool_version'] not in VERSIONS
                    or row['job_state']!='ok' or row['dataset_state']!='ok'
                    or row['dataset_deleted'] is not False or row['dataset_purged'] is not False
                    or not isinstance(row['job_stdout'],str) or len(row['job_stdout'])>MAX_BYTES):
                raise ValueError('unrecognized_or_unready_native_producer')
            markers=[line[len(MARKER):] for line in row['job_stdout'].splitlines() if line.startswith(MARKER)]
            if len(markers)!=1 or not checksum(markers[0]):raise ValueError('missing_or_ambiguous_server_digest')
            certificate=read(row['certificate_path'])
            if file_sha(row['certificate_path'])!=markers[0]:raise ValueError('certificate_digest_changed')
            if certificate['implementation_sha256']!=package_signature():raise ValueError('scientific_implementation_changed')
            accepted.append(dict(sha256=markers[0],certificate=certificate))
        except (OSError,ValueError,KeyError,TypeError):
            rejected.append('preparation_origin_not_verified')
    unique={row['sha256']:row for row in accepted}
    return [unique[k] for k in sorted(unique)],rejected


@contextmanager
def server_scope(envelope,workspace):
    """The fixed native wrapper owns this boundary, including an empty scope.

    An empty scope deliberately suppresses any inherited authority variable at
    the outer tool entry point. Request JSON can never establish this scope.
    """
    global _owned
    if not _lock.acquire(blocking=False):raise RuntimeError('Another preparation authority owns this worker')
    prior=_owned
    try:
        records,rejected=verify_envelope(envelope)
        report=dict(verified_certificates=len(records),rejected_origins=len(rejected),
            source='Galaxy creating-job metadata',qualification_persisted_as_trust=False)
        if not records:
            _owned={};yield report
        else:
            with tempfile.TemporaryDirectory(prefix='.radar-preparation-authority-',dir=Path(workspace).resolve()) as directory:
                path=Path(directory)/'authority.json'
                capsule=dict(schema=CAPSULE,implementation_sha256=package_signature(),records=records)
                raw=json.dumps(capsule,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                if len(raw)>MAX_BYTES:raise ValueError('Native preparation authority exceeds its bound')
                with path.open('xb') as stream:stream.write(raw)
                path.chmod(0o600)
                _owned={ENV_PATH:str(path),ENV_HASH:file_sha(path)}
                yield report
    finally:
        _owned=prior;_lock.release()


def _inherited(environment=None):
    environment=os.environ if environment is None else environment
    path=environment.get(ENV_PATH);sha=environment.get(ENV_HASH)
    if not path or not checksum(sha):return None
    path=Path(path)
    try:
        if (path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_BYTES
                or path.name!='authority.json' or not path.parent.name.startswith('.radar-preparation-authority-')
                or file_sha(path)!=sha):return None
        value=json.loads(path.read_text())
        if (not isinstance(value,dict) or set(value)!={'schema','implementation_sha256','records'}
                or value['schema']!=CAPSULE or value['implementation_sha256']!=package_signature()
                or not isinstance(value['records'],list) or len(value['records'])>256):return None
        for record in value['records']:
            if (not isinstance(record,dict) or set(record)!={'sha256','certificate'} or not checksum(record['sha256'])):return None
            validate(record['certificate'])
            if record['certificate']['implementation_sha256']!=value['implementation_sha256']:return None
        return value
    except (OSError,ValueError,TypeError):return None


def child_environment():
    if _owned is not None:return dict(_owned)
    return {key:os.environ[key] for key in (ENV_PATH,ENV_HASH)} if _inherited() is not None else {}


def _current_capsule():
    """Validate the current authority once, without caching its trust.

    Child-environment filtering is a launch operation. Going through it and
    then parsing the same capsule again at each scientific lookup repeats all
    integrity and package checks. A lookup needs one complete current check;
    an explicitly empty outer scope must still suppress inherited variables.
    """
    return _inherited(_owned) if _owned is not None else _inherited()


def structural_bank_origin(bank_sha256,receipt_sha256,physics_sha256):
    """Match only certified bytes AND their current native physical context."""
    if not all(checksum(v) for v in (bank_sha256,receipt_sha256,physics_sha256)):return None
    # Owned scopes and child processes use the same bounded parser. Never
    # interpret a library's saved qualification/provenance flag as authority.
    capsule=_current_capsule()
    if capsule is None:return None
    for record in capsule['records']:
        for entry in record['certificate']['entries']:
            if (entry['kind']=='structural' and entry['physics_sha256']==physics_sha256
                    and entry['artifacts']=={'starts.npz':bank_sha256,'starts.json':receipt_sha256}):
                return dict(certificate_sha256=record['sha256'],entry_sha256=entry['entry_sha256'],
                    origin='Galaxy native preparation with current-context native probes')
    return None


def magnetic_factor_origin(factor_sha256):
    """Authorize only exact native amplitude-file bytes, never a fitted score.

    The reader separately matches the file's scattering identity to the full
    current geometry, symmetry, moment basis and native dependencies. The
    certificate cannot authorize another bank's binned response or likelihood.
    """
    if not checksum(factor_sha256):return None
    capsule=_current_capsule()
    if capsule is None:return None
    for record in capsule['records']:
        for entry in record['certificate']['entries']:
            if entry['kind']=='magnetic' and entry['artifacts'].get('scattering.npz')==factor_sha256:
                return dict(certificate_sha256=record['sha256'],entry_sha256=entry['entry_sha256'],
                    reference_physics_sha256=entry['physics_sha256'],
                    origin='Galaxy native preparation; exact amplitude bytes and current scattering identity')
    return None


def profile_table_origin(table_sha256,receipt_sha256,physics_sha256):
    """Authorize producer bytes for proposals only, never native likelihood."""
    if not all(checksum(v) for v in (table_sha256,receipt_sha256,physics_sha256)):return None
    capsule=_current_capsule()
    if capsule is None:return None
    for record in capsule['records']:
        for entry in record['certificate']['entries']:
            if (entry['kind']=='reference' and entry['physics_sha256']==physics_sha256
                    and entry['artifacts']=={'profile.npz':table_sha256,'profile.json':receipt_sha256}):
                return dict(certificate_sha256=record['sha256'],entry_sha256=entry['entry_sha256'],
                    origin='Galaxy native-compiled TOF proposal table; current shape and runtime required',
                    native_likelihood_authority=False)
    return None


def magnetic_response_origin(response_sha256, physics_sha256):
    """Match an exact native-produced response, including its physical scope."""
    if not all(checksum(v) for v in (response_sha256, physics_sha256)):return None
    capsule=_current_capsule()
    if capsule is None:return None
    for record in capsule['records']:
        for entry in record['certificate']['entries']:
            if (entry['kind']=='magnetic' and entry['physics_sha256']==physics_sha256
                    and entry['artifacts'].get('response.npz')==response_sha256):
                return dict(certificate_sha256=record['sha256'],entry_sha256=entry['entry_sha256'],
                    origin='Galaxy native preparation; exact unweighted rendered response and current physical context')
    return None


def kernel_origin(arrays_sha256,runtime_sha256):
    """Authorize exact semantic arrays only in their current native runtime.

    Array integrity is checked by the bounded decoder. The creating wrapper
    attests a complete original native construction. Current-runtime probes
    and exact actual-use primitive-input checks remain in the consumer.
    """
    if not all(checksum(v) for v in (arrays_sha256,runtime_sha256)):return None
    return _kernel_origin(_current_capsule(),arrays_sha256,runtime_sha256)


def current_kernel_origin(arrays_sha256,runtime_fingerprint):
    """Check provenance before computing a costly current runtime identity.

    The callback is installed Python code, never a field in a native request.
    No capsule or caller-supplied validation flag can establish authority here.
    Every call rechecks the actual launch capsule and scientific package.
    """
    if not checksum(arrays_sha256) or not callable(runtime_fingerprint):return None
    capsule=_current_capsule()
    if capsule is None or not any(entry['kind']=='kernel'
            for row in capsule['records'] for entry in row['certificate']['entries']):
        return None
    runtime_sha256=runtime_fingerprint()
    if not checksum(runtime_sha256):return None
    return _kernel_origin(capsule,arrays_sha256,runtime_sha256)


def _kernel_origin(capsule,arrays_sha256,runtime_sha256):
    from .kernel_runtime import origin_identity
    expected=origin_identity(arrays_sha256,runtime_sha256)
    if capsule is None:return None
    for record in capsule['records']:
        for entry in record['certificate']['entries']:
            if entry['kind']=='kernel' and entry['physics_sha256']==expected:
                return dict(certificate_sha256=record['sha256'],entry_sha256=entry['entry_sha256'],
                    native_runtime_sha256=runtime_sha256,
                    origin='Galaxy complete native kernel construction and matching numerical runtime')
    return None
