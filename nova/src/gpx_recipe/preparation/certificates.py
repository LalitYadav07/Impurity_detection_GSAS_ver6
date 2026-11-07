"""Data-only receipts announced by the trusted Galaxy tool command.

A receipt is not self-authenticating. Reuse requires its exact hash in the
creating job's server-supplied stdout, not a claim inside an uploaded bundle.
"""
import argparse
import json
from pathlib import Path
import re

from ..execution import write_json
from ..symmetry.compiler import file_sha,package_signature

SCHEMA='radar-native-preparation-certificate/v1'
MARKER='[RADAR PREPARATION CERTIFICATE] '
MAX_BYTES=2*1024*1024
MAX_ENTRIES=256


def checksum(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'schema','implementation_sha256','result_archive_sha256','entries'}
            or value['schema']!=SCHEMA or not checksum(value['implementation_sha256'])
            or not checksum(value['result_archive_sha256']) or not isinstance(value['entries'],list)
            or len(value['entries'])>MAX_ENTRIES or len(json.dumps(value,allow_nan=False))>MAX_BYTES):
        raise ValueError('Invalid bounded native preparation certificate')
    seen=set()
    for row in value['entries']:
        if (not isinstance(row,dict) or set(row)!={'kind','entry_sha256','physics_sha256','artifacts'}
                or row['kind'] not in {'magnetic','structural','kernel'}
                or not checksum(row['entry_sha256']) or row['entry_sha256'] in seen
                or not checksum(row['physics_sha256']) or not isinstance(row['artifacts'],dict)
                or not row['artifacts'] or len(row['artifacts'])>8
                or any(not isinstance(k,str) or not re.fullmatch('[a-z][a-z0-9_.-]{0,80}',k)
                    or not checksum(v) for k,v in row['artifacts'].items())):
            raise ValueError('Invalid native preparation certificate member')
        if row['kind']=='structural' and set(row['artifacts'])!={'starts.npz','starts.json'}:
            raise ValueError('Structural certificate requires the complete native bank and receipt')
        if row['kind']=='magnetic' and set(row['artifacts']) not in ({'response.npz'},{'response.npz','scattering.npz'}):
            raise ValueError('Magnetic certificate requires its declared native arrays')
        if row['kind']=='kernel' and set(row['artifacts'])!={'kernel.npz','kernel.json'}:
            raise ValueError('Kernel certificate requires its complete numeric chunk and metadata')
        seen.add(row['entry_sha256'])
    return value


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_BYTES:
        raise ValueError('Native preparation certificate is absent or oversized')
    return validate(json.loads(path.read_text()))


def produce(response,worker_root,archive,destination):
    """Called only after the installed native wrapper completed and packed.

    Native inputs cannot provide certificates or choose published array hashes.
    Failed and unstarted members never appear in this receipt.
    """
    root=Path(worker_root).resolve();entries=[]
    inner=response.get('result',{}) if response.get('schema')=='radar-model-native-result/v1' else {}
    if inner.get('schema')=='radar-experiment-preparation-worker/v1':
        results=[inner]
    elif inner.get('schema')=='radar-preparation-batch-result/v1':
        results=[r['result'] for r in inner['entries'] if r['state']=='prepared']
    else:results=[]
    for result in results:
        if result.get('kind') not in {'structural','magnetic','kernel'}:continue
        if (result.get('schema')!='radar-experiment-preparation-worker/v1' or result.get('status')!='prepared'
                or result.get('observations_included') is not False or result.get('scientific_result') is not False):
            raise ValueError('Only completed theoretical native preparation can be certified')
        output=Path(result['output']).resolve()
        if not output.is_relative_to(root):raise ValueError('Certified preparation escaped its native workspace')
        path=output/'entry.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_BYTES:
            raise ValueError('Native preparation manifest is absent or oversized')
        manifest=json.loads(path.read_text())
        from .operations import portable_entry_artifacts
        portable_entry_artifacts(result['kind'],manifest)
        if (manifest.get('schema')!='radar-experiment-preparation-entry/v1' or manifest.get('kind')!=result['kind']
                or file_sha(path)!=result['entry_sha256'] or manifest['implementation_sha256']!=package_signature()
                or manifest['observations_included'] is not False or manifest['scientific_result'] is not False):
            raise ValueError('Native preparation certificate differs from its completed calculation')
        for name,sha in manifest['artifacts'].items():
            artifact=output/name
            if artifact.is_symlink() or file_sha(artifact)!=sha:
                raise ValueError('Native preparation array changed before certification')
        if result['kind']=='kernel':
            from .tof_kernel_arrays import read as read_kernel
            from .kernel_runtime import origin_identity,validate_record
            from ..symmetry.contracts import digest
            data=read_kernel(output/'kernel.npz',output/'kernel.json')
            detail=manifest['detail'];proof=detail.get('native_kernel_proof',{})
            runtime=validate_record(detail.get('kernel_runtime'))
            if (detail.get('arrays_sha256')!=data.metadata['arrays_sha256']
                    or runtime['native']!={k:data.metadata[k] for k in ('kernel_sha256','binary_sha256')}
                    or digest(runtime)!=detail.get('kernel_runtime_sha256')
                    or detail.get('physics_sha256')!=origin_identity(
                        data.metadata['arrays_sha256'],detail.get('kernel_runtime_sha256'))
                    or proof.get('verified_all_peaks') is not True
                    or proof.get('native_peak_calls')!=data.metadata['reflection_count']
                    or proof.get('method') not in {'original native kernel construction',
                                                   'complete original native peak verification'}):
                raise ValueError('Kernel certificate needs its completed native construction and exact runtime')
        # Bind the certificate to the exact files already inventoried and
        # packed by the wrapper, including the manifest. A later mutation
        # cannot receive a receipt for different bytes than the result ZIP.
        for artifact in [path]+[output/name for name in manifest['artifacts']]:
            if response.get('files',{}).get(str(artifact.relative_to(root)))!=file_sha(artifact):
                raise ValueError('Native preparation certificate differs from the published inventory')
        entries.append(dict(kind=result['kind'],entry_sha256=file_sha(path),
            physics_sha256=manifest['detail']['physics_sha256'],artifacts=manifest['artifacts']))
    document=validate(dict(schema=SCHEMA,implementation_sha256=package_signature(),
        result_archive_sha256=file_sha(archive),entries=sorted(entries,key=lambda r:r['entry_sha256'])))
    destination=Path(destination)
    if destination.is_symlink():raise ValueError('Certificate output must not be a symlink')
    write_json(destination,document)
    return document


def announce(path):
    read(path)
    print(MARKER+file_sha(path),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path');args=parser.parse_args();announce(args.path)
