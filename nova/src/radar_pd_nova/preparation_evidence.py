"""Find compact origin receipts through the user's existing Galaxy access.

References in a portable library are lookup hints. The fixed Galaxy tool
template independently supplies actual creating-job provenance to workers.
"""
from pathlib import Path
import logging
import uuid
import requests

from gpx_recipe.preparation.certificates import read,MAX_BYTES
from gpx_recipe.preparation.evidence import site,select
from gpx_recipe.symmetry.compiler import file_sha

LOG=logging.getLogger(__name__)


def remember(library,runtime,handle,root):
    try:return _remember(library,runtime,handle,root)
    except (OSError,ValueError,requests.RequestException):
        # Optional acceleration provenance must not invalidate checked arrays.
        LOG.warning('Native preparation origin receipt unavailable; imported arrays require native requalification')
        return False


def _remember(library,runtime,handle,root):
    service=runtime.service
    if not getattr(service,'galaxy_url',None):return False
    from .native_operations import recover_job
    job=recover_job(runtime,handle)
    if job is None:return False
    service.refresh_utility(job)
    identifier=job.outputs.get('preparation_certificate')
    if not identifier:return False
    from .model_runtime import identifier as checked_id
    identifier=checked_id(identifier)
    path=Path(root)/('origin-'+uuid.uuid4().hex+'.json')
    try:
        service._download_dataset(identifier,path,maximum_bytes=MAX_BYTES)
        certificate=read(path)
        available={a['sha256'] for row in library.document['entries'] for a in row['files'] if a['path'].endswith('/entry.json')}
        entries=sorted(available&{r['entry_sha256'] for r in certificate['entries']})
        if not entries:return False
        library.add_evidence(dict(site=site(service.galaxy_url),dataset_id=identifier,sha256=file_sha(path),entries=entries))
        return True
    finally:path.unlink(missing_ok=True)


def input_ids(runtime,request):
    document=request.get('prepared_scattering') or {}
    if request.get('operation') in {'structural-snapshot','structural-snapshot-ranking'}:
        document=request.get('payload',{}).get('prepared_scattering') or {}
    if not document.get('evidence'):return []
    service=runtime.service
    capability=getattr(service,'tool_has_input',None)
    if not capability:return []
    try:
        if not capability(runtime.tool_id,'preparation_evidence'):return []
    except requests.RequestException:
        LOG.warning('Native preparation origin input unavailable; using native requalification')
        return []
    expected=[r['entry_sha256'] for r in document.get('entries',[])]
    from gpx_recipe.preparation.tof_profile_hints import entry_hashes
    expected+=entry_hashes(document)
    references=select(document['evidence'],expected);matches=[]
    current=site(service.galaxy_url);cache=runtime.root/'preparation-origin-cache';cache.mkdir(parents=True,exist_ok=True)
    for row in references:
        # Never contact a URL supplied by a library or transfer its receipt
        # reference to an unrelated Galaxy installation.
        if row['site']!=current:continue
        path=cache/(row['sha256']+'.json')
        try:
            if not path.is_file():service._download_dataset(row['dataset_id'],path,maximum_bytes=MAX_BYTES)
            if path.is_symlink() or file_sha(path)!=row['sha256']:continue
            certificate=read(path)
            if set(row['entries']).issubset({r['entry_sha256'] for r in certificate['entries']}):
                matches.append(row['dataset_id'])
        except (OSError,ValueError,requests.RequestException):
            LOG.warning('Native preparation origin could not be read; using native requalification')
            continue
    return list(dict.fromkeys(matches))[:256]
