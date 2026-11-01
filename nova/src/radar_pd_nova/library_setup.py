"""Independent workflow drafts for the common candidate-library builder."""

from __future__ import annotations

import asyncio
import json
import hashlib
import math
import os
from pathlib import Path
import re
import zipfile
from typing import Any, Callable

LIBRARY_FIELDS = (
    "library_review", "library_review_source", "library_review_preview", "library_review_loading", "library_review_message",
    "library_review_filter",
    "library_builder_allow_rounding", "library_builder_rounding_note",
    "database_source",
    "library_archive_source",
    "database_archive_path",
    "library_builder_cif_ids",
    "library_builder_local_paths",
    "library_builder_name",
    "library_builder_mode",
    "library_builder_active",
    "library_builder_status",
    "library_builder_progress",
    "library_builder_built_count",
    "library_builder_skipped_count",
    "library_builder_failure_rows",
    "library_builder_message",
    "library_builder_file_rows",
    "library_builder_elements",
    "library_builder_unknown_elements",
    "library_builder_requested_count",
    "library_builder_duplicate_count",
    "library_builder_duplicate_rows",
    "library_builder_generation",
    "library_builder_partial_review",
    "library_builder_previous_library",
    "library_builder_decision_pending",
    "library_builder_decision_token",
    "library_builder_failure_search",
)


def library_source_key(state: Any) -> str:
    return json.dumps([getattr(state,'database_source',None),getattr(state,'library_archive_source',None),
                       getattr(state,'history_database_id',None),getattr(state,'database_archive_path',None)])


def file_sha256(path: str | Path) -> str:
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def _hash_library_stream(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def _library_file_identity(value) -> tuple:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def inspect_library_review(path: str | Path) -> dict[str, Any]:
    """Bind the declared review and digest to one unchanged open archive."""
    path = Path(path)
    with path.open('rb') as stream:
        before = _library_file_identity(os.fstat(stream.fileno()))
        result = _inspect_library_review_contents(stream)
        digest = _hash_library_stream(stream)
        after = _library_file_identity(os.fstat(stream.fileno()))
        try:
            current = _library_file_identity(path.stat())
        except OSError as exc:
            raise ValueError('The library archive changed during review; select and review it again') from exc
        if before != after or before != current:
            raise ValueError('The library archive changed during review; select and review it again')
        result['archive_sha256'] = digest
        return result


def _inspect_library_review_contents(source) -> dict[str, Any]:
    """Inspect bounded members of the caller's already-open archive."""
    with zipfile.ZipFile(source) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('Library archive contains duplicate member names')
        names=[name for name in archive.namelist() if name.rstrip('/').endswith('/manifest.json') or name=='manifest.json']
        if not names:
            return {'policy_set_sha256':hashlib.sha256(b'{}').hexdigest(),'phase_count':None,
                    'reviews':[],'failures':[],'requires_review':False,
                    'evidence_verification':'unverified archive declaration','actual_affected_site_count':None}
        if len(names)!=1 or archive.getinfo(names[0]).file_size>2_000_000:
            raise ValueError('Library must contain one bounded manifest')
        manifest=json.loads(archive.read(names[0]))
        declared_counts={}
        if not any(key in manifest for key in ('phase_count','n_phases')) and 'n_added_phases' in manifest:
            # Historical augmented packs list only the added source IDs. They
            # must not be checked against (or labelled as) the full base count.
            added=library_manifest_phase_count({**manifest,'phase_count':manifest['n_added_phases']})
            base=manifest.get('n_base_phases')
            if base is not None and (isinstance(base,bool) or not isinstance(base,int) or base<0):
                raise ValueError('Manifest n_base_phases must be a non-negative integer')
            declared_counts={'added':added,'base':base,'total':base+added if base is not None else None}
            count=None
        else:
            count=library_manifest_phase_count(manifest)
        reviews=manifest.get('occupancy_reviews') or {}
        if not isinstance(reviews,dict):raise ValueError('Invalid library occupancy reviews')
        root=names[0][:-len('manifest.json')]
        mapping=json.loads(archive.read(root+'cif_map.json')) if reviews else {}
        if not isinstance(mapping,dict) or not set(reviews).issubset(mapping):
            raise ValueError('Reviewed sources are missing from the library CIF map')
        rows=[]
        for phase,review in reviews.items():
            if not isinstance(review,dict) or review.get('schema')!='radar-cif-occupancy-review/v1' or review.get('reviewed') is not True or review.get('normalization_applied') is not False:
                raise ValueError('Invalid reviewed occupancy record')
            excess=review.get('maximum_site_excess')
            if isinstance(excess,bool) or not isinstance(excess,(int,float)) or not 0<excess<=.001:
                raise ValueError('Reviewed occupancy exceeds the 0.001 cap')
            if not isinstance(review.get('review_note'),str) or not review['review_note'].strip():
                raise ValueError('Reviewed occupancy declaration has no review note')
            body={k:v for k,v in review.items() if k!='policy_sha256'}
            digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if digest!=review.get('policy_sha256'):raise ValueError('Library occupancy policy hash mismatch')
            if not isinstance(mapping[phase],str):
                raise ValueError('Invalid reviewed CIF mapping')
            member=root+mapping[phase]
            if Path(mapping[phase]).is_absolute() or '..' in Path(member).parts or archive.getinfo(member).file_size>2_000_000:
                raise ValueError('Invalid reviewed CIF member')
            file_hash=hashlib.sha256(archive.read(member)).hexdigest()
            if file_hash not in {review.get('source_sha256'),review.get('canonical_sha256')}:
                raise ValueError('Reviewed CIF bytes do not match the manifest')
            sites=review.get('affected_sites')
            composition=review.get('composition')
            def nonnegative_number(value):
                return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0
            if not isinstance(sites,list) or not sites:
                raise ValueError('An occupancy allowance needs declared affected-site evidence; an empty declaration cannot be reviewed')
            if not isinstance(composition,dict) or not composition or not all(nonnegative_number(v) for v in composition.values()) or sum(composition.values())<=0:
                raise ValueError('Reviewed occupancy declaration has no finite source composition')
            for site in sites:
                if not isinstance(site,dict):raise ValueError('Invalid declared site evidence')
                components=site.get('components');total=site.get('total');xyz=site.get('fractional_coordinates');labels=site.get('labels')
                if (not isinstance(components,dict) or not components
                    or not all(nonnegative_number(v) for v in components.values())
                    or not nonnegative_number(total) or not 1<total<=1+excess+1e-12
                    or not math.isclose(math.fsum(components.values()),total,rel_tol=0,abs_tol=1e-12)
                    or not isinstance(xyz,list) or len(xyz)!=3 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in xyz)
                    or not isinstance(labels,list) or not labels or not all(isinstance(v,str) and v.strip() for v in labels)):
                    raise ValueError('Inconsistent declared occupancy site evidence')
                rows.append({'phase':phase,'source_name':review.get('source_name') or phase,
                             'labels':labels,'total':total,'components':components,
                             'labels_display':', '.join(labels),
                             'components_display':json.dumps(components,sort_keys=True),
                             'maximum_site_excess':excess,'evidence_verification':'unverified declaration'})
        set_hash=hashlib.sha256(json.dumps({k:v['policy_sha256'] for k,v in reviews.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        try:
            from .library_review import partial_library_archive_review
        except ImportError:
            partial = None
        else:
            partial = partial_library_archive_review(manifest)
        incomplete = partial['requires_review'] if partial is not None else bool(manifest.get('failures'))
    return {'policy_set_sha256':set_hash,
            'phase_count':count,'reviews':rows,'failures':list(manifest.get('failures') or []),
            'declared_augmented_counts':declared_counts,
            'requires_review':bool(reviews or incomplete),'requires_incomplete_review':incomplete,
            'manifest':manifest,'partial_review':partial,
            'evidence_verification':'unverified archive declaration',
            'actual_affected_site_count':None,'declared_affected_site_count':len(rows)}


def activation_for_review(preview: dict[str, Any]) -> dict[str, Any]:
    return {'acknowledged':True,'archive_sha256':preview['archive_sha256'],
            'policy_set_sha256':preview['policy_set_sha256'],
            'allow_incomplete':bool(preview.get('requires_incomplete_review',preview.get('failures')))}


def require_library_activation(preview: dict[str, Any], activation: dict[str, Any]) -> None:
    if not preview['requires_review'] and not activation:return
    expected=activation_for_review(preview)
    if not activation or any(activation.get(k)!=v for k,v in expected.items()):
        raise ValueError('Review the selected library occupancy allowances and failed sources before analysis; the source or policy may have changed')


class LibraryManifestReadError(ValueError):
    """The completed build's manifest could not be retrieved and verified."""


async def read_library_manifest(
    reader: Callable[[str], Any],
    dataset_id: str | None,
    *,
    on_retry: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Re-read publication once; never resubmit the completed scientific job."""
    if not dataset_id:
        raise LibraryManifestReadError("The completed build has no manifest output.")
    failure = "empty or non-object document"
    for attempt in range(2):
        try:
            document = await asyncio.to_thread(reader, dataset_id)
        except Exception as exc:
            # Preserve the failure category without exposing response bodies or
            # credential-bearing request details in the UI.
            failure = type(exc).__name__
        else:
            if isinstance(document, dict) and document:
                return document
            failure = "empty or non-object document"
        if attempt == 0:
            if on_retry:
                on_retry(failure)
            await asyncio.sleep(1.0)
    raise LibraryManifestReadError(f"Manifest retrieval failed after two reads ({failure}).")


def library_manifest_phase_count(manifest: dict[str, Any]) -> int:
    """Require consistent integral counts, including IDs when supplied."""
    counts = []
    for key in ("phase_count", "n_phases"):
        if key not in manifest:
            continue
        value = manifest[key]
        if isinstance(value, bool):
            raise ValueError(f"Manifest {key} must be a non-negative integer.")
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
            value = int(value)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Manifest {key} must be a non-negative integer.")
        counts.append(value)
    if not counts:
        raise ValueError("Manifest contains no usable phase count.")
    if len(set(counts)) != 1:
        raise ValueError("Manifest phase counts disagree.")
    count = counts[0]
    if "phase_ids" in manifest:
        identifiers = manifest["phase_ids"]
        if (
            not isinstance(identifiers, list)
            or any(not isinstance(item, str) or not item.strip() for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or len(identifiers) != count
        ):
            raise ValueError("Manifest phase IDs do not match its usable phase count.")
    return count


class LibraryDraft:
    """Give the existing builder one explicit destination for all state changes."""

    def __init__(self, state: Any, workflow: str = "single"):
        if workflow not in {"single", "live"}:
            raise ValueError("Unknown candidate-library workflow")
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "workflow", workflow)

    def _key(self, name: str) -> str:
        if self.workflow == "live":
            if name == "history_database_id":
                return "powgen_database_dataset_id"
            if name in LIBRARY_FIELDS or name == "radiation":
                return "live_" + name
        return name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, self._key(name))

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._state, self._key(name), value)

    def signature(self) -> str:
        return json.dumps(
            {
                name: getattr(self, name, None)
                for name in (
                    "database_source",
                    "library_builder_name",
                    "library_builder_cif_ids",
                    "library_builder_local_paths",
                    "library_builder_allow_rounding", "library_builder_rounding_note",
                    "radiation",
                )
            },
            sort_keys=True,
        )
