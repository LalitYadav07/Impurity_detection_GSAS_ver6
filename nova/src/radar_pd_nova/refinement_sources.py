"""Published-GPX discovery and explicit experiment/sample/scan scope.

This module never unpickles a project. The scientific worker owns GPX inspection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


@dataclass(frozen=True)
class RefinementScope:
    experiment: str
    mode: str = 'all'
    samples: tuple[str, ...] = ()
    scans: tuple[str, ...] = ()
    banks: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.experiment or self.mode not in {'all', 'samples', 'scans'}:
            raise ValueError('Choose an experiment and a valid refinement scope')
        for name in ('samples', 'scans', 'banks'):
            object.__setattr__(self, name, tuple(sorted(set(str(x) for x in getattr(self, name) if str(x)))))
        if self.mode == 'samples' and not self.samples:
            raise ValueError('Select at least one sample')
        if self.mode == 'scans' and not self.scans:
            raise ValueError('Select at least one scan')

    def matches(self, source: dict) -> bool:
        return (source['experiment'] == self.experiment
                and (not self.banks or str(source.get('bank', '')) in self.banks)
                and (self.mode != 'samples' or source.get('sample_key') in self.samples)
                and (self.mode != 'scans' or source.get('scan_key') in self.scans))

    def to_dict(self):
        return asdict(self)

    @property
    def signature(self):
        return digest(self.to_dict())


def experiment_key(history: str, instrument: str, ipts: str, *, record_uid='') -> str:
    if instrument and ipts:
        return digest([history, str(instrument).upper(), str(ipts).upper()])[:24]
    return digest([history, 'single', record_uid])[:24]


def _inside(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    target = (root / str(value)).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.suffix.lower() != '.gpx':
        return None
    return target


def select_published_project(result: dict, index: dict) -> dict:
    """Select the exact indexed baseline, never an arbitrary filesystem GPX."""
    projects = [p for p in (result.get('gpx_projects') or index.get('projects') or [])
                if isinstance(p, dict) and p.get('status') not in {'failed', 'rollback'}]
    if not projects:
        raise ValueError('The completed RADAR-PD run has no published GPX yet')
    model_hash = (result.get('observables') or {}).get('source_gpx_sha256')
    if not model_hash:
        model_hash = (result.get('pattern_observation') or {}).get('source_gpx_sha256')
    if model_hash:
        exact = [p for p in projects if p.get('sha256') == model_hash or p.get('source_sha256') == model_hash]
        if len(exact) == 1:
            return dict(exact[0])
        if len(exact) > 1:
            raise ValueError('Several published GPXs claim the selected model; choose a checkpoint explicitly')
        # A stale summary/index must not silently select a different solution.
        raise ValueError('The published GPX index does not identify the model used by this result')
    priorities = {'final_refinement': 0, 'main_phase_anchor': 1, 'targeted_refinement': 2}
    ranked = [p for p in projects if p.get('stage') in priorities and p.get('status') == 'accepted']
    if not ranked:
        raise ValueError('No accepted final GPX is identified; choose a checkpoint explicitly')
    best = min(priorities[p['stage']] for p in ranked)
    finalists = [p for p in ranked if priorities[p['stage']] == best]
    if len(finalists) != 1:
        raise ValueError('Several accepted GPX solutions exist; choose a checkpoint explicitly')
    return dict(finalists[0])


def published_project_path(root: Path, project: dict) -> Path | None:
    root = root.resolve()
    published = project.get('collection_path')
    name = project.get('collection_name') or (Path(str(published)).name if published else '')
    candidates = []
    if published:
        candidates.extend(_inside(root, p) for p in [published, 'ndip/' + str(published)])
    if name:
        candidates.extend(_inside(root, p) for p in ['gpx/' + name, 'ndip/gpx/' + name])
    candidates = list(dict.fromkeys(p for p in candidates if p is not None))
    if not candidates and name:
        candidates = [p.resolve() for p in root.rglob(name)
                      if p.is_file() and p.resolve().is_relative_to(root)]
    if not published and not name:
        candidates = list(dict.fromkeys(p for p in [_inside(root, project.get('path')),
                                                    _inside(root, project.get('source_path'))] if p))
    if len(candidates) > 1:
        expected = project.get('published_sha256')
        if expected:
            candidates = [p for p in candidates if file_sha(p) == expected]
        if len(candidates) > 1 and len({file_sha(p) for p in candidates}) == 1:
            candidates = candidates[:1]
    if len(candidates) > 1:
        raise ValueError('Ambiguous cached published GPX; several different files have the same publication name')
    return candidates[0] if candidates else None


def source_from_record(record, payload: dict, scan: dict | None, history: str,
                       *, sha_cache: dict | None = None, root_override: Path | None = None) -> dict:
    scan = scan or {}
    result = payload.get('summary') or {}
    index = payload.get('gpx_index') or {}
    project = select_published_project(result, index)
    root = root_override or record.local_output_dir
    if root is None:
        raise ValueError('RADAR-PD result files are not downloaded yet')
    path = published_project_path(root, project)
    checksum = None
    if path is not None:
        stat = path.stat(); stamp = (str(path), stat.st_size, stat.st_mtime_ns)
        checksum = (sha_cache or {}).get(stamp)
        if checksum is None:
            checksum = file_sha(path)
            if sha_cache is not None:
                sha_cache[stamp] = checksum
        expected = project.get('published_sha256')
        # Legacy indexes hash the internal source, before publication rewrites.
        if expected and checksum != expected:
            raise ValueError('Published GPX checksum changed; reload this RADAR-PD result')
    collection = str(record.output_dataset_ids.get('gpx_projects') or '')
    if path is None and not collection:
        raise ValueError('The indexed published GPX is unavailable locally and has no Galaxy collection')
    if checksum is None:checksum=project.get('published_sha256')
    selection=getattr(record,'inputs',None)
    acquisition=result.get('acquisition') or {}
    metadata=dict(acquisition.get('metadata') or {})
    metadata.update(scan.get('metadata') or {})
    scan={**dict(run_number=getattr(selection,'run_number',None),bank=getattr(selection,'bank',''),instrument=getattr(selection,'instrument',''),ipts=getattr(selection,'ipts','')),**scan}
    instrument = str(scan.get('instrument') or metadata.get('instrument') or '')
    ipts = str(scan.get('ipts') or metadata.get('ipts') or '')
    exp = experiment_key(history, instrument, ipts, record_uid=record.uid)
    sample_id = str(metadata.get('sample_id') or '').strip()
    sample_name = str(metadata.get('sample_name') or '').strip()
    sample_key = str((scan.get('sample_key') if scan.get('sample_key')!='unassigned' else None) or (('id:' + sample_id) if sample_id else 'scan:' + record.uid))
    run_number=scan.get('run_number')
    run_id = str(scan.get('run_id') or (f'{instrument}_{run_number}' if run_number else record.uid))
    bank = str(scan.get('bank') or metadata.get('source_bank') or metadata.get('bank') or '')
    scan_key = digest([exp,scan.get('experiment_number'),run_number or record.uid,bank])[:24]
    publication = str(project.get('collection_name') or Path(str(project.get('collection_path') or project.get('path') or '')).name)
    # Watch metadata can arrive after the archive. It must not create a second
    # source identity or enroll the same published result a second time.
    uid = digest([history, record.uid, project.get('sha256'), project.get('published_sha256'), publication])[:24]
    scan_label=str(scan.get('scan_label') or record.name)
    if bank and 'bank' not in scan_label.lower():scan_label+=' · bank '+bank
    return dict(id=uid, experiment=exp, experiment_label=(f'{instrument} · {ipts}' if ipts else record.name),
                scan_key=scan_key, record_uid=record.uid, run_id=run_id,
                run_number=scan.get('run_number'), bank=bank, sample_key=sample_key,
                sample_label=str(scan.get('sample_label') or ' · '.join(x for x in [sample_id, sample_name] if x) or 'Unassigned sample'),
                scan_label=scan_label, metadata=metadata,
                source_path=str(path) if path else None, source_sha256=checksum,
                internal_source_sha256=project.get('sha256'), collection_id=collection,
                collection_name=publication, published_sha256=project.get('published_sha256'),
                base_rwp=scan.get('rwp', result.get('final_rwp', result.get('rwp'))),
                project=project)
