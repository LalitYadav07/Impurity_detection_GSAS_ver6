"""Durable state under the existing acquisition owner; no second polling loop."""
from pathlib import Path
from dataclasses import asdict
import json
import threading
from ..execution import write_json
from ..symmetry.compiler import package_signature
from .native import sha
from .service import SearchSettings, fit_snapshot
from .observations import array_digest
from .reduced import ReducedObservation


class PublicationUnavailable(ValueError):
    """A saved result needs repair/re-export before it can be reused."""


class StructuralSession:
    def __init__(self, prepared, directory, *, settings=None, runner=fit_snapshot):
        self.prepared=Path(prepared).resolve();self.root=Path(directory).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.settings=settings or SearchSettings();self.runner=runner
        self.identity=dict(preparation=sha(self.prepared/'prepared.json'),implementation=package_signature(),settings=asdict(self.settings))
        self.lock=threading.Lock();self.previous=None
        self.record=None
        path=self.root/'latest.json'
        if path.exists():
            record=json.loads(path.read_text())
            if record['identity']!=self.identity:
                raise ValueError('Structural preparation or implementation changed; create a new session')
            self._verify_publication(record)
            self.record=record

    def _local(self,relative):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise PublicationUnavailable('Published structural artifact path is not confined to this session')
        resolved=(self.root/path).resolve()
        if not resolved.is_relative_to(self.root):
            raise PublicationUnavailable('Published structural artifact escapes this session')
        return resolved

    def _verify_publication(self,record):
        try:
            result=self._local(record['result'])
            if sha(result)!=record['result_sha256']:
                raise PublicationUnavailable('Published structural result changed; re-export is required')
            receipt=json.loads(result.read_text())
            artifacts=record.get('artifacts',{})
            if set(artifacts)!={'frozen','continuation'}:
                raise PublicationUnavailable('Published structural artifact manifest is missing; re-export is required')
            resolved={}
            for role,item in artifacts.items():
                artifact=self._local(item['path'])
                if (not artifact.is_relative_to(result.parent) or sha(artifact)!=item['sha256']
                        or receipt.get(role,{}).get('sha256')!=item['sha256']):
                    raise PublicationUnavailable('Published structural GPX changed; re-export is required')
                resolved[role]=str(artifact)
            return receipt,resolved
        except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise PublicationUnavailable('Published structural result or GPX is unavailable; re-export is required') from exc

    def update(self, observation):
        if not isinstance(observation,ReducedObservation):
            raise TypeError('This session requires supplied-variance reduced observations')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('One acquisition owner must serialize structural updates')
        try:
            if observation.protocol_sha256!=self.identity['preparation']:
                raise ValueError('Observation does not bind this prepared structural context')
            if sha(self.prepared/'prepared.json')!=self.identity['preparation'] or package_signature()!=self.identity['implementation']:
                raise ValueError('Prepared native context changed')
            if self.previous is not None:
                observation.require_continuation(self.previous)
            if self.record:
                old=self.record
                previous,resolved=self._verify_publication(old)
                if observation.acquisition!=old['acquisition'] or observation.protocol_sha256!=old['protocol_sha256']:
                    raise ValueError('Different acquisition or observation protocol requires a new session')
                if array_digest(observation.x)!=old['grid_sha256']:
                    raise ValueError('Observation grid changed; prepare a new session')
                if ((observation.exposure is None)!=(old['exposure'] is None) or
                        (observation.exposure is not None and observation.exposure<old['exposure'])):
                    raise ValueError('Cumulative exposure changed semantics or reset')
                if observation.revision<old['revision']:
                    raise ValueError('Stale structural observation')
                if observation.revision==old['revision']:
                    if observation.observation_sha256!=old['observation_sha256']:
                        raise ValueError('Observation changed without a revision')
                    return dict(state='duplicate',new_fit=False,result=str(self.root/old['result']),artifacts=resolved)
            warm=[] if self.record is None else [previous['parameters']]
            import uuid
            path=self.root/('revision-'+str(observation.revision)+'-'+uuid.uuid4().hex)
            # A failure leaves the previous published result and warm start intact.
            result=self.runner(self.prepared,path,settings=self.settings,observation=observation,warm_starts=warm)
            result_path=path/'fit.json'
            # Seal both required GPXs. A fit receipt alone cannot establish
            # that its scientific artifacts survived or can be continued.
            artifacts={}
            for role in ('frozen','continuation'):
                item=result.get(role,{})
                try:
                    artifact=Path(item['path']).resolve()
                    if not artifact.is_relative_to(path) or sha(artifact)!=item['sha256']:
                        raise PublicationUnavailable('New structural GPX is unconfined or changed')
                    artifacts[role]=dict(path=str(artifact.relative_to(self.root)),sha256=item['sha256'])
                except (KeyError,OSError,TypeError) as exc:
                    raise PublicationUnavailable('New structural GPX is unavailable; previous result retained') from exc
            record=dict(identity=self.identity,acquisition=observation.acquisition,
                protocol_sha256=observation.protocol_sha256,revision=observation.revision,
                observation_sha256=observation.observation_sha256,
                grid_sha256=array_digest(observation.x),
                exposure=observation.exposure,result=str(result_path.relative_to(self.root)),result_sha256=sha(result_path),
                artifacts=artifacts)
            saved,_=self._verify_publication(record)
            if saved!=result:
                raise PublicationUnavailable('New structural receipt differs from the returned result')
            write_json(self.root/'latest.json',record)
            self.record=record;self.previous=observation
            return result
        finally:self.lock.release()
