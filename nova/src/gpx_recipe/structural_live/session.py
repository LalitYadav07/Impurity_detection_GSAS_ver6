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
            result=self.root/record['result']
            if not result.resolve().is_relative_to(self.root) or sha(result)!=record['result_sha256']:
                raise ValueError('Published structural result changed')
            self.record=record

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
                    return dict(state='duplicate',new_fit=False,result=str(self.root/old['result']))
            warm=[] if self.record is None else [json.loads((self.root/self.record['result']).read_text())['parameters']]
            import uuid
            path=self.root/('revision-'+str(observation.revision)+'-'+uuid.uuid4().hex)
            # A failure leaves the previous published result and warm start intact.
            result=self.runner(self.prepared,path,settings=self.settings,observation=observation,warm_starts=warm)
            result_path=path/'fit.json'
            record=dict(identity=self.identity,acquisition=observation.acquisition,
                protocol_sha256=observation.protocol_sha256,revision=observation.revision,
                observation_sha256=observation.observation_sha256,
                grid_sha256=array_digest(observation.x),
                exposure=observation.exposure,result=str(result_path.relative_to(self.root)),result_sha256=sha(result_path))
            write_json(self.root/'latest.json',record)
            self.record=record;self.previous=observation
            return result
        finally:self.lock.release()
