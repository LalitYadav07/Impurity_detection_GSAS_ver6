"""Read-only source discovery and sealed local experiment context.

A deployment supplies a context manifest after checkpoint review; it is never
accepted as an arbitrary browser path. Local acceptance uses the same adapter
against a bounded mirror. Source files are never modified.
"""
import csv,hashlib,json,re,copy
from pathlib import Path
from .snapshot import stable_read,parse_live_ascii,SnapshotArchive
from .contracts import validate_request
from .registry import CheckpointRegistry
from .doctor import validate_config

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class WebmonSession:
    def __init__(self,manifest):
        self.manifest_path=None if isinstance(manifest,dict) else Path(manifest).resolve()
        self.config=copy.deepcopy(manifest) if isinstance(manifest,dict) else json.loads(self.manifest_path.read_text())
        # Lightweight discovery-only fixtures need no launch settings.
        if 'worker_command' in self.config:validate_config(self.config)
        self.root=Path(self.config['facility_root']).resolve();self.archive=SnapshotArchive(self.config['archive'])
        self.active=None;self.selected_checkpoints={};self.extra_checkpoints={}
        self.registry=CheckpointRegistry(self.config['checkpoint_registry']) if self.config.get('checkpoint_registry') else None
        if self.registry:self.extra_checkpoints=self.registry.read()
    def scope(self,ipts):
        ipts=str(ipts).removeprefix('IPTS-')
        if not ipts.isdigit():raise ValueError('A numeric IPTS is required')
        path=(self.root/'PG3'/('IPTS-'+ipts)).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Source escapes facility root')
        return path
    def runs(self,ipts):
        scope=self.scope(ipts);summary=scope/'shared/autoreduce'/f'PG3_IPTS-{str(ipts).removeprefix("IPTS-")}_runsummary.csv'
        metadata={}
        if summary.exists():
            with summary.open() as f:metadata={r['RUN']:r for r in csv.DictReader(f)}
        rows=[]
        for p in sorted((scope/'shared/livereduce').glob('PG3_*.dat'),reverse=True):
            m=re.fullmatch(r'PG3_(\d+)\.dat',p.name)
            if not m or not p.resolve().is_relative_to(scope):continue
            run=m[1];r=metadata.get(run,{})
            rows.append(dict(run=run,path=str(p),sample_id=r.get('ITEMS-ID',''),sample=r.get('SAMPLE','Unknown'),temperature=r.get('TEMP','Unknown'),wavelength=r.get('WAVELENGTH','Unknown'),final_duration=r.get('DURATION','Unknown'),title=r.get('TITLE',''),value=run,
                label=f"{run} · {r.get('SAMPLE','Unknown')} · {r.get('TEMP','?')} K · {r.get('WAVELENGTH','?')} Å"))
        return sorted(rows,key=lambda r:int(r['run']),reverse=True)
    def capture(self,ipts,run):
        row=next((r for r in self.runs(ipts) if r['run']==str(run)),None)
        if row is None:raise ValueError('Run is not in the selected IPTS')
        receipt=self.archive.capture(row['path'],ipts=str(ipts).removeprefix('IPTS-'),run=run,epoch='stored-live',metadata=dict(sample_id=row['sample_id'],cumulative_exposure=None))
        return row,receipt,parse_live_ascii(self.archive.read(receipt))
    def checkpoints(self,ipts,run):
        key=f"IPTS-{str(ipts).removeprefix('IPTS-')}/{run}"
        values=[*self.config.get('checkpoint_options',{}).get(key,[]),*self.extra_checkpoints.get(key,[])]
        return list({c['id']:c for c in values}.values())
    def select_checkpoint(self,ipts,run,uid):
        key=f"IPTS-{str(ipts).removeprefix('IPTS-')}/{run}"
        choice=next((c for c in self.checkpoints(ipts,run) if c['id']==uid),None)
        if choice is None:raise ValueError('Choose a checkpoint from this experiment and target run')
        if digest(choice['source_path'])!=choice['source_sha256']:raise ValueError('The selected RADAR-PD checkpoint changed')
        self.selected_checkpoints[key]=uid
        return choice
    def selected_checkpoint(self,ipts,run):
        key=f"IPTS-{str(ipts).removeprefix('IPTS-')}/{run}"
        choices=self.checkpoints(ipts,run)
        uid=self.selected_checkpoints.get(key,choices[0]['id'] if choices else '')
        return self.select_checkpoint(ipts,run,uid) if choices else None
    def register_checkpoint(self,ipts,run,choice):
        key=f"IPTS-{str(ipts).removeprefix('IPTS-')}/{run}"
        if choice.get('parent_checkpoint') not in {c['id'] for c in self.checkpoints(ipts,run)}:raise ValueError('Refined checkpoint has no parent in this run')
        for p,h in [('source_path','source_sha256'),('view_path','view_sha256')]:
            if digest(choice[p])!=choice[h]:raise ValueError('Prepared checkpoint artifact changed')
        old=self.selected_checkpoints.get(key)
        entries=self.extra_checkpoints.setdefault(key,[])
        prior_entries=list(entries)
        entries[:]=[c for c in entries if c['id']!=choice['id']];entries.append(choice)
        try:
            self.select_checkpoint(ipts,run,choice['id']);self.binding(ipts,run)
            if self.registry:self.registry.add(key,choice)
        except Exception:
            entries[:]=prior_entries
            if old is None:self.selected_checkpoints.pop(key,None)
            else:self.selected_checkpoints[key]=old
            raise
    def binding(self,ipts,run):
        key=f"IPTS-{str(ipts).removeprefix('IPTS-')}/{run}"
        choice=self.selected_checkpoint(ipts,run)
        b=choice.get('binding') if choice else self.config.get('bindings',{}).get(key)
        if b is None:raise ValueError('No reviewed nuclear checkpoint and native preparation are bound to this run/configuration')
        if choice and (choice['source_path']!=b['reference_gpx'] or choice['source_sha256']!=b['reference_sha256']):raise ValueError('Checkpoint and reference binding disagree')
        for path_key,hash_key in [('reference_gpx','reference_sha256'),('source_gpx','source_sha256'),('source_dat','source_dat_sha256')]:
            if digest(b[path_key])!=b[hash_key]:raise ValueError('Reviewed input changed: '+path_key)
        row=next((r for r in self.runs(ipts) if r['run']==str(run)),None)
        if row is None or Path(row['path']).resolve()!=Path(b['source_dat']).resolve():raise ValueError('Snapshot binding mismatch')
        return b
    def request(self,ipts,run,*,k,ion,lower,upper,accepted):
        if not accepted:raise ValueError('Review and accept the nuclear revision first')
        b=self.binding(ipts,run)
        if list(k)!=b['k'] or ion!=b['ion']:raise ValueError('Changed k/ion requires a new native catalogue and preparation')
        if not 0<=lower<upper<=b['maximum_moment']:raise ValueError('Invalid ordered-moment range')
        req=json.loads(Path(b['request']).read_text());req.update(lower=lower,upper=upper)
        validate_request(req,b)
        return copy.deepcopy(b),copy.deepcopy(req)

    def restore_binding(self,ipts,run,req,*,allow_archived=False):
        if not req.get('observation'):return self.binding(ipts,run)
        from .stream import scope_of
        choice=self.selected_checkpoint(ipts,run)
        if not choice:raise ValueError('Select the reviewed nuclear checkpoint first')
        b=copy.deepcopy(choice['binding'])
        if choice['source_sha256']!=b['reference_sha256']:raise ValueError('Reference revision mismatch')
        for p,h in [('reference_gpx','reference_sha256'),('source_gpx','source_sha256')]:
            if digest(b[p])!=b[h]:raise ValueError('Reviewed native input changed')
        if not str(req.get('source_revision','')).startswith(f'PG3_{run}:'):
            raise ValueError('Saved stream result belongs to another run')
        if not allow_archived:
            row,receipt,_=self.capture(ipts,run)
            if req.get('stream_scope')!=scope_of(row) or req.get('observation_source_sha256')!=receipt['content_sha256']:
                raise ValueError('Saved stream result is not the current compatible snapshot')
        # Reviewing a sealed old result does not require its mutable facility
        # source to remain current. Its own archived bytes are still verified.
        b.update(source_dat=req['observation_source'],source_dat_sha256=req['observation_source_sha256'],observation_sha256=req['observation_sha256'])
        validate_request(req,b);return b
