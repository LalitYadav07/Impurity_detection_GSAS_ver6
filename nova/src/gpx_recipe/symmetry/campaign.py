"""Durable finite campaigns. A restart never means silently repeating a trial.

The transport owns process/job identity; this ledger owns enrollment, the finite
denominator and receipt verification. Neither component makes scientific choices.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .compiler import compile_bundle, confined, file_sha, verify_inputs,assets
from .contracts import digest, trial_id, validate_protocol,iter_trials
from ..symmetry_results import TERMINAL, summarize_portfolio


def verify_receipt(directory, expected):
    """Rehash terminal artifacts on every collection/restart, not just once."""
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    if result.get('schema')!='radar-symmetry-trial/v1' or result.get('id')!=expected['id'] or result.get('protocol_sha256')!=expected['protocol_sha256'] or result.get('status') not in TERMINAL:
        raise ValueError('Terminal receipt does not match this approved trial')
    job=json.loads((directory/'job.json').read_text())
    protocol=validate_protocol(job['protocol'])
    if digest(protocol)!=expected['protocol_sha256'] or trial_id(protocol,job['candidate_id'],job['seed_id'])!=expected['id']:
        raise ValueError('Terminal job identity changed')
    verify_inputs(protocol,directory)
    for relative,sha in result.get('files',{}).items():
        if file_sha(confined(directory,relative))!=sha:raise ValueError('A terminal artifact changed: '+relative)
    if result['status']=='valid':
        if result.get('effects_verified') is not True or result.get('inputs_verified') is not True or not result.get('files') or not result.get('audit',{}).get('passed'):
            raise ValueError('A qualified fit needs independently audited native artifacts')
    return result


class Campaign:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self._terminal_checks={}
        self.database=self.root/'campaign.sqlite'
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS protocols(hash TEXT PRIMARY KEY,body TEXT NOT NULL,source TEXT NOT NULL,metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY,protocol TEXT NOT NULL,candidate TEXT NOT NULL,seed TEXT NOT NULL,state TEXT NOT NULL,handle TEXT,receipt TEXT,error TEXT,updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,trial TEXT,state TEXT,detail TEXT,at REAL);
            ''')
            if 'directory' not in {row[1] for row in db.execute('PRAGMA table_info(trials)')}:db.execute('ALTER TABLE trials ADD COLUMN directory TEXT')

    def connect(self):
        db=sqlite3.connect(self.database,timeout=30);db.row_factory=sqlite3.Row
        return db

    @contextmanager
    def exclusive(self):
        """Serialize dispatch and enrollment across processes, without a stale lease."""
        with (self.root/'dispatcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    @staticmethod
    def event(db,uid,state,detail=None):
        db.execute('INSERT INTO events(trial,state,detail,at) VALUES(?,?,?,?)',(uid,state,detail,time.time()))

    def enroll(self,protocol,input_root,*,approved_hash,metadata=None):
        """A source revision is a new protocol; identical incoming scans deduplicate."""
        protocol=validate_protocol(protocol);sha=digest(protocol)
        if sha!=approved_hash:raise ValueError('Approval does not bind this complete protocol/source revision')
        verify_inputs(protocol,input_root)
        metadata=metadata or {}
        # Only scientific provenance belongs in this ledger, never transport keys.
        if set(metadata)-{'scan','sample','bank','condition','condition_unit','series'}:raise ValueError('Unknown source provenance fields')
        with self.exclusive(),self.connect() as db:
            existing=db.execute('SELECT body FROM protocols WHERE hash=?',(sha,)).fetchone()
            if existing:return [r['id'] for r in db.execute('SELECT id FROM trials WHERE protocol=? ORDER BY rowid',(sha,))]
            rows=[]
            for candidate,seed in iter_trials(protocol):
                    uid=trial_id(protocol,candidate['id'],seed['id']);destination=self.root/'trials'/uid
                    if destination.exists():
                        # A crash after atomic preparation but before ledger commit
                        # is recoverable only if the already sealed job is identical.
                        job=json.loads((destination/'job.json').read_text())
                        if job['id']!=uid or job['protocol_sha256']!=sha:raise ValueError('Conflicting prepared job identity')
                        verify_inputs(protocol,destination)
                    else:
                        staging=self.root/'preparing'/uid
                        if staging.exists():shutil.rmtree(staging)
                        compile_bundle(protocol,input_root,staging,candidate_id=candidate['id'],seed_id=seed['id'],approved_hash=sha)
                        destination.parent.mkdir(exist_ok=True);staging.replace(destination)
                    rows.append((uid,sha,candidate['id'],seed['id'],'prepared',time.time()))
            db.execute('INSERT INTO protocols VALUES(?,?,?,?)',(sha,json.dumps(protocol),protocol['source']['sha256'],json.dumps(metadata)))
            db.executemany('INSERT INTO trials(id,protocol,candidate,seed,state,updated) VALUES(?,?,?,?,?,?)',rows)
            for row in rows:self.event(db,row[0],'prepared')
        return [r[0] for r in rows]

    def trials(self):
        with self.connect() as db:return [dict(r) for r in db.execute('SELECT * FROM trials ORDER BY rowid')]

    def trial_directory(self,row):
        return Path(row['directory']) if row.get('directory') else self.root/'trials'/row['id']

    def adopt_completed(self,directories,*,metadata=None):
        """Index existing sealed evidence read-only; do not rerun or rewrite it."""
        metadata=metadata or {}
        if set(metadata)-{'scan','sample','bank','condition','condition_unit','series'}:raise ValueError('Unknown source provenance fields')
        with self.exclusive(),self.connect() as db:
            for directory in directories:
                directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text());p=validate_protocol(job['protocol']);sha=digest(p)
                result=verify_receipt(directory,dict(id=job['id'],protocol_sha256=sha))
                db.execute('INSERT OR IGNORE INTO protocols VALUES(?,?,?,?)',(sha,json.dumps(p),p['source']['sha256'],json.dumps(metadata)))
                db.execute('INSERT OR IGNORE INTO trials(id,protocol,candidate,seed,state,receipt,updated,directory) VALUES(?,?,?,?,?,?,?,?)',
                    (job['id'],sha,job['candidate_id'],job['seed_id'],'terminal',json.dumps(result),time.time(),str(directory)))
                self.event(db,job['id'],'terminal_evidence_imported_read_only')

    def set_state(self,uid,state,*,handle=None,receipt=None,error=None):
        with self.connect() as db:
            db.execute('UPDATE trials SET state=?,handle=COALESCE(?,handle),receipt=COALESCE(?,receipt),error=?,updated=? WHERE id=?',
                (state,json.dumps(handle) if handle else None,json.dumps(receipt) if receipt else None,error,time.time(),uid))
            self.event(db,uid,state,error)

    @staticmethod
    def _artifact_stamps(directory,paths):
        stamps=[]
        for relative in paths:
            path=confined(directory,relative);st=path.stat()
            # ctime catches in-place edits even when mtime is restored. Inode,
            # real path and device also detect replacement/symlink changes.
            stamps.append((relative,str(path),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_mode))
        return tuple(stamps)

    def collect(self,row):
        try:
            directory=self.trial_directory(row);cached=self._terminal_checks.get(row['id'])
            if row['state']=='terminal' and cached and cached['anchor']==row.get('receipt') and time.monotonic()-cached['verified_at']<300:
                if self._artifact_stamps(directory,cached['paths'])==cached['stamps']:return
            original=json.loads((directory/'result.json').read_text())
            job=json.loads((directory/'job.json').read_text());protocol=validate_protocol(job['protocol'])
            paths=sorted(set(['result.json','job.json']+[asset['path'] for asset in assets(protocol)]+list(original.get('files',{}))))
            before=self._artifact_stamps(directory,paths)
            receipt=verify_receipt(directory,dict(id=row['id'],protocol_sha256=row['protocol']))
            after=self._artifact_stamps(directory,paths)
            if before!=after or digest(original)!=digest(receipt):raise ValueError('An artifact changed during terminal verification')
            if row.get('receipt') and digest(json.loads(row['receipt']))!=digest(receipt):raise ValueError('A terminal receipt changed after collection')
            if row['state']!='terminal':self.set_state(row['id'],'terminal',receipt=receipt)
            self._terminal_checks[row['id']]=dict(anchor=json.dumps(receipt),paths=paths,stamps=after,verified_at=time.monotonic())
        except Exception as exc:
            self._terminal_checks.pop(row['id'],None)
            self.set_state(row['id'],'needs_recovery',error=str(exc))

    def tick(self,transport,*,slots=1):
        """One nonblocking scheduler pass. Persist intent BEFORE submitting.

        recover(key) must return absent only when absence is definitive. Unknown
        acknowledgments stop that branch; they never cause duplicate submission.
        """
        if type(slots) is not int or not 1<=slots<=16:raise ValueError('Invalid declared concurrency')
        with self.exclusive():
            for row in self.trials():
                directory=self.trial_directory(row)
                if row['state']=='terminal':
                    self.collect(row);continue
                if row['state']=='cancelled':continue
                if (directory/'result.json').is_file():self.collect(row);continue
                if row['state']=='needs_recovery':continue
                if row['state'] in {'submitting','running','cancelling'}:
                    try:
                        state=transport.recover(row['id'],directory) if row['state']=='submitting' else transport.poll(json.loads(row['handle']),directory)
                        status=state['state']
                        if status=='running':self.set_state(row['id'],'cancelling' if row['state']=='cancelling' else 'running',handle=state['handle'])
                        elif status=='finished':self.collect(row)
                        elif status=='cancelled':self.set_state(row['id'],'cancelled')
                        elif status=='absent' and row['state']=='submitting':self.set_state(row['id'],'prepared')
                        else:self.set_state(row['id'],'needs_recovery',error=state.get('reason','Transport state is ambiguous'))
                    except Exception as exc:self.set_state(row['id'],'needs_recovery',error=str(exc))
            rows=self.trials();available=slots-sum(r['state'] in {'running','submitting','cancelling'} for r in rows)
            for row in [r for r in rows if r['state']=='prepared'][:max(0,available)]:
                self.set_state(row['id'],'submitting')
                try:
                    handle=transport.submit(row['id'],self.root/'trials'/row['id'])
                    self.set_state(row['id'],'running',handle=handle)
                except Exception as exc:
                    # Leave intent durable: next pass recovers by dispatch key.
                    with self.connect() as db:self.event(db,row['id'],'submission_acknowledgment_missing',str(exc))
        return self.trials()

    def verify_continuation(self,implementation_sha256):
        """Resume sealed work, without silently upgrading its implementation."""
        pending=[r for r in self.trials() if r['state'] not in {'terminal','cancelled'}]
        if not pending:raise ValueError('This campaign has no pending trials to continue')
        keys={r['protocol'] for r in pending}
        with self.connect() as db:
            protocols=[json.loads(r['body']) for r in db.execute('SELECT hash,body FROM protocols') if r['hash'] in keys]
        if any(p['implementation_sha256']!=implementation_sha256 for p in protocols):
            raise ValueError('Pending trials use an older scientific implementation; review a new plan instead of silently upgrading them')
        if all(r['state']=='needs_recovery' and not (self.trial_directory(r)/'result.json').is_file() for r in pending):
            raise ValueError('These trials need explicit transport recovery; no duplicate job was submitted')
        return [r['id'] for r in pending]

    def verify_sealed_continuation(self):
        """Resume the original bundled worker, never substitute current code.

        A dispatcher can be fixed/upgraded independently of the scientific
        scripts already approved. Every pending bundle must still match its
        full original package hash and the known v1 trusted entry point.
        The native worker also rechecks its pinned GSAS-II runtime at launch.
        """
        pending=[r for r in self.trials() if r['state'] not in {'terminal','cancelled'}]
        if not pending:raise ValueError('This campaign has no pending trials to continue')
        if all(r['state']=='needs_recovery' and not (self.trial_directory(r)/'result.json').is_file() for r in pending):
            raise ValueError('These trials need explicit transport recovery; no duplicate job was submitted')
        entrypoint='from pathlib import Path\nimport sys\nroot=Path(__file__).resolve().parent\nsys.path[:0]=[str(root/"runtime"),str(root/"runtime/scripts")]\nfrom gpx_recipe.symmetry.worker import main\nif __name__ == "__main__":\n    main(root)\n'
        for row in pending:
            directory=self.trial_directory(row)
            job=json.loads(confined(directory,'job.json').read_text());protocol=validate_protocol(job['protocol'])
            if job.get('schema')!='radar-symmetry-job/v1' or digest(protocol)!=row['protocol'] or job.get('protocol_sha256')!=row['protocol'] or job.get('id')!=row['id'] or trial_id(protocol,job['candidate_id'],job['seed_id'])!=row['id']:
                raise ValueError('A sealed job identity changed; no replacement was submitted')
            if job.get('package_sha256')!=protocol['implementation_sha256']:
                raise ValueError('A sealed scientific implementation differs from its approved protocol')
            root=confined(directory,'runtime/gpx_recipe')
            files={str(p.relative_to(root)):file_sha(confined(root,str(p.relative_to(root))))
                   for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'}}
            if digest(files)!=protocol['implementation_sha256']:
                raise ValueError('The original sealed scientific package changed')
            for name in ['gsas_magnetic_preflight.py','gsas_isodistort_import.py','gsas_cif_import.py']:
                if file_sha(confined(directory,'runtime/scripts/'+name))!=files.get('symmetry/adapters/'+name):
                    raise ValueError('A sealed native adapter differs from its approved package')
            if confined(directory,'worker.py').read_text()!=entrypoint:
                raise ValueError('Unknown or modified sealed worker entry point')
            verify_inputs(protocol,directory)
        return [row['id'] for row in pending]

    def cancel(self,uid,transport):
        with self.exclusive():
            row=next(r for r in self.trials() if r['id']==uid)
            if row['state'] in {'terminal','cancelled'}:return
            directory=self.trial_directory(row)
            if row['state']=='prepared':self.set_state(uid,'cancelled');return
            state=transport.recover(uid,directory)
            if state['state']=='finished':self.collect(row)
            elif state['state']=='running':
                confirmed=transport.cancel(state['handle'],directory)
                self.set_state(uid,'cancelled' if confirmed else 'cancelling',handle=state['handle'])
            elif state['state']=='cancelled':self.set_state(uid,'cancelled')
            elif state['state']=='absent':self.set_state(uid,'cancelled')
            else:self.set_state(uid,'needs_recovery',error='Cancellation could not be confirmed')

    def summaries(self):
        rows=self.trials();summaries=[]
        with self.connect() as db:protocols=[dict(r) for r in db.execute('SELECT * FROM protocols ORDER BY rowid')]
        for row in protocols:
            trials=[r for r in rows if r['protocol']==row['hash']];p=json.loads(row['body'])
            receipts=[json.loads(r['receipt']) for r in trials if r['state']=='terminal']
            expected=[trial_id(p,c['id'],s['id']) for c,s in iter_trials(p)]
            summaries.append(dict(protocol=row['hash'],metadata=json.loads(row['metadata']),
                **summarize_portfolio(expected,receipts,fit_band_percent=p['policy']['fit_band_percent'])))
        return summaries
