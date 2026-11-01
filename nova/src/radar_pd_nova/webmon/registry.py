"""Atomic, cooperating-writer registry of derived checkpoint revisions."""
import fcntl,json,os,tempfile
from pathlib import Path

class CheckpointRegistry:
    def __init__(self,path):self.path=Path(path)
    def read(self):
        if not self.path.exists():return {}
        value=json.loads(self.path.read_text())
        if value.get('schema')!='webmon-checkpoints/v1' or not isinstance(value.get('scopes'),dict):raise ValueError('Unsupported checkpoint registry')
        return value['scopes']
    def add(self,scope,choice):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            values=self.read();rows=values.setdefault(scope,[])
            previous=next((v for v in rows if v['id']==choice['id']),None)
            if previous and previous!=choice:raise ValueError('An immutable checkpoint ID already has different preparation')
            if not previous:rows.append(choice)
            fd,name=tempfile.mkstemp(prefix='.registry-',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w') as stream:
                    json.dump(dict(schema='webmon-checkpoints/v1',scopes=values),stream,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
                os.replace(name,self.path)
            finally:
                if os.path.exists(name):os.unlink(name)
