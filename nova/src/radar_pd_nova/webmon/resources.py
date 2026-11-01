"""Cooperating local worker admission; container CPU/RAM limits still apply."""
import asyncio,fcntl,time
from pathlib import Path
class Lease:
    def __init__(self,stream):self.stream=stream
    def release(self):
        if self.stream:
            fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close();self.stream=None
async def acquire(root,capacity=1,timeout=600):
    if not isinstance(capacity,int) or not 1<=capacity<=8:raise ValueError('Invalid native job capacity')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+timeout
    while True:
        for i in range(capacity):
            f=(root/f'{i}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return Lease(f)
            except BlockingIOError:f.close()
        if time.monotonic()>=deadline:raise TimeoutError('Native capacity unavailable within deadline')
        await asyncio.sleep(.1)
