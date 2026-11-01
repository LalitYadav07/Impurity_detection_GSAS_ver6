"""Process-local guard for GSAS-II's size-only GPX offset cache.

No installed GSAS-II files are modified. Both path changes and same-size rewrites
must invalidate offsets before any record is sought. Explicit read=True still
uses the native reader's complete read behavior.
"""
from pathlib import Path


def install_fresh_gpx_index(module=None):
    if module is None:
        from GSASII import GSASIIstrIO as module
    if getattr(module.IndexGPX, '_virtual_powder_fresh_index', False):
        return
    original = module.IndexGPX
    previous = None

    def indexed(filename, read=False):
        nonlocal previous
        path = Path(filename).resolve()
        stat = path.stat()
        stamp = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stamp != previous:
            module.gpxSize = -1
        result = original(filename, read=read)
        previous = stamp
        return result

    indexed._virtual_powder_fresh_index = True
    module.IndexGPX = indexed
