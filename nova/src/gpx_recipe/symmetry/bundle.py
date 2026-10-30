"""Bounded data-only packets for the native Galaxy model worker."""
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_BYTES=512*1024*1024
DATA_SUFFIXES={'.json','.gpx','.cif','.npz','.log','.txt','.lst'}


def unpack(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();seen=set();total=0
        if len(entries)>10000:raise ValueError('Model packet contains too many files')
        for entry in entries:
            name=PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or stat.S_ISLNK(entry.external_attr>>16):
                raise ValueError('Model packet path is not confined')
            if entry.is_dir():continue
            if str(name).casefold() in seen or name.suffix.lower() not in DATA_SUFFIXES:
                raise ValueError('Model packet contains a duplicate or executable/unsupported file')
            seen.add(str(name).casefold());total+=entry.file_size
            if total>MAX_BYTES:raise ValueError('Model packet exceeds its expanded-size bound')
            path=destination.joinpath(*name.parts)
            if not path.resolve().is_relative_to(destination):raise ValueError('Model packet escaped its root')
            path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entry) as source,path.open('xb') as target:
                import shutil
                shutil.copyfileobj(source,target,1024*1024)
    return destination


def pack(root,archive,paths):
    root=Path(root).resolve()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for relative in sorted(set(paths)):
            path=(root/relative).resolve()
            if not path.is_file() or not path.is_relative_to(root) or path.suffix.lower() not in DATA_SUFFIXES:
                raise ValueError('Only confined scientific data belongs in a model packet')
            z.write(path,relative)
