"""Bounded data-only packets for the native Galaxy model worker."""
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_BYTES=512*1024*1024
DATA_SUFFIXES={'.json','.gpx','.cif','.npz','.log','.txt','.lst'}


def hydrate_source(inputs,source,hashes,expected):
    """Reuse the separately supplied Galaxy GPX only at declared matching paths.

    Inputs still have the identical expanded layout and hashes expected by the
    existing native verifier. This avoids uploading the GPX again inside a ZIP.
    """
    import re,shutil
    from .compiler import file_sha,confined
    inputs=Path(inputs).resolve();source=Path(source)
    if not isinstance(hashes,dict) or len(hashes)>10000 or file_sha(source)!=expected:
        raise ValueError('Invalid captured source manifest')
    copies=[]
    for name,checksum in hashes.items():
        if not isinstance(name,str) or not isinstance(checksum,str) or not re.fullmatch('[a-f0-9]{64}',checksum):
            raise ValueError('Invalid captured input hash')
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or str(relative)!=name:
            raise ValueError('Captured source path is not confined')
        target=confined(inputs,name,exists=False)
        if not target.exists() and relative.suffix.lower()=='.gpx' and checksum==expected:copies.append(target)
    existing=[p for p in inputs.rglob('*') if p.is_file()]
    if len(existing)+len(copies)>10000 or sum(p.stat().st_size for p in existing)+len(copies)*source.stat().st_size>MAX_BYTES:
        raise ValueError('Hydrated model packet exceeds its expanded-size bound')
    for target in copies:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if file_sha(target)!=expected:raise ValueError('Galaxy source changed while capturing the model input')


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
