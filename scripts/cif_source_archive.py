"""Read-only ZIP intake guard before the Galaxy wrapper flattens CIF sources."""
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import sys
import zipfile


def validate_cif_member_names(path, *, max_members=10000) -> int:
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > max_members:
            raise ValueError(f'CIF archive has more than {max_members} members; split this source bundle')
        found={}
        for member in archive.infolist():
            if member.is_dir():continue
            name=PurePosixPath(member.filename.replace('\\','/')).name
            if not name.lower().endswith('.cif'):continue
            key=name.casefold()
            if key in found:
                raise ValueError(f'CIF basename collision: {found[key]!r} and {member.filename!r}. Give the files distinct names; no source files were copied.')
            found[key]=member.filename
        if not found:raise ValueError('The source archive contains no CIF files')
        return len(found)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive')
    args=parser.parse_args(argv)
    try:count=validate_cif_member_names(args.archive)
    except (ValueError,zipfile.BadZipFile) as exc:
        print(f'CIF archive intake rejected: {exc}',file=sys.stderr);return 2
    print(f'CIF archive intake: {count} distinct source filenames verified')
    return 0


if __name__=='__main__':raise SystemExit(main())
