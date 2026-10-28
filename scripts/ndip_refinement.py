#!/usr/bin/env python3
"""Galaxy transport wrapper for the credential-free refinement worker."""
import argparse
import json
from pathlib import Path
import zipfile

from gpx_recipe.worker import work


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','preview','run']);parser.add_argument('source');parser.add_argument('request');parser.add_argument('directory');parser.add_argument('archive');args=parser.parse_args()
    request=json.loads(Path(args.request).read_text());directory=Path(args.directory).resolve()
    work(args.action,args.source,directory,plan=request.get('plan'),approval=request.get('approval'),expected_source_sha256=request.get('expected_source_sha256'))
    with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():archive.write(path,path.name)


if __name__=='__main__':main()
