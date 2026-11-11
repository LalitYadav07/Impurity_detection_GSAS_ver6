"""Build-time correction of periodic site comparisons in the pinned GSAS-II.

This changes equivalent-position classification, not coordinates, operators,
spin characters or the existing absolute position tolerance. It must be
qualified as a new native runtime before any scientific protocol can use it.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from .kernel_patch import function_source, replace_once, sha256

VERSION='radar-gsas-periodic-site-comparison/1'
ORIGINAL_FILE_SHA256='3f375b5358a7e92e3de76aaf4072d7e1c831bbfa38737f06cde2a2e01a9425f4'
DESKTOP_FILE_SHA256='f785948511748573047cf8a861848d3f23380002d91411d26a0e61a92fc72c0b'
ORIGINAL_FUNCTION_SHA256='5db32ccb2096afa3fe491c8bc33ed0e37c10c2bc4b1442a640ea9ffe89c9464e'


def corrected_source(source):
    source_sha=sha256(source)
    if source_sha not in {ORIGINAL_FILE_SHA256,DESKTOP_FILE_SHA256}:
        raise ValueError('Unreviewed GSAS-II space-group source')
    original=function_source(source,'GenAtom')
    if sha256(original)!=ORIGINAL_FUNCTION_SHA256:
        raise ValueError('The pinned equivalent-position function changed')
    fixed=replace_once(original,'np.allclose(newX,X,atol=2.e-4)',
        'np.allclose((newX-X+.5)%1.-.5,0.,atol=2.e-4,rtol=0.)')
    fixed=replace_once(fixed,'np.allclose(newX%1.,oldX%1.,atol=2.e-4)',
        'np.allclose((newX-oldX+.5)%1.-.5,0.,atol=2.e-4,rtol=0.)')
    updated=replace_once(source,original,fixed)
    ast.parse(updated)
    return updated,dict(version=VERSION,original_file_sha256=source_sha,
        corrected_file_sha256=sha256(updated),original_function_sha256=ORIGINAL_FUNCTION_SHA256,
        corrected_function_sha256=sha256(fixed),absolute_position_tolerance=2e-4,
        coordinates_changed=False,scope='Periodic equivalent-position and site-stabilizer comparisons')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',required=True)
    parser.parse_args()
    spec=importlib.util.find_spec('GSASII.GSASIIspc')
    if not spec or not spec.origin:
        raise RuntimeError('The pinned GSAS-II space-group module is unavailable')
    path=Path(spec.origin)
    updated,record=corrected_source(path.read_text())
    path.write_text(updated)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['corrected_file_sha256']:
        raise RuntimeError('The written equivalent-position source failed verification')
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':main()
