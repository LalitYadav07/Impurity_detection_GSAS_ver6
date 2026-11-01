"""Credential-free transport envelope around the shared provisional scan worker."""
from __future__ import annotations
import argparse,json,os,zipfile,shutil
from pathlib import Path
from .packet import unpack_packet,unpack_cache,pack_cache,canonical
from .contracts import file_hash


def run_packet(packet,directory,result_archive,cache_archive,*,cache=None):
    root=Path(directory).resolve()
    request,packet_id=unpack_packet(packet,root)
    if cache:unpack_cache(cache,root)
    # The native worker uses absolute input paths in this private process, while
    # the transfer manifest keeps their portable identity and immutable bytes.
    if request.get('action')=='prepare':
        from .prepare_source import prepare
        from .validated_backend import corrected_backend
        with corrected_backend():prepared=prepare(request,root)
        result=dict(status='prepared',rows=[],source_revision=request['source_revision'],reference_revision=request['reference_revision'],preparation=prepared,forecast_enabled=False,exclusion_authorized=False)
        (root/'result.json').write_bytes(canonical(result))
    else:
        from .worker import run
        run(root/'request.json')
    native=json.loads((root/'result.json').read_text())
    native_path=root/'native-result.json';native_path.write_bytes(canonical(native))
    output=dict(native)
    for row in output['rows']:
        for key in ('posterior','fitted_gpx'):
            if row.get(key):
                path=Path(row[key]).resolve()
                if not path.is_relative_to(root):raise ValueError('Worker result escapes its job directory')
                row[key]=str(path.relative_to(root))
    projects=root/'verified_projects';projects.mkdir(exist_ok=True)
    for row in output['rows']:
        if row.get('fitted_gpx') and row.get('fitted_gpx_sha256'):
            path=root/row['fitted_gpx']
            if file_hash(path)!=row['fitted_gpx_sha256']:raise ValueError('Inferred GPX changed before publication')
            name=row['id'].replace(':','_')
            shutil.copyfile(path,projects/(name+'.gpx'));row['published_project']=name
    if output.get('preparation'):
        path=root/output['preparation']['source_gpx']
        if file_hash(path)!=output['preparation']['source_sha256']:raise ValueError('Prepared GPX changed before publication')
        shutil.copyfile(path,projects/'prepared.gpx');output['preparation']['published_project']='prepared'
    output.update(packet_id=packet_id,native_result_sha256=file_hash(native_path))
    (root/'result.json').write_bytes(canonical(output))
    files={str(path.relative_to(root)):file_hash(path) for path in root.rglob('*') if path.is_file()}
    (root/'transport.json').write_bytes(canonical({'schema':'radar-live-result/v1','packet_id':packet_id,'files':files}))
    with zipfile.ZipFile(result_archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():archive.write(path,str(path.relative_to(root)))
    pack_cache(root,cache_archive)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet');parser.add_argument('directory');parser.add_argument('result_archive');parser.add_argument('cache_archive');parser.add_argument('--cache')
    args=parser.parse_args()
    run_packet(args.packet,args.directory,args.result_archive,args.cache_archive,cache=args.cache)


if __name__=='__main__':main()
