"""Observation-bound structural ranking through the shared native GPX engine.

This worker uses the reviewed nuclear transfer and exact current snapshot.
It fits no distortion, changes no calibration and excludes no native model.
Native scratch GPXs and experimental arrays are not exported as preparation.
"""
import json
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..live.nuclear_transfer import prepare_nuclear_snapshot
from ..live.snapshot import parse_live_ascii,stable_read,uniform_density_to_tof
from ..preparation.structural_ranking_native import rank_gpx
from ..preparation.structural_ranking_result import validate
from ..symmetry.compiler import file_sha,package_signature,verify_inputs
from ..symmetry.contracts import digest
from .acquisition import _mapping
from .job import captured_context,transfer_request


def rank_transferred(payload,context,transfer_root,transfer,root):
    """Consume the already-required transfer without constructing it again.

    The native caller owns a fresh output directory and has checked context.
    No child adapter may remain active across this parent-ranking operation.
    """
    job,protocol,_,inputs,source,raw,anchor=context
    began=time.monotonic();root=Path(root);transfer_root=Path(transfer_root)
    if not root.is_dir() or any((root/name).exists() for name in ('ranking','ranking-parent.gpx','result.json')):
        raise ValueError('Ranking requires a fresh owned output directory')
    transferred=transfer_root/transfer['source_gpx'];transferred_sha=file_sha(transferred)
    write_json(root/'transfer.json',transfer)
    before=parse_live_ascii(anchor);current=parse_live_ascii(raw)
    if current.spectrum_marker!=before.spectrum_marker or not np.array_equal(current.x,before.x):
        raise ValueError('Structural ranking snapshot changed its reviewed spectrum or grid')
    mapping=payload['mapping']
    transformed=uniform_density_to_tof(current,export_difc=mapping['export_geometry']['microseconds_per_angstrom'])
    from .native import modules
    G,*_=modules();path=root/'ranking-parent.gpx'
    project=G.G2Project(gpxfile=str(transferred),newgpx=str(path));histogram=project.histogram(protocol['histogram'])
    if not np.array_equal(np.asarray(histogram.data['data'][1][0]),transformed['x']):
        raise ValueError('The native ranking context changed its reviewed producer grid')
    histogram.data['data'][1]=[transformed['x'].copy(),transformed['y'].copy(),transformed['weights'].copy(),
        *[np.zeros_like(current.x) for _ in range(3)]]
    project.data.setdefault('Covariance',{})['data']={};project.save()
    observation=dict(snapshot_sha256=payload['receipt']['content_sha256'],calculation_source_sha256=file_sha(path),
        acquisition_sha256=digest([_mapping(mapping),payload['source_scope']]),revision=payload['revision'])
    ranked=rank_gpx(inputs,protocol,root/'ranking',source_override=path,observation=observation)
    preparation=payload.get('prepared_scattering')
    validate(ranked,protocol,snapshot=preparation['snapshot_sha256'] if preparation else None,observation=observation)
    result=dict(schema='radar-structural-snapshot-ranking-result/v1',kind='structural_ranking',
        protocol_sha256=job['protocol_sha256'],implementation_sha256=package_signature(),source_sha256=protocol['source']['sha256'],
        observation=observation,ranking=ranked,ranking_sha256=file_sha(root/'ranking/ranking.json'),
        output=str(root),artifacts=['ranking/ranking.json','transfer.json','result.json'],
        seconds=time.monotonic()-began,candidate_pruning_authorized=False,scientific_result=False)
    write_json(root/'result.json',result)
    if (file_sha(source)!=protocol['source']['sha256'] or stable_read(payload['snapshot'])!=raw
            or stable_read(payload['anchor_snapshot'])!=anchor or file_sha(path)!=observation['calculation_source_sha256']
            or file_sha(transferred)!=transferred_sha):
        raise ValueError('Structural ranking inputs changed during the calculation')
    verify_inputs(protocol,inputs)
    return result


def execute(payload):
    context=captured_context(payload);protocol=context[1]
    root=Path(payload['destination']);root.mkdir(parents=True,exist_ok=False)
    transfer=prepare_nuclear_snapshot(transfer_request(payload,protocol),root)
    return rank_transferred(payload,context,root,transfer,root)


if __name__=='__main__':
    import sys
    request=json.loads(Path(sys.argv[1]).read_text())
    write_json(sys.argv[2],execute(request))
