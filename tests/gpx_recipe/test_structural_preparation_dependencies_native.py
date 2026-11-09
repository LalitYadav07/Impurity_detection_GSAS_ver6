"""Real tensor training publishes into the same portable preparation library."""
from pathlib import Path
import pytest

pytest.importorskip('torch')

from gpx_recipe.preparation.worker import execute
from gpx_recipe.preparation.operations import stage_entry,publish_entry
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.structural_ranker import Ranker
from test_structural_preparation_dependencies import library_inputs,complete_examples,TRAINING


def test_dependency_training_and_portable_import_keep_the_model_usable(tmp_path):
    library,originals=library_inputs(tmp_path)
    complete_examples(library,originals,tmp_path/'examples-attempts')
    training=next(row for row in library.document['entries'] if row['model']['kind']==TRAINING)
    attempt=library.start(training['id'])
    request=stage_entry(library,training['id'],attempt,tmp_path/'training-attempt')
    result=execute(TRAINING,request)
    assert publish_entry(library,training['id'],attempt,request,result)['all_prepared']
    library.export(tmp_path/'complete.zip')
    restored=import_library(tmp_path/'complete.zip',tmp_path/'restored')
    assert restored.document.get('storage',{}).get('deferred_assets')
    assert restored.document['entries']==library.document['entries']
    assert not list((restored.root/'entries').rglob('examples.npz'))
    assert list((library.root/'entries').rglob('examples.npz'))
    assets=restored.prepared_assets(training['id'])
    metadata=next(p for p in assets.values() if p.name=='model.json')
    ranker=Ranker(metadata.parent)
    assert ranker.metadata['classes']==[63,65]
    assert ranker.metadata['screening_qualified'] is False
    assert len(ranker.metadata['candidates'])==2
    assert not list(restored.root.rglob('features.npy'))


def test_trained_library_ranks_a_later_pattern_through_the_captured_worker_packet(tmp_path):
    from gpx_recipe.preparation import hints,structural_ranking_hints as ranking
    from gpx_recipe.preparation.structural_cohort import read_block
    from gpx_recipe.symmetry.bundle import pack,unpack
    library,originals=library_inputs(tmp_path)
    complete_examples(library,originals,tmp_path/'examples-attempts')
    trainer=next(row for row in library.document['entries'] if row['model']['kind']==TRAINING)
    attempt=library.start(trainer['id'])
    request=stage_entry(library,trainer['id'],attempt,tmp_path/'training-attempt')
    result=execute(TRAINING,request)
    publish_entry(library,trainer['id'],attempt,request,result)
    candidates=[r['model']['settings']['candidate'] for r in library.document['entries']
                if r['model']['kind']=='structural-examples']
    protocol=dict(kind='structural',candidates=candidates)
    inputs=tmp_path/'inputs';document=hints.capture(library,protocol,inputs)
    names=hints.packet_paths(inputs,protocol)
    pack(inputs,tmp_path/'packet.zip',names)
    received=unpack(tmp_path/'packet.zip',tmp_path/'received')
    target=tmp_path/'worker-inputs';attached=hints.restore(document,protocol,received,target)
    assert attached['ranking_models']['attached']==1
    row=document['ranking_models']['entries'][0];data=read_block(originals['first']).arrays
    arguments=dict(background=data['background'],parent_sha256=row['files']['parent']['sha256'],
                   instrument_sha256=row['files']['instrument']['sha256'],entry_id=row['entry'])
    ranked=ranking.rank(target,protocol,data['q'],data['theory_intensity'][1],**arguments)
    assert ranked['ranking_available'] and not ranked['discarded_entries']
    assert set(ranked['candidate_order'])=={'first','second'} and len(ranked['candidate_order'])==2
    assert ranked['candidate_pruning_authorized'] is False and ranked['complete_native_search_required']
    assert all(set(f['candidate_ids'])<=set(ranked['candidate_order']) for f in ranked['families'])
    # A later selected k scope never acquires other constructions from the
    # broader prepared library, and scores are not renormalized into confidence.
    smaller=dict(protocol,candidates=candidates[1:]);small=tmp_path/'smaller'
    hints.capture(library,smaller,small)
    restricted=ranking.rank(small,smaller,data['q'],data['theory_intensity'][1],**arguments)
    assert restricted['ranking_available'] and restricted['candidate_order']==['second']
    assert all(f['candidate_ids']==['second'] for f in restricted['families'])
    import numpy as np
    valid=np.ones(len(data['q']),bool);valid[30:40]=False
    missing=ranking.rank(target,protocol,data['q'],data['theory_intensity'][1],valid=valid,**arguments)
    assert not missing['ranking_available'] and missing['candidate_order']==['first','second']
