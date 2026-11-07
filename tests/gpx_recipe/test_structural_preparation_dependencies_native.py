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
    assets=restored.prepared_assets(training['id'])
    metadata=next(p for p in assets.values() if p.name=='model.json')
    ranker=Ranker(metadata.parent)
    assert ranker.metadata['classes']==[63,65]
    assert ranker.metadata['screening_qualified'] is False
    assert len(ranker.metadata['candidates'])==2
    assert not list(restored.root.rglob('features.npy'))
