"""Training dispatch binds a closed, observation-free numerical inventory."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.preparation.structural_jobs import (
    create_plan, checked_blocks, read_plan, plan_inputs,
)
from gpx_recipe.preparation.worker import validate_request
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.native_protocol import validate_native_payload
from test_structural_ranker_cohort import block


def inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    source = root/'parent.cif'; source.write_text('data_public_training_contract\n')
    instrument=root/'profile.instprm';instrument.write_text('# public profile identity control\n')
    grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=101,bank=1)
    directories = [block(root), block(root, 'second', label=65, offset=.013)]
    for directory in directories:
        path = directory/'examples.json'; d = json.loads(path.read_text())
        d['source_sha256'] = file_sha(source)
        d['instrument_sha256']=file_sha(instrument);d['grid']=grid
        d['groups'] = [g.replace('a'*64, d['source_sha256']) for g in d['groups']]
        path.write_text(json.dumps(d))
    plan_path = root/'plan.json'
    create_plan(source, directories, plan_path)
    return dict(source=str(source),instrument=str(instrument),grid=grid,plan=str(plan_path),destination=str(root/'trained'),
                settings=dict(bins=128, epochs=1, workers=1, augmentation=None)), directories


def test_example_worker_preserves_full_portfolio_and_shard_contract():
    request = dict(source='parent.cif', instrument='bank.instprm', child='child.cif',
        grid=dict(kind='PNT', lower=3000., upper=40000., bins=701, bank=1), destination='examples',
        settings=dict(candidate=dict(id='child', transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],
            origin_shift=[0,0,0], child=dict(path='child.cif', sha256='c'*64)),
            mode_radius=.05, cell_radius=.002, seed=1947, trajectories=64,
            trajectory_slice=[8,16]))
    before = copy.deepcopy(request)
    result = validate_request('structural-examples', request)
    assert request == before
    assert result['settings']['steps'] == 4
    assert result['settings']['trajectory_slice'] == [8,16]
    validate_native_payload('preparation-structural-examples', request)
    for interval in [[0,0], [0,65], [-1,2], [False,2], [0,1,2]]:
        with pytest.raises(ValueError):
            validate_request('structural-examples', dict(request,
                settings=dict(request['settings'], trajectory_slice=interval)))


@pytest.mark.parametrize('extra', ['api_key','python','screening_qualified','observations'])
def test_training_settings_do_not_admit_secrets_code_or_qualification_flags(tmp_path, extra):
    request, _ = inputs(tmp_path)
    request['settings'][extra] = 'not allowed'
    with pytest.raises(ValueError):
        validate_native_payload('preparation-structural-training', request)


def test_plan_binds_source_profile_and_every_native_block(tmp_path):
    request, directories = inputs(tmp_path)
    plan = read_plan(request['plan'])
    assert checked_blocks(request['plan'], plan) == directories
    assert len(plan_inputs(request['plan'])) == 4
    before = copy.deepcopy(request)
    validate_native_payload('preparation-structural-training', request)
    assert request == before
    path = directories[0]/'examples.json'; path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='block changed'):
        checked_blocks(request['plan'], plan)


@pytest.mark.parametrize('change', ['duplicate','escape','extra','profile','parent'])
def test_ambiguous_or_foreign_training_inventory_is_rejected(tmp_path, change):
    request, _ = inputs(tmp_path); path = Path(request['plan']); d = json.loads(path.read_text())
    if change == 'duplicate': d['blocks'].append(d['blocks'][0])
    elif change == 'escape': d['blocks'][0]['directory'] = '../outside'
    elif change == 'extra': d['python'] = 'not accepted'
    elif change == 'profile': d['instrument_sha256'] = 'd'*64
    else: d['parent_sha256'] = 'e'*64
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        checked_blocks(path, read_plan(path))


def test_training_job_cannot_silently_invent_missing_impurity_templates(tmp_path):
    request, _ = inputs(tmp_path)
    request['settings']['augmentation'] = dict(impurity_fraction=[0.,.1])
    with pytest.raises(ValueError, match='no declared native impurity'):
        validate_request('structural-training', request)


def test_training_plan_reserves_packet_slots_for_parent_and_plan(tmp_path):
    request, _ = inputs(tmp_path); path = Path(request['plan']); d = json.loads(path.read_text())
    d['blocks'] = [dict(d['blocks'][0], directory=f'block-{i}') for i in range(5000)]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='bounded parent/profile'):
        read_plan(path)
