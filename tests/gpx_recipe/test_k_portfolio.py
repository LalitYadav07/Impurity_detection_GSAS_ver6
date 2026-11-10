"""Scheduling/fallback contracts; no native calculations or facility data."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest

from test_symmetry_contracts import make_protocol
from gpx_recipe.symmetry import k_native,k_group
from gpx_recipe.symmetry.compiler import compile_native_bundle as compile_bundle,package_signature
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.k_search import portfolio_batch


def jobs(root,*,moment_model='symmetry_allowed',count=4):
    protocol=make_protocol(root/'inputs')
    protocol['implementation_sha256']=package_signature()
    protocol['policy'].update(moment_model=moment_model,maximum_trials=count)
    protocol['seeds']=[dict(protocol['seeds'][0],id=f'start-{i}') for i in range(count)]
    directories=[]
    for seed in protocol['seeds']:
        directory=root/seed['id']
        compile_bundle(protocol,root/'inputs',directory,candidate_id=protocol['candidates'][0]['id'],
            seed_id=seed['id'],approved_hash=digest(protocol))
        directories.append(directory.resolve())
    return protocol,directories


def test_groups_keep_full_exact_context_and_collinear_remains_isolated(tmp_path):
    _,first=jobs(tmp_path/'first');_,other=jobs(tmp_path/'other',count=2)
    # Same numerical source and protocol can share even across private folders.
    assert k_native.portfolio_groups(first)==[first]
    protocol,rows=jobs(tmp_path/'collinear',moment_model='collinear')
    assert k_native.portfolio_groups(rows)==[[p] for p in rows]
    # A different policy changes the physical scope and cannot share a context.
    job=json.loads((other[0]/'job.json').read_text())
    job['protocol']['policy']['maximum_moment']=5.
    (other[0]/'job.json').write_text(json.dumps(job))
    grouped=k_native.portfolio_groups([first[0],other[0],first[1]])
    assert grouped==[[first[0],first[1]],[other[0]]]


def test_finite_batches_keep_candidate_starts_together_and_other_contexts_visible():
    protocol=dict(policy=dict(moment_model='symmetry_allowed'))
    staged={f'{candidate}-{seed}':dict(task=(0,protocol,None,candidate,seed))
            for seed in range(3) for candidate in 'ABCDE'}
    batch=portfolio_batch(staged,1)
    assert batch==['A-0','B-0','A-1','B-1','A-2','B-2']
    assert len(staged)==15
    collinear=copy.deepcopy(staged)
    for row in collinear.values():row['task'][1]['policy']['moment_model']='collinear'
    assert len(portfolio_batch(collinear,1))==2


def fake_native(directory,root):
    root=Path(root);(root/'ordinary-was-run').touch()
    return dict(status='simulated',id=root.name)


def fake_group(directories):
    return [dict(directory=p,response=dict(status='failed' if i==1 else 'simulated',id=p.name,
        reason='Deliberate accelerator failure')) for i,p in enumerate(directories)]


def broken_group(directories):
    import os
    os._exit(3)


def slow_group(directories):time.sleep(2)


@pytest.mark.skipif(sys.platform!='linux',reason='Native process isolation is Linux-only')
@pytest.mark.parametrize('group_function',[fake_group,broken_group])
def test_failed_acceleration_retries_without_losing_or_repeating_successful_starts(tmp_path,monkeypatch,group_function):
    _,directories=jobs(tmp_path)
    monkeypatch.setattr(k_native,'runtime_identity',lambda:dict(magnetic_function_sha256='b'*64))
    monkeypatch.setattr(k_native,'require_runtime',lambda *_:None)
    monkeypatch.setattr(k_native,'template',fake_native)
    monkeypatch.setattr(k_group,'simulate_group',group_function)
    result=k_native.portfolio(dict(directories=[p.name for p in directories],workers=2,seconds=10),tmp_path)
    assert not result['pending'] and len(result['results'])==len(directories)
    by_name={r['directory']:r['response'] for r in result['results']}
    assert set(by_name)=={p.name for p in directories}
    assert all(r['status']=='simulated' for r in by_name.values())
    if group_function is fake_group:
        assert [p.name for p in directories if (p/'ordinary-was-run').exists()]==['start-1']
        assert by_name['start-1']['acceleration']['method']=='ordinary_native'
        assert len(result['group_runs'])==1
    else:assert all((p/'ordinary-was-run').exists() for p in directories)


@pytest.mark.skipif(sys.platform!='linux',reason='Native process isolation is Linux-only')
def test_operation_timeout_keeps_every_unreturned_start_pending(tmp_path,monkeypatch):
    _,directories=jobs(tmp_path)
    monkeypatch.setattr(k_native,'runtime_identity',lambda:dict(magnetic_function_sha256='b'*64))
    monkeypatch.setattr(k_native,'require_runtime',lambda *_:None)
    monkeypatch.setattr(k_group,'simulate_group',slow_group)
    result=k_native.portfolio(dict(directories=[p.name for p in directories],workers=1,seconds=.1),tmp_path)
    assert not result['results'] and set(result['pending'])=={p.name for p in directories}
