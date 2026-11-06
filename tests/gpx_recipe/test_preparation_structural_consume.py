"""Structural handoff adversaries; artificial curves, no facility inputs."""
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_consume as consume
from gpx_recipe.structural_live.optimization import offline_native_start_bank as bank_module
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_live_offline_start_bank import source


def choice(path):
    return dict(entry='a'*24, bank='artificial bank',
                files=dict(path=path, receipt_path=path.with_suffix('.json')),
                sha256=file_sha(path), receipt_sha256=file_sha(path.with_suffix('.json')))


def native_builder(monkeypatch):
    calls=[]
    def build(s, path, policy, **kwargs):
        calls.append(dict(policy=policy, **kwargs))
        return bank_module.build(s, path, policy=policy)
    monkeypatch.setattr(consume, 'build', build)
    return calls


def rewrite_bank(path, change):
    with np.load(path, allow_pickle=False) as data:
        values={k: data[k].copy() for k in data.files}
    receipt=json.loads(path.with_suffix('.json').read_text())
    change(values, receipt)
    np.savez_compressed(path, **values)
    receipt['sha256']=file_sha(path)
    path.with_suffix('.json').write_text(json.dumps(receipt))


def test_only_exact_bytes_built_in_the_owned_process_skip_native_work(tmp_path, monkeypatch):
    s, _, _, _=source(monkeypatch)
    path=tmp_path/'original.npz'
    bank_module.build(s, path, policy='joint_cell_v1')
    calls=native_builder(monkeypatch)
    output=tmp_path/'received'/'starts.npz'
    receipt, report=consume.prepare_starting_bank(s, output, [choice(path)])
    assert report['status']=='reused_native_bank' and report['curve_bytes_reused']
    assert report['native_curves_recomputed']==0 and not calls
    assert receipt['native_curves']==185
    assert output.read_bytes()==path.read_bytes()
    assert report['observations_reused'] is False
    assert report['scientific_candidates_removed'] is False


def test_cold_import_recomputes_curves_and_proposal_steps_despite_valid_checksums(tmp_path, monkeypatch):
    s, _, _, _=source(monkeypatch)
    path=tmp_path/'uploaded.npz'
    bank_module.build(s, path, policy='joint_cell_v1')
    def corrupt(values, _):
        values['curves'][0, 0]+=1e6
        values['proposal_cell_steps'][:]=.111
    rewrite_bank(path, corrupt)
    uploaded=path.read_bytes()
    calls=native_builder(monkeypatch)
    output=tmp_path/'owned'/'starts.npz'
    receipt, report=consume.prepare_starting_bank(s, output, [choice(path)])
    assert len(calls)==1 and calls[0]['policy']=='joint_cell_v1'
    assert report['status']=='native_requalified'
    assert not report['curve_bytes_reused'] and not report['supplied_curves_match_current']
    assert report['native_curves_recomputed']==receipt['native_curves']==185
    assert report['supplied_maximum_relative_curve_error']>1
    bank=bank_module.OfflineNativeStartBank(s, output)
    for point, curve in zip(bank.points, bank.curves):
        np.testing.assert_array_equal(curve, s.native(point*s.radii))
    assert np.any(bank.proposal_cell_steps!=.111)
    assert path.read_bytes()==uploaded


@pytest.mark.parametrize('change', ['context', 'omitted_feasible_start', 'smaller_portfolio', 'bad_archive'])
def test_optional_preparation_cannot_narrow_the_complete_start_portfolio(tmp_path, monkeypatch, change):
    s, _, _, _=source(monkeypatch)
    path=tmp_path/'provided.npz'
    policy='multiscale_v1' if change=='smaller_portfolio' else 'joint_cell_v1'
    bank_module.build(s, path, policy=policy)
    if change=='context':
        s.identity='different-current-candidate'
    elif change=='bad_archive':
        receipt=json.loads(path.with_suffix('.json').read_text())
        path.write_bytes(b'not an NPZ archive')
        receipt['sha256']=file_sha(path)
        path.with_suffix('.json').write_text(json.dumps(receipt))
    elif change=='omitted_feasible_start':
        def omit(values, receipt):
            values['feasible'][values['original_indices'][-1]]=False
            values['points']=values['points'][:-1]
            values['curves']=values['curves'][:-1]
            values['original_indices']=values['original_indices'][:-1]
            receipt['native_curves']-=1
            receipt['chemically_infeasible_points']+=1
        rewrite_bank(path, omit)
    calls=native_builder(monkeypatch)
    output=tmp_path/'current'/'starts.npz'
    receipt, report=consume.prepare_starting_bank(s, output, [choice(path)])
    assert len(calls)==1 and report['status']=='rebuilt_current_context'
    assert len(report['discarded_entries'])==1
    assert receipt['native_curves']==185
    assert len(bank_module.OfflineNativeStartBank(s, output).points)==185
    assert report['scientific_candidates_removed'] is False


def test_valid_bank_after_process_provenance_is_lost_is_rechecked(tmp_path, monkeypatch):
    s, _, _, _=source(monkeypatch)
    path=tmp_path/'previous-process.npz'
    bank_module.build(s, path, policy='joint_cell_v1')
    monkeypatch.setattr(bank_module, '_native_banks', bank_module.OrderedDict())
    calls=native_builder(monkeypatch)
    _, report=consume.prepare_starting_bank(s, tmp_path/'current.npz', [choice(path)])
    assert len(calls)==1 and report['status']=='native_requalified'
    assert report['supplied_curves_match_current']
    assert not report['curve_bytes_reused']
    assert report['qualification_persisted_as_trust'] is False
