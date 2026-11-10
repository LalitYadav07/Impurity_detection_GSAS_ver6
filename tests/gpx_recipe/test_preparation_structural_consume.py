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


def certified_bank(tmp_path,monkeypatch):
    from gpx_recipe.preparation import authority,certificates
    from gpx_recipe.symmetry.contracts import digest
    from gpx_recipe.symmetry.compiler import package_signature
    s,_,_,_=source(monkeypatch)
    context=bank_module.context
    monkeypatch.setattr(bank_module,'context',lambda s:dict(context(s),physics_sha256=digest(context(s))))
    path=tmp_path/'prepared.npz';receipt=bank_module.build(s,path,policy='joint_cell_v1')
    entry=dict(kind='structural',entry_sha256='e'*64,physics_sha256=receipt['context']['physics_sha256'],
        artifacts={'starts.npz':file_sha(path),'starts.json':file_sha(path.with_suffix('.json'))})
    certificate=tmp_path/'certificate.json';certificate.write_text(json.dumps(dict(schema=certificates.SCHEMA,
        implementation_sha256=package_signature(),result_archive_sha256='f'*64,entries=[entry])))
    envelope=tmp_path/'server.json';envelope.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[
        dict(certificate_path=str(certificate),tool_id='neutrons_radar_pd_model_campaign_prototype',tool_version='0.2.17',
            job_state='ok',dataset_state='ok',dataset_deleted=False,dataset_purged=False,
            job_stdout=certificates.MARKER+file_sha(certificate))])))
    monkeypatch.setattr(bank_module,'_native_banks',bank_module.OrderedDict())
    return s,path,envelope,certificate


def test_certified_cold_bank_retains_all_185_curves_with_three_current_native_probes(tmp_path,monkeypatch):
    from gpx_recipe.preparation import authority
    s,path,envelope,_=certified_bank(tmp_path,monkeypatch)
    calls=native_builder(monkeypatch);native=s.native;probes=[]
    def counted(point):probes.append(point.copy());return native(point)
    s.native=counted
    output=tmp_path/'received'/'starts.npz'
    with authority.server_scope(envelope,tmp_path):
        receipt,report=consume.prepare_starting_bank(s,output,[choice(path)])
    assert report['status']=='reused_certified_native_bank' and report['curve_bytes_reused']
    assert not calls and len(probes)==report['native_curves_recomputed']==3
    assert receipt['native_curves']==185 and output.read_bytes()==path.read_bytes()
    assert report['scientific_candidates_removed'] is False
    assert bank_module.native_in_this_process(s,output)


@pytest.mark.parametrize('change',['certificate','bank','context','native_probe'])
def test_certified_origin_never_bypasses_changed_bytes_context_or_native_predictions(tmp_path,monkeypatch,change):
    from gpx_recipe.preparation import authority
    s,path,envelope,certificate=certified_bank(tmp_path,monkeypatch)
    if change=='certificate':
        data=json.loads(certificate.read_text());data['entries'][0]['entry_sha256']='b'*64
        certificate.write_text(json.dumps(data))
    elif change=='bank':
        def corrupt(values,_):values['curves'][10,0]+=1e6
        rewrite_bank(path,corrupt)
    elif change=='context':s.identity='new-actual-context'
    else:
        native=s.native;s.native=lambda point:native(point)+.5
    calls=native_builder(monkeypatch)
    with authority.server_scope(envelope,tmp_path):
        receipt,report=consume.prepare_starting_bank(s,tmp_path/'owned.npz',[choice(path)])
    assert len(calls)==1 and not report['curve_bytes_reused']
    assert receipt['native_curves']==report['native_curves_recomputed']==185
    assert report['scientific_candidates_removed'] is False
    if change=='native_probe':assert any(p['relative_error']>1e-11 for p in report['current_native_probes'])
