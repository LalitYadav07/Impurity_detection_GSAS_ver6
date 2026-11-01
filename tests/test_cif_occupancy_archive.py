import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nova/src'))
from cif_occupancy import AUTH_ENV, activate_archive, load_authorized_structure, policy_set_hash
from db_pack_builder import build_mini_db_pack
from aniso_db_loader import DBLoader,CatalogPaths
from radar_pd_nova.library_setup import inspect_library_review,activation_for_review,require_library_activation
from test_cif_occupancy_validator import file


def pack(tmp_path):
    p=file(tmp_path)
    result=build_mini_db_pack([p],tmp_path/'db',source_type='neutron',reviewed_occupancy_excess=.001,occupancy_review_note='Synthetic reviewed allowance')
    archive=tmp_path/'db.zip'
    with zipfile.ZipFile(archive,'w') as output:
        for path in result.pack_root.rglob('*'):
            if path.is_file():output.write(path,arcname='db/'+path.relative_to(result.pack_root).as_posix())
    return p,result,archive


def test_builder_to_archive_to_activated_loader_preserves_source(tmp_path,monkeypatch):
    monkeypatch.delenv(AUTH_ENV,raising=False)
    p,result,archive=pack(tmp_path)
    preview=inspect_library_review(archive)
    assert preview['requires_review'] and preview['reviews'][0]['total']==pytest.approx(1.00066)
    with pytest.raises(ValueError):require_library_activation(preview,{})
    activation=activation_for_review(preview);require_library_activation(preview,activation)
    with pytest.raises(ValueError):activate_archive(archive,result.pack_root,tmp_path,{})
    proof=activate_archive(archive,result.pack_root,tmp_path,activation)
    assert proof['policy_set_sha256']==preview['policy_set_sha256']
    paths=result.db_config
    loader=DBLoader(CatalogPaths(catalog_csv=paths['catalog_csv'],cif_map_json=paths['cif_map_json'],original_json=paths['original_json']))
    path=Path(loader.ensure_cif_on_disk(result.phase_ids[0]))
    assert path.read_bytes()==p.read_bytes()
    assert sum(loader.load_structure(result.phase_ids[0])[0].species.values())==pytest.approx(1.00066)
    monkeypatch.delenv(AUTH_ENV)
    with pytest.raises(ValueError):loader.load_structure(result.phase_ids[0])


def test_replacement_archive_or_policy_never_reuses_approval(tmp_path,monkeypatch):
    monkeypatch.delenv(AUTH_ENV,raising=False)
    p,result,archive=pack(tmp_path);activation=activation_for_review(inspect_library_review(archive))
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed after review'):activate_archive(archive,result.pack_root,tmp_path,activation)
    assert AUTH_ENV not in __import__('os').environ


def test_review_cap_and_note_required_at_producer(tmp_path):
    p=file(tmp_path)
    with pytest.raises(ValueError):build_mini_db_pack([p],tmp_path/'a',source_type='neutron',reviewed_occupancy_excess=.002,occupancy_review_note='Review')
    with pytest.raises(ValueError):build_mini_db_pack([p],tmp_path/'b',source_type='neutron',reviewed_occupancy_excess=.001)


def test_runner_recomputes_review_before_any_scientific_invocation(tmp_path,monkeypatch):
    import ndip_runner
    p,result,archive=pack(tmp_path)
    config=tmp_path/'config.yaml'
    assert ndip_runner.main(['configure','--mode','full','--radiation','neutron','--allowed-elements','C, Li','--output',str(config)])==0
    data=tmp_path/'data.gsa';data.write_text('BANK 1\n')
    instrument=tmp_path/'profile.instprm';instrument.write_text('#GSAS-II instrument parameter file\nType:PNT\ndifC:10000\n')
    activation=activation_for_review(inspect_library_review(archive))
    review_path=tmp_path/'permission.json';review_path.write_text(json.dumps(activation))
    arguments=['analyze','--config',str(config),'--data',str(data),'--instrument',str(instrument),
               '--db-pack',str(archive),'--db-root',str(tmp_path/'unused_builtin'),
               '--work-dir',str(tmp_path/'work'),'--output-dir',str(tmp_path/'portal'),'--dry-run']
    called=[]
    monkeypatch.setattr(ndip_runner.subprocess,'Popen',lambda *a,**k:called.append((a,k)))
    assert ndip_runner.main(arguments)==1
    assert not (tmp_path/'portal'/'command.json').exists()
    assert not called
    assert ndip_runner.main(arguments+['--library-review-activation',str(review_path)])==0
    receipt=json.loads((tmp_path/'portal'/'input_manifest.json').read_text())['library_review']
    assert receipt['verification']=='raw and symmetry-expanded source evidence recomputed'
    assert receipt['archive_sha256']==activation['archive_sha256']
    assert not called  # dry-run never dispatches even after valid review


def test_nudge_cache_rechecks_revoked_policy_and_original_bytes(tmp_path,monkeypatch):
    from lattice_nudger import LatticeNudger,_rebuild_with_lattice
    from pymatgen.core import Lattice
    p,result,archive=pack(tmp_path)
    activate_archive(archive,result.pack_root,tmp_path,activation_for_review(inspect_library_review(archive)))
    paths=result.db_config
    loader=DBLoader(CatalogPaths(catalog_csv=paths['catalog_csv'],cif_map_json=paths['cif_map_json'],original_json=paths['original_json']))
    nudger=object.__new__(LatticeNudger);nudger.db=loader;nudger.radiation='neutron'
    nudger._structure_cache={};nudger._cif_source_cache={};nudger._structure_source_keys={}
    original=nudger._structure_for_pid(result.phase_ids[0])
    changed=_rebuild_with_lattice(original,Lattice.cubic(4.01))
    assert sum(changed[0].species.values())==pytest.approx(1.00066)
    output=Path(nudger._write_nudged_cif(result.phase_ids[0],changed,str(tmp_path/'nudged')))
    assert sum(load_authorized_structure(output)[0].species.values())==pytest.approx(1.00066)
    monkeypatch.delenv(AUTH_ENV)
    with pytest.raises(ValueError,match='occupancy total'):
        nudger._structure_for_pid(result.phase_ids[0])


def test_forged_nonempty_evidence_is_rejected_before_activation(tmp_path,monkeypatch):
    from cif_occupancy import policy_hash
    p,result,archive=pack(tmp_path)
    manifest=json.loads(result.manifest_path.read_text())
    review=manifest['occupancy_reviews'][result.phase_ids[0]]
    review['composition']={'C':999,'Li':666}
    review['policy_sha256']=policy_hash(review)
    result.manifest_path.write_text(json.dumps(manifest))
    with zipfile.ZipFile(archive,'w') as output:
        for path in result.pack_root.rglob('*'):
            if path.is_file():output.write(path,arcname='db/'+path.relative_to(result.pack_root).as_posix())
    declared=inspect_library_review(archive)
    assert declared['actual_affected_site_count'] is None
    with pytest.raises(ValueError,match='evidence'):
        activate_archive(archive,result.pack_root,tmp_path,activation_for_review(declared))
    assert AUTH_ENV not in __import__('os').environ


def test_mixed_valid_and_rounded_sources_only_declare_actual_excess(tmp_path,monkeypatch):
    import db_pack_builder
    monkeypatch.setattr(db_pack_builder,'_pick_workers',lambda _:1)
    rounded=file(tmp_path);rounded.rename(tmp_path/'rounded.cif')
    valid=file(tmp_path,'0.4')
    result=build_mini_db_pack([tmp_path/'rounded.cif',valid],tmp_path/'mixed',source_type='neutron',
        reviewed_occupancy_excess=.001,occupancy_review_note='Reviewed synthetic round-off')
    manifest=json.loads(result.manifest_path.read_text())
    assert len(result.phase_ids)==2 and not result.failures
    assert len(manifest['occupancy_reviews'])==1
    assert next(iter(manifest['occupancy_reviews'].values()))['affected_sites']
