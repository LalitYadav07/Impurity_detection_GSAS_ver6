from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest

from radar_pd_nova.battery.discovery import DiscoveryLimits, discover_proposal, family_runs, inspect_run


def test_idl_reference_setup_is_evidence_and_cannot_replace_proposal_identity(tmp_path):
    family=tmp_path/'NOM/IPTS-42/shared/autoNOM';family.mkdir(parents=True)
    (family/'exp.ini').write_text('required *****\nDia 123\nDiaBg 124\nVana 125\nMTc 126\nrecali no\n# IPTS 7\n')
    (family/'nomad_123.calfile').write_text('detector calibration candidate')
    (family/'nomad_123.extcalfile').write_text('extended calibration candidate')
    (family/'los.ini').write_text('Scans 140-145\nTemplates do_not_execute.bat\n')
    result=discover_proposal('NOM','IPTS-42',facility_root=tmp_path)
    assert result['ipts']=='IPTS-42' and len(result['reduction_setups'])==2
    setup=next(row for row in result['reduction_setups'] if row['path'].endswith('exp.ini'))
    assert setup['proposal_comment_differs'] and setup['proposal_comments']==['IPTS-7']
    assert setup['reference_runs'][0]==dict(role='diamond',run=123,qualification='unverified_reference')
    assert 'acquisition identity unchanged' in setup['summary']
    assert all(row['qualification']=='not_a_gsas_peak_profile' for row in result['calibration_candidates'])
    assert next(row for row in result['reduction_setups'] if row['path'].endswith('los.ini'))['fields']['Templates']=='do_not_execute.bat'


def test_duplicate_or_oversized_idl_setup_is_reported_without_trusting_partial_fields(tmp_path):
    family=tmp_path/'NOM/IPTS-42/shared/autoNOM';family.mkdir(parents=True)
    (family/'exp.ini').write_text('Dia 123\nDia 456\n')
    result=discover_proposal('NOM','IPTS-42',facility_root=tmp_path)
    assert not result['reduction_setups'] and any(i['code']=='reduction_setup_unavailable' for i in result['issues'])
    (family/'exp.ini').write_text('Dia 123\n'+'#'*100)
    result=discover_proposal('NOM','IPTS-42',facility_root=tmp_path,limits=DiscoveryLimits(header_bytes=32))
    assert not result['reduction_setups'] and any('bounded' in i['message'] for i in result['issues'])


def raw_file(root, *, run=138504, ipts="IPTS-21886", layout="nexus", start="2019-12-09T12:00:05.166293667-05:00"):
    proposal = root / "NOM" / ipts
    if layout == "nexus":
        path = proposal / "nexus" / f"NOM_{run}.nxs.h5"
    else:
        path = proposal / "0" / str(run) / "NeXus" / f"NOM_{run}_event.nxs"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in {
            "run_number": run, "experiment_identifier": ipts, "title": "LNO_final at temperature 0.0 K.",
            "start_time": start, "end_time": "2019-12-09T12:05:05.349549667-05:00",
            "sample/id": "66548", "sample/name": "ZDND003", "sample/chemical_formula": "registered mixture",
        }.items(): f.create_dataset("entry/" + name, data=value)
        f.create_dataset("entry/bank1_events/event_time_offset", shape=(10_000_000,), dtype="f8")
        v = f.create_dataset("entry/DASlogs/BL1B:SE:SampleVolt/value", data=[0.0])
        v.attrs["units"] = "V"
        f.create_dataset("entry/DASlogs/cycler_current/value", shape=(200_000,), dtype="f8")
    return path


def test_shallow_discovery_finds_distinct_reductions_and_unbound_cyclers(tmp_path):
    shared = tmp_path / "NOM/IPTS-21886/shared"
    for folder in ("autoreduce", "autoNOM/GSAS", "autoNOM/Input", "Echem", "results/deep/private"):
        (shared / folder).mkdir(parents=True, exist_ok=True)
    (shared / "autoreduce/NOM_138504.gsa").write_text("placeholder")
    (shared / "autoNOM/GSAS/NOM138504tof.gsa").write_text("placeholder")
    (shared / "autoNOM/los.csv").write_text("RUN,SAMPLE\n138504,ZDND003\n")
    (shared / "Echem/cell.mpt").write_text("EC-Lab ASCII FILE\n")
    (shared / "Echem/cell.csv").write_text("timestamp,voltage_v,current_a\n2019-12-09T12:00:00-05:00,3.0,0.1\n")
    (shared / "Echem/plot.csv").write_text("x,intensity,error\n1,2,3\n")
    (shared / "results/deep/private/hidden.mpr").write_bytes(b"not read")
    result = discover_proposal("NOM", "IPTS-21886", facility_root=tmp_path)
    assert {r["folder"] for r in result["product_families"]} == {"shared/autoreduce", "shared/autoNOM/GSAS"}
    assert {r["path"] for r in result["cycler_candidates"]} == {"shared/Echem/cell.mpt", "shared/Echem/cell.csv"}
    assert all(r["binding"] == "unbound" for r in result["cycler_candidates"])
    assert result["metadata_tables"][0]["path"] == "shared/autoNOM/los.csv"


def test_powgen_hr_hi_collections_remain_distinct_without_crawling_unrelated_results(tmp_path):
    from radar_pd_nova.battery.discovery import family_runs
    shared=tmp_path/'PG3/IPTS-42/shared'
    for folder in ('2018_calibration/HR','2018_calibration/HI','2018_calibration/private/deep'):
        (shared/folder).mkdir(parents=True)
        (shared/folder/'PG3_10.gsa').write_text('Inventory only; no chemistry inferred')
    (shared/'2018_calibration/HR/PG3_10-2.xye').write_text('Alternative representation')
    (shared/'2018_calibration/HR/NOM_11.gsa').write_text('Another instrument')
    result=discover_proposal('PG3','IPTS-42',facility_root=tmp_path)
    assert {r['folder'] for r in result['product_families']}=={'shared/2018_calibration/HR','shared/2018_calibration/HI'}
    assert family_runs(result,'shared/2018_calibration/HR')==[10]
    assert result['status']=='available'
    assert next(r for r in result['product_families'] if r['folder'].endswith('/HR'))['file_count']==2


def test_archive_members_are_inventory_only_and_path_traversal_is_ignored(tmp_path):
    shared = tmp_path / "NOM/IPTS-21886/shared"; shared.mkdir(parents=True)
    with zipfile.ZipFile(shared / "echem.zip", "w") as z:
        z.writestr("LNO/cell.mpr", b"binary placeholder")
        z.writestr("../outside.mpr", b"invalid")
        z.writestr("/absolute.mpr", b"invalid")
    result = discover_proposal("NOM", "IPTS-21886", facility_root=tmp_path)
    assert [r["member"] for r in result["cycler_candidates"]] == ["LNO/cell.mpr"]
    assert not (shared / "LNO").exists()


def test_child_symlinks_are_not_followed(tmp_path):
    shared = tmp_path / "NOM/IPTS-21886/shared"; shared.mkdir(parents=True)
    outside = tmp_path / "private"; outside.mkdir()
    (outside / "cell.mpr").write_bytes(b"private")
    (shared / "escape").symlink_to(outside, target_is_directory=True)
    assert discover_proposal("NOM", "IPTS-21886", facility_root=tmp_path)["cycler_candidates"] == []


@pytest.mark.parametrize("layout", ["nexus", "legacy"])
def test_old_and_new_nexus_metadata_preserves_exposure_and_never_reads_events(tmp_path, monkeypatch, layout):
    raw_file(tmp_path, layout=layout)
    original = h5py.Dataset.__getitem__
    def guarded(node, key):
        assert "events" not in node.name
        assert node.size <= 32
        return original(node, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    result = inspect_run("NOM", "IPTS-21886", 138504, facility_root=tmp_path)
    assert result["status"] == "available"
    assert result["fields"]["sample_id"]["value"] == "66548"
    assert result["exposure"]["duration_seconds"] == pytest.approx(300.183256)
    assert result["exposure"]["start_utc"].startswith("2019-12-09T17:00:05")
    assert all(c["binding"] == "unreviewed" for c in result["condition_channels"])
    assert "values" not in next(c for c in result["condition_channels"] if c["name"] == "cycler_current")["value"]
    assert result["fields"]["run_title"]["value"].endswith("0.0 K.")
    assert "temperature" not in result


def test_missing_timezone_and_conflicting_run_are_visible(tmp_path):
    path = raw_file(tmp_path, start="2019-12-09T12:00:05")
    with h5py.File(path, "r+") as f: f["entry/run_number"][()] = 138502
    result = inspect_run("NOM", "IPTS-21886", 138504, facility_root=tmp_path)
    assert result["status"] == "conflict"
    assert {i["code"] for i in result["issues"]} == {"run_conflict", "exposure_unresolved"}
    assert "exposure" not in result


def test_external_hdf5_links_are_not_metadata_sources(tmp_path):
    path = raw_file(tmp_path)
    with h5py.File(path, "r+") as f:
        del f["entry/sample/id"]
        f["entry/sample/id"] = h5py.ExternalLink("/private/secret.h5", "/id")
    assert "sample_id" not in inspect_run("NOM", "IPTS-21886", 138504, facility_root=tmp_path)["fields"]


def test_discovery_budget_is_not_reported_as_complete(tmp_path):
    shared = tmp_path / "NOM/IPTS-21886/shared"
    (shared / "Echem").mkdir(parents=True)
    (shared / "other").mkdir()
    result = discover_proposal("NOM", "IPTS-21886", facility_root=tmp_path, limits=DiscoveryLimits(directories=1))
    assert result["status"] == "partial"
    assert any(i["code"] == "directory_budget" for i in result["issues"])


def test_missing_proposal_is_a_reviewable_state(tmp_path):
    result = discover_proposal("NOM", "IPTS-21886", facility_root=tmp_path)
    assert result["status"] == "unavailable"
    assert result["issues"][0]["code"] == "proposal_unavailable"


def test_unreadable_candidate_does_not_discard_other_discovered_sources(tmp_path,monkeypatch):
    from radar_pd_nova.battery import discovery
    shared=tmp_path/'NOM/IPTS-42/shared';shared.mkdir(parents=True)
    (shared/'unreadable.instprm').write_text('profile')
    (shared/'cell.mpr').write_bytes(b'candidate')
    original=discovery._signature
    def signature(path):
        if path.name=='unreadable.instprm':raise PermissionError('Selected profile is not readable')
        return original(path)
    monkeypatch.setattr(discovery,'_signature',signature)
    result=discover_proposal('NOM','IPTS-42',facility_root=tmp_path)
    assert result['status']=='partial' and len(result['cycler_candidates'])==1
    assert result['issues'][0]['code']=='candidate_unavailable'


def test_two_raw_copies_require_explicit_resolution(tmp_path):
    raw_file(tmp_path); raw_file(tmp_path, layout="legacy")
    with pytest.raises(ValueError, match="Several distinct"):
        inspect_run("NOM", "IPTS-21886", 138504, facility_root=tmp_path)


def test_raw_path_cannot_escape_the_selected_proposal(tmp_path):
    path = raw_file(tmp_path)
    relocated = tmp_path / "outside.nxs.h5"
    path.rename(relocated); path.symlink_to(relocated)
    with pytest.raises(ValueError, match="outside"):
        inspect_run("NOM", "IPTS-21886", 138504, facility_root=tmp_path)


def family_inventory(tmp_path):
    return dict(root=str(tmp_path), instrument='NOM', product_families=[dict(folder='shared/reduced')])


def test_family_run_listing_skips_type_checks_for_unrelated_names(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from radar_pd_nova.battery import discovery

    class Unrelated:
        def __init__(self, name): self.name = name
        def is_symlink(self): raise AssertionError('Unrelated entry was inspected')
        def is_dir(self, **kwargs): raise AssertionError('Unrelated entry was statted')

    folder = tmp_path/'shared/reduced'; folder.mkdir(parents=True)
    (folder/'NOM_12.gsa').write_text('No contents need to be read')
    with discovery.os.scandir(folder) as stream:
        valid = list(stream)
    entries = [Unrelated('NOM_1.pdf'), Unrelated('PG3_11.gsa'), Unrelated('subdirectory'), *valid]
    monkeypatch.setattr(discovery.os, 'scandir', lambda path: nullcontext(iter(entries)))
    assert family_runs(family_inventory(tmp_path), 'shared/reduced') == [12]


def test_family_run_listing_keeps_symlink_and_directory_protection(tmp_path):
    folder = tmp_path/'shared/reduced'; folder.mkdir(parents=True)
    (folder/'NOM_12.gsa').write_text('valid')
    (folder/'NOM_12-2.xye').write_text('same run, another bank')
    (folder/'NOM_2.gsa').write_text('valid')
    (folder/'NOM_8.gsa').mkdir()
    outside = tmp_path/'outside.gsa'; outside.write_text('outside the selected family')
    (folder/'NOM_99.gsa').symlink_to(outside)
    assert family_runs(family_inventory(tmp_path), 'shared/reduced') == [2, 12]


def test_family_run_listing_rejects_entry_budget_without_a_partial_list(tmp_path):
    folder = tmp_path/'shared/reduced'; folder.mkdir(parents=True)
    for run in range(1, 4):
        (folder/f'NOM_{run}.gsa').write_text('valid')
    with pytest.raises(ValueError, match='entry budget; no incomplete run list was used'):
        family_runs(family_inventory(tmp_path), 'shared/reduced', limits=DiscoveryLimits(entries_per_directory=2))


@pytest.mark.parametrize('stall', ['between_entries', 'final_entry'])
def test_family_run_listing_rejects_elapsed_budget_without_a_partial_list(tmp_path, monkeypatch, stall):
    from contextlib import nullcontext
    from radar_pd_nova.battery import discovery

    clock = [0.]
    class Entry:
        name = 'NOM_12.gsa'
        def is_symlink(self): return False
        def is_dir(self, **kwargs):
            if stall == 'final_entry': clock[0] = 2.
            return False

    def entries():
        yield Entry()
        if stall == 'between_entries':
            clock[0] = 2.
            yield Entry()

    monkeypatch.setattr(discovery.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(discovery.os, 'scandir', lambda path: nullcontext(entries()))
    with pytest.raises(ValueError, match='1 s budget; no incomplete run list was used'):
        family_runs(family_inventory(tmp_path), 'shared/reduced', limits=DiscoveryLimits(elapsed_seconds=1.))
