from pathlib import Path

from scripts.ndip_lattice import lattice_from_records, lattice_quality, final_project_key


def phase(name, scale, mass, cell):
    return [name, {"General": {"Cell": [False, *cell], "Mass": mass, "SGData": {"SpGrp": "P 1"}},
        "Histograms": {"PWDR test": {"Scale": [scale, True], "Use": True}}}]


def test_cells_and_weight_fractions_use_gsas_mass_normalization():
    cell = [4, 4, 4, 90, 90, 90, 64]
    records = [[["Phases", {}], phase("a", 1, 600, cell), phase("b", 1.5, 160, [5, 5, 5, 90, 90, 90, 125])]]
    rows, rwp = lattice_from_records(records, source="final.gpx")
    assert rows[0]["weight_percent"] == 100 * 600 / 840
    assert rows[1]["a"] == 5
    assert rows[1]["a_esd"] is None
    assert rows[1]["uncertainty_status"] == "not_estimated"
    assert lattice_quality(rows) == []
    rows[0]["weight_percent"] = 0
    assert lattice_quality(rows)[0]["code"] == "vanishing_phase"
    rows[0]["a"] = -1
    assert lattice_quality(rows)[0]["code"] == "invalid_cell"


def test_final_project_prefers_final_polish_then_highest_accepted_pass():
    files = [Path(p) for p in ["pattern_project.gpx", "seq_pass1_kept_polished.gpx", "seq_pass10_kept_polished.gpx"]]
    assert max(files, key=final_project_key).name == "seq_pass10_kept_polished.gpx"
    assert max([*files, Path("seq_final_main_polished.gpx")], key=final_project_key).name == "seq_final_main_polished.gpx"
    assert max([*files,Path('seq_final_joint_polished.gpx'),Path('seq_final_accepted_model.gpx')],key=final_project_key).name=='seq_final_accepted_model.gpx'


def test_native_symmetry_label_is_distinct_from_catalog_without_changing_identity():
    from copy import deepcopy
    row=phase('mp-27630',1,100,[4,5,6,90,105,90,115])
    row[1]['General']['SGData']['SpGrp']='P -1'
    records=[[['Phases',{}],row]];before=deepcopy(records)
    rows,_=lattice_from_records(records,source='final.gpx',labels={'mp-27630':'mp-27630 (SG C2/m, 12)'})
    assert rows[0]['phase']=='mp-27630 · P -1 (catalog C2/m)'
    assert rows[0]['phase_id']=='mp-27630' and rows[0]['space_group']=='P -1'
    assert rows[0]['catalog_space_group']=='C2/m, 12'
    assert rows[0]['native_space_group']=='P -1'
    assert records==before


def test_published_gpx_renaming_preserves_native_ids_references_and_original(tmp_path):
    from copy import deepcopy
    import pickle,shutil
    from scripts.ndip_outputs import _rewrite_gpx_phase_names,_read_gpx_records
    old='mp-27630';label='mp-27630 (SG C2/m, 12)';new='mp-27630 · P -1 (catalog C2/m)'
    general={'Name':old,'SGData':{'SpGrp':'P -1'},'Cell':[False,4,5,6,90,105,90,115],'pId':7}
    records=[[['Phases',{}],[old,{'General':general,'pId':7,'Atoms':[['Cr',0,0,0]],'Histograms':{'PWDR test':{'Scale':[1,True]}}}]],
             [['PWDR test',{}],['Reflection Lists',{old:{'RefList':[[1,2,3]]}}]],
             [['Restraints',{old:{'Bond':[]}}]], [['Constraints',{'Phase':[['native-index',7]]}]]]
    original=tmp_path/'original.gpx';published=tmp_path/'published.gpx'
    with original.open('wb') as stream:
        for record in records:pickle.dump(record,stream,protocol=1)
    before=original.read_bytes();shutil.copyfile(original,published)
    assert _rewrite_gpx_phase_names(published,{old:label})
    after=_read_gpx_records(published)
    assert after[0][1][0]==new and after[0][1][1]['General']['Name']==new
    assert after[0][1][1]['pId']==7 and after[0][1][1]['General']['SGData']==general['SGData']
    assert after[1][1][1][new]==records[1][1][1][old] and new in after[2][0][1]
    assert after[3]==records[3] and original.read_bytes()==before
    assert after[0][1][1]['Atoms']==records[0][1][1]['Atoms']
