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


def test_phase_summary_joins_native_symmetry_by_id_not_same_composition():
    from copy import deepcopy
    from scripts.ndip_outputs import _native_phase_summaries
    phases=[dict(phase_id='mp-27630',compound_name='CrCl3',space_group='C2/m (12)',weight_fraction_pct='48.635',is_main='1'),
            dict(phase_id='mp-567504',compound_name='CrCl3',space_group='R-3 (148)',weight_fraction_pct='51.365',is_main='0')]
    native=[dict(phase_id='mp-567504',space_group='P -1',source_gpx='accepted.gpx'),
            dict(phase_id='mp-27630',space_group='P -1',source_gpx='accepted.gpx')]
    before=deepcopy((phases,native))
    result=_native_phase_summaries(phases,native,source='accepted.gpx')
    assert [row['phase_id'] for row in result]==['mp-27630','mp-567504']
    assert [row['space_group'] for row in result]==['P -1','P -1']
    assert [row['catalog_space_group'] for row in result]==['C2/m (12)','R-3 (148)']
    assert [row['weight_fraction_pct'] for row in result]==['48.635','51.365']
    assert [row['is_main'] for row in result]==['1','0']
    assert all(row['space_group_source']=='retained_native_gpx' for row in result)
    assert (phases,native)==before


def test_phase_summary_absent_ambiguous_or_foreign_native_keeps_catalog():
    from scripts.ndip_outputs import _native_phase_summaries
    phase=dict(phase_id='a',compound_name='CrCl3',space_group='C2/m (12)')
    row=dict(phase_id='a',space_group='P -1',source_gpx='accepted.gpx')
    for phases,native in [([phase],[]),([phase],[dict(row,phase_id='other')]),
                          ([phase],[dict(row,source_gpx='rejected.gpx')]),
                          ([phase],[row,row]),([phase,phase],[row]),
                          ([phase],[dict(row,space_group='unknown')]),
                          ([phase],[dict(row,space_group=None)])]:
        assert _native_phase_summaries(phases,native,source='accepted.gpx')==phases


def test_published_summary_uses_native_symmetry_without_changing_source_or_fit(tmp_path):
    import csv,json,pickle
    from unittest.mock import patch
    from scripts.ndip_outputs import collect_outputs
    run=tmp_path/'run';project=run/'Technical/GSAS_Projects/seq_final_accepted_model.gpx'
    project.parent.mkdir(parents=True)
    records=[[['Phases',{}],phase('mp-27630',1,100,[4,5,6,90,105,90,115])]]
    records[0][1][1]['General']['SGData']['SpGrp']='P -1'
    with project.open('wb') as handle:
        for record in records:pickle.dump(record,handle,protocol=1)
    table=run/'Summary_Fractions.csv'
    with table.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['phase_id','compound_name','space_group','weight_fraction_pct','is_main'])
        writer.writeheader();writer.writerow(dict(phase_id='mp-27630',compound_name='CrCl3',space_group='C2/m (12)',weight_fraction_pct='100',is_main='1'))
    original_project=project.read_bytes();original_table=table.read_bytes()
    native=[dict(phase_id='mp-27630',space_group='P -1',source_gpx=project.relative_to(run).as_posix(),weight_percent=100,a=4)]
    with patch('scripts.ndip_lattice.extract_lattice',return_value=(native,12.34)):
        result=collect_outputs(run,tmp_path/'portal',mode='full',include_archive=False)
    published=json.loads((tmp_path/'portal/summary.json').read_text())
    assert published['phases']==result['phases']
    assert published['phases'][0]['space_group']=='P -1'
    assert published['phases'][0]['catalog_space_group']=='C2/m (12)'
    assert published['phases'][0]['native_space_group']=='P -1'
    assert published['phases'][0]['weight_fraction_pct']=='100'
    assert result['final_rwp']==12.34 and result['lattice_parameters']==native
    assert project.read_bytes()==original_project and table.read_bytes()==original_table
