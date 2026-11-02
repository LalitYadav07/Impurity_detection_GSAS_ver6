"""Characterize the current CIF path before a reviewed-policy implementation.

Only synthetic structures are used. This covers the original failures and
the explicit policy's propagation through each consumer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aniso_db_loader import DBLoader
from db_pack import get_db_pack_layout
from db_pack_builder import _materialize_phase, _prepare_output_dirs, collect_phase_inputs, resolve_simulation_settings
from gsas_cif_import import add_verified_phase
from pymatgen.core import Structure


HEADER = """data_synthetic
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""


def source(tmp_path, occupancy='0.4', *, negative=False, crlf=False):
    second = '-0.1' if negative else occupancy
    text = HEADER + f'C1 C 0 0 0 0.6\nLi1 Li 0 0 0 {second}\n'
    data = text.replace('\n', '\r\n').encode() if crlf else text.encode()
    path = tmp_path/'synthetic.cif';path.write_bytes(data)
    return path, data


def mapped_loader(path, *, original_json=None):
    loader = object.__new__(DBLoader)
    loader._cif_map = {'synthetic':str(path)}
    loader.paths = SimpleNamespace(original_json=original_json)
    return loader


def review_document(path, maximum=0.001):
    return {'schema':'reviewed-cif-occupancy/v1', 'reviewed':True,
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'maximum_site_excess':maximum,
            'review_note':'Synthetic test: preserve source values; no normalization'}


def test_default_closed_structure_and_original_mapped_bytes_survive(tmp_path):
    path, raw=source(tmp_path)
    loader=mapped_loader(path)
    assert Path(loader.ensure_cif_on_disk('synthetic')).read_bytes()==raw
    structure=loader.load_structure('synthetic')
    assert sum(structure[0].species.values())==pytest.approx(1)
    assert path.read_bytes()==raw


def test_current_materializer_reports_default_overfill_and_keeps_source_text(tmp_path):
    path, raw=source(tmp_path,'0.40066')
    phase=collect_phase_inputs([path])[0]
    layout=get_db_pack_layout(tmp_path/'library')
    _prepare_output_dirs(layout,overwrite=False)
    row,profile,stable,failure=_materialize_phase(phase,resolve_simulation_settings('neutron'),layout)
    assert row is profile is stable is None
    assert 'occupancy total' in failure['error']
    assert (layout.cifs_dir/(phase.phase_id+'.cif')).read_bytes()==raw
    assert path.read_bytes()==raw


def test_collection_preserves_crlf_and_original_bytes(tmp_path):
    path,raw=source(tmp_path,crlf=True)
    phase=collect_phase_inputs([path])[0]
    assert phase.cif_content.encode()==raw
    assert path.read_bytes()==raw


def test_dbloader_default_rejects_overfilled_mapped_original(tmp_path):
    path,_=source(tmp_path,'0.40066')
    with pytest.raises(ValueError,match='occupancy total'):
        mapped_loader(path).load_structure('synthetic')


def test_native_import_guard_fails_before_calling_gsas(tmp_path):
    path,_=source(tmp_path,'0.40066')
    class Project:
        filename=str(tmp_path/'work.gpx')
        called=False
        def add_phase(self,*args,**kwargs):
            self.called=True
            raise AssertionError('GSAS must not be invoked after parser rejection')
    project=Project()
    with pytest.raises(ValueError,match='occupancy total'):
        add_verified_phase(project,path,phasename='synthetic')
    assert project.called is False


def test_original_json_fallback_keeps_source_occupancy_values(tmp_path):
    path,raw=source(tmp_path,'0.40066')
    original=tmp_path/'original.json'
    original.write_text(json.dumps({'synthetic':{'cif_content':raw.decode()}}))
    loader=mapped_loader(path,original_json=str(original));loader._cif_map={}
    recovered=Path(loader.ensure_cif_on_disk('synthetic',out_dir=str(tmp_path/'cache')))
    assert recovered.read_bytes()==raw


def test_original_json_cache_cannot_materialize_an_earlier_source_revision(tmp_path):
    path,raw=source(tmp_path)
    original=tmp_path/'original.json'
    original.write_text(json.dumps({'synthetic':{'cif_content':raw.decode()}}))
    loader=mapped_loader(path,original_json=str(original));loader._cif_map={}
    cache=tmp_path/'cache'
    first=Path(loader.ensure_cif_on_disk('synthetic',out_dir=str(cache)))
    changed=raw.replace(b'0.6',b'0.55')
    original.write_text(json.dumps({'synthetic':{'cif_content':changed.decode()}}))
    assert Path(loader.ensure_cif_on_disk('synthetic',out_dir=str(cache))).read_bytes()==changed


def test_nudge_export_keeps_raw_sites_and_tracks_original_hash(tmp_path,monkeypatch):
    import CifFile
    from lattice_nudger import LatticeNudger
    from cif_occupancy import load_authorized_structure
    path,raw=source(tmp_path,'0.40066')
    activate_test_review(tmp_path,path,monkeypatch)
    structure=load_authorized_structure(path)
    nudger=object.__new__(LatticeNudger)
    nudger._cif_source_cache={'synthetic':str(path)};nudger.radiation='neutron'
    output=Path(nudger._write_nudged_cif('synthetic',structure,str(tmp_path/'nudged')))
    document=CifFile.ReadCif(str(output));block=document[list(document.keys())[0]]
    assert [float(v) for v in block['_atom_site_occupancy']]==[0.6,0.40066]
    receipt=json.loads(Path(str(output)+'.nudge.json').read_text())
    assert receipt['source_sha256']==hashlib.sha256(raw).hexdigest()
    assert path.read_bytes()==raw


def activate_test_review(tmp_path,path,monkeypatch):
    from cif_occupancy import AUTH_ENV, make_review, write_review
    review=make_review(path.read_bytes(),.001,'Synthetic reviewed source')
    write_review(path,review)
    activation=tmp_path/'activation.json'
    activation.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{review['policy_sha256']:review}}))
    monkeypatch.setenv(AUTH_ENV,str(activation))
    return review


def test_reviewed_policy_must_reach_dbloader_without_normalizing(tmp_path,monkeypatch):
    path,raw=source(tmp_path,'0.40066')
    review=activate_test_review(tmp_path,path,monkeypatch)
    original=tmp_path/'highsymm_metadata.json'
    original.write_text(json.dumps({'synthetic':{'cif_content':raw.decode(),'occupancy_review':review}}))
    structure=mapped_loader(path,original_json=str(original)).load_structure('synthetic')
    assert sum(structure[0].species.values())==pytest.approx(1.00066,abs=1e-12)
    assert path.read_bytes()==raw


def test_reviewed_policy_must_reach_native_guard_before_gsas(tmp_path,monkeypatch):
    path,raw=source(tmp_path,'0.40066')
    activate_test_review(tmp_path,path,monkeypatch)
    import gsas_cif_import
    monkeypatch.setattr(gsas_cif_import,'prepare_cif_for_gsas',lambda source,cache:source)
    class Project:
        filename=str(tmp_path/'work.gpx')
        def add_phase(self,p,**kwargs):
            assert Path(p).read_bytes()==raw
            return SimpleNamespace(name='synthetic',data={
                'General':{'NoAtoms':{'C':0.6,'Li':0.40066},'AtomPtrs':[3,1,7,9],
                    'SGData':{'SpGrp':'P 1','SGInv':False,'SGCen':[[0,0,0]],
                              'SGOps':[([[1,0,0],[0,1,0],[0,0,1]],[0,0,0])]}},
                'Atoms':[['C1','C','',0,0,0,.6,'1',1],['Li1','Li','',0,0,0,.40066,'1',1]]})
    phase=add_verified_phase(Project(),path,phasename='synthetic')
    assert phase.name=='synthetic'
    native_review=gsas_cif_import._occupancy_review_module()
    assert phase.data[native_review.DATA_KEY]['review']['policy_sha256']
    assert native_review.validate_receipt(phase)['reviewed_groups']==[['C1','Li1']]


def test_strict_default_must_reject_negative_raw_component(tmp_path):
    path,_=source(tmp_path,negative=True)
    with pytest.raises(ValueError,match='occupan'):
        mapped_loader(path).load_structure('synthetic')


@pytest.mark.parametrize('changed',['normalized','redistributed'])
def test_native_review_rejects_equal_ratio_or_equal_total_occupancy_changes(tmp_path,monkeypatch,changed):
    path,raw=source(tmp_path,'0.40066')
    activate_test_review(tmp_path,path,monkeypatch)
    import gsas_cif_import
    monkeypatch.setattr(gsas_cif_import,'prepare_cif_for_gsas',lambda source,cache:source)
    class Project:
        filename=str(tmp_path/'work.gpx')
        def add_phase(self,*args,**kwargs):
            counts={'C':.6,'Li':.40066}
            atoms=[['C1','C','',0,0,0,.6],['Li1','Li','',0,0,0,.40066]]
            if changed=='normalized':counts={key:value/1.00066 for key,value in counts.items()}
            else:atoms[0][6],atoms[1][6]=.40066,.6
            return SimpleNamespace(name='synthetic',data={'General':{'NoAtoms':counts,'AtomPtrs':[3,1,7,9]},'Atoms':atoms})
    with pytest.raises(ValueError,match='reviewed atom counts|reviewed site occupancies'):
        add_verified_phase(Project(),path,phasename='synthetic')


def test_reviewed_import_receipt_cannot_bind_changed_source_bytes(tmp_path, monkeypatch):
    path, raw = source(tmp_path, '0.40066')
    activate_test_review(tmp_path, path, monkeypatch)
    import gsas_cif_import
    monkeypatch.setattr(gsas_cif_import, 'prepare_cif_for_gsas', lambda source, cache: source)
    class Project:
        filename = str(tmp_path / 'work.gpx')
        def add_phase(self, *args, **kwargs):
            path.write_bytes(raw + b'\n# changed during import\n')
            return SimpleNamespace()
    with pytest.raises(ValueError, match='CIF changed during verified import'):
        add_verified_phase(Project(), path, phasename='synthetic')
