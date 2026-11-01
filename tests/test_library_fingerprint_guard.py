"""Saved custom-library isotope validation; no GSAS or native job dispatch."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aniso_db_loader import _ORIG_DB_CACHE
from database_catalog_builder import process_one_phase
from db_pack import get_db_pack_layout
from db_pack_builder import resolve_simulation_settings, _build_profile_from_phase_npz, _write_profiles_bundle
from library_fingerprint_guard import validate_custom_neutron_library, LibraryFingerprintError, MAX_UNCACHED_ORIGINAL_BYTES
import ndip_runner


HEADER = '''data_control
_cell_length_a 4.11
_cell_length_b 5.23
_cell_length_c 6.37
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
'''


def pack(root, isotope='D', *, legacy=False):
    layout=get_db_pack_layout(root)
    layout.phases_dir.mkdir(parents=True);layout.cifs_dir.mkdir();layout.profiles_dir.mkdir()
    raw=HEADER+f'X1 {isotope} .12 .23 .34 1\nO1 O .67 .51 .19 1\n'
    (layout.cifs_dir/'p.cif').write_text(raw)
    settings=resolve_simulation_settings('neutron')
    row, failure, _=process_one_phase('p',{'cif_content':raw},str(root),
        settings.two_theta_min,settings.two_theta_max,settings.topM,False,False,False,False,
        'neutron',settings.wavelength)
    if failure:raise AssertionError(failure)
    pd.DataFrame([row]).to_csv(layout.catalog_csv,index=False)
    layout.cif_map_json.write_text(json.dumps({'p':'cifs/p.cif'}))
    layout.original_json.write_text(json.dumps({'p':{'cif_content':raw}}))
    profile=_build_profile_from_phase_npz(layout.phases_dir/'p.npz',settings)
    _write_profiles_bundle(layout,np.asarray([profile]),pd.DataFrame({'id':['p'],'row':[0]}),settings,['p'])
    layout.manifest_json.write_text(json.dumps({'kind':'mini','phase_ids':['p'],'simulation':settings.to_manifest_dict()}))
    if legacy:
        with np.load(layout.phases_dir/'p.npz') as source:
            old={key:source[key] for key in source.files if not key.startswith('fingerprint_')}
        np.savez_compressed(layout.phases_dir/'p.npz',**old)
    return layout


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


class LibraryFingerprintGuardContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_new_deuterium_and_tritium_phase_and_bundle_receipts_pass(self):
        for isotope in ('D','T'):
            with self.subTest(isotope=isotope):
                root=self.root/isotope;pack(root,isotope)
                before=hashes(root)
                result=validate_custom_neutron_library(root,{'allowed_elements':['H','O']})
                self.assertEqual(result['isotopes'],{'p':[isotope]})
                self.assertEqual(result['io']['phase_npz_reads'],1)
                self.assertEqual(result['io']['profile_bundle_reads'],1)
                self.assertEqual(hashes(root),before)

    def test_legacy_isotopes_fail_with_rebuild_action_and_no_file_changes(self):
        for isotope in ('D','T'):
            with self.subTest(isotope=isotope):
                root=self.root/isotope;pack(root,isotope,legacy=True)
                before=hashes(root)
                with self.assertRaisesRegex(LibraryFingerprintError,'Rebuild.*CIF Library Builder'):
                    validate_custom_neutron_library(root,{'allowed_elements':['H','O']})
                self.assertEqual(hashes(root),before)

    def test_legacy_natural_hydrogen_needs_no_generator_migration(self):
        root=self.root/'H';pack(root,'H',legacy=True)
        before=hashes(root)
        result=validate_custom_neutron_library(root,{'allowed_elements':['H','O']})
        self.assertEqual(result['checked_isotope_phase_ids'],[])
        self.assertEqual(result['io']['phase_npz_reads'],0)
        self.assertEqual(result['io']['profile_bundle_reads'],0)
        self.assertEqual(hashes(root),before)

    def test_local_cif_sources_do_not_require_optional_original_json(self):
        for species in ('Fe','H','D','T'):
            with self.subTest(species=species):
                root=self.root/species;layout=pack(root,species,legacy=species in {'Fe','H'})
                layout.original_json.unlink()
                before=hashes(root)
                result=validate_custom_neutron_library(root,{})
                self.assertEqual(result['io']['original_json_reads'],0)
                self.assertEqual(result['checked_isotope_phase_ids'],['p'] if species in {'D','T'} else [])
                self.assertEqual(hashes(root),before)

    def test_explicit_missing_original_json_is_an_actionable_error(self):
        root=self.root/'explicit';pack(root,'H',legacy=True)
        with self.assertRaisesRegex(LibraryFingerprintError,'configured original JSON file is unavailable'):
            validate_custom_neutron_library(root,{},original_json=root/'missing.json')

    def test_changed_source_or_missing_bundle_receipt_or_wrong_row_fails(self):
        for corruption in ('source','bundle','row','generator','index'):
            with self.subTest(corruption=corruption):
                root=self.root/corruption;layout=pack(root)
                if corruption=='source':
                    with (layout.cifs_dir/'p.cif').open('a') as stream:stream.write('# changed source\n')
                elif corruption=='index':
                    pd.DataFrame({'id':['p'],'row':[-1]}).to_csv(layout.profiles_index_csv,index=False)
                else:
                    path=layout.phases_dir/'p.npz' if corruption=='generator' else layout.profiles_npz
                    with np.load(path) as source: values={k:source[k] for k in source.files}
                    if corruption=='row':values['profiles']=np.zeros_like(values['profiles'])
                    else:
                        receipt=json.loads(str(values['fingerprint_provenance'].item()))
                        if corruption=='bundle':receipt['phases']={}
                        else:receipt['generator']['generator']='older-generator'
                        values['fingerprint_provenance']=np.asarray(json.dumps(receipt))
                    np.savez_compressed(path,**values)
                with self.assertRaises(LibraryFingerprintError):validate_custom_neutron_library(root,{})

    def test_uncached_large_base_metadata_is_not_read(self):
        root=self.root/'base';layout=pack(root)
        layout.cif_map_json.unlink()
        with layout.original_json.open('wb') as stream:stream.truncate(MAX_UNCACHED_ORIGINAL_BYTES+1)
        stats={}
        with self.assertRaisesRegex(LibraryFingerprintError,'source-access review'):
            validate_custom_neutron_library(root,{},io_stats=stats)
        self.assertEqual(stats['original_json_reads'],0)
        self.assertEqual(stats['original_json_bytes_read'],0)

    def test_existing_valid_large_metadata_cache_is_reused_for_natural_source(self):
        root=self.root/'cache';layout=pack(root,'H',legacy=True)
        raw=(layout.cifs_dir/'p.cif').read_text();layout.cif_map_json.unlink()
        with layout.original_json.open('wb') as stream:stream.truncate(MAX_UNCACHED_ORIGINAL_BYTES+1)
        stat=layout.original_json.stat();key=str(layout.original_json.resolve())
        with patch.dict(_ORIG_DB_CACHE,{key:((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns),{'p':{'cif_content':raw}})}):
            result=validate_custom_neutron_library(root,{})
        self.assertEqual(result['io']['original_json_reads'],0)
        self.assertEqual(result['checked_isotope_phase_ids'],[])

    def test_element_exclusion_avoids_source_and_fingerprint_reads(self):
        root=self.root/'excluded';layout=pack(root,legacy=True)
        (layout.cifs_dir/'p.cif').unlink()
        result=validate_custom_neutron_library(root,{'allowed_elements':['Fe']})
        self.assertEqual(result['io']['eligible_rows'],0)
        self.assertEqual(result['io']['cif_file_reads'],0)
        self.assertEqual(result['io']['phase_npz_reads'],0)

    def test_another_configured_ranking_database_is_not_silently_validated(self):
        root=self.root/'selected';pack(root,'H',legacy=True)
        config={'db':{'catalog_csv':str(self.root/'other/catalog_deduplicated.csv')}}
        with self.assertRaisesRegex(LibraryFingerprintError,'configured ranking database differs'):
            validate_custom_neutron_library(root,config)

    def test_alternate_source_paths_are_rejected_at_each_effective_config_scope(self):
        root=self.root/'selected';selected=pack(root,'D')
        other=pack(self.root/'other','H',legacy=True)
        base={'catalog_csv':str(selected.catalog_csv),'profiles_dir':str(selected.profiles_dir),
            'cif_map_json':str(selected.cif_map_json),'original_json':str(selected.original_json)}
        for key in ('cif_map_json','original_json'):
            changed={**base,key:str(getattr(other,key))}
            cases=[{'db':changed}, {'db':base,'datasets':[{'db':{key:changed[key]}}]},
                {'db':base,'datasets':[{key:changed[key]}]},
                {'db_source':'neutron','db_neutron':changed},
                {'db':base,'datasets':[{}, {'db':{key:changed[key]}}]}]
            for config in cases:
                for mode in ('full','rapid'):
                    with self.subTest(key=key,config=config,mode=mode):
                        with self.assertRaisesRegex(LibraryFingerprintError,'source contract'):
                            validate_custom_neutron_library(root,config,
                                original_json=selected.original_json,pipeline_mode=mode)

    def test_actual_original_only_source_is_used_instead_of_available_cif_map(self):
        for species in ('H','D'):
            with self.subTest(species=species):
                root=self.root/species;layout=pack(root,species,legacy=species=='H')
                config={'db':{'catalog_csv':str(layout.catalog_csv),'profiles_dir':str(layout.profiles_dir),
                    'original_json':str(layout.original_json)}}
                result=validate_custom_neutron_library(root,config,original_json=layout.original_json)
                self.assertEqual(result['source_bindings'],[{'cif_map_json':None,'original_json':str(layout.original_json.resolve())}])
                self.assertEqual(result['io']['cif_file_reads'],0)
                self.assertEqual(result['io']['original_json_reads'],1)
                self.assertEqual(result['checked_isotope_phase_ids'],['p'] if species=='D' else [])

    def test_matching_aliases_and_db_neutron_precedence_remain_valid(self):
        root=self.root/'aliases';layout=pack(root,'D')
        base={'catalog_csv':str(layout.catalog_csv),'profiles_dir':str(layout.profiles_dir),
            'cif_map_json':'${CONFIG_DIR}/folder/../cif_map.json','original_json':'./highsymm_metadata.json'}
        configs=[{'db':base,'CONFIG_DIR':str(root)},
            {'db_source':'neutron','db_neutron':base,'CONFIG_DIR':str(root)},
            {'db':base,'db_source':'neutron','db_neutron':{'original_json':'ignored'},'CONFIG_DIR':str(root)}]
        with patch.dict(os.environ,{'CONFIG_DIR':str(root)}):
            for config in configs:
                for mode in ('full','rapid'):
                    with self.subTest(mode=mode,config=config):
                        result=validate_custom_neutron_library(root,config,
                            original_json=layout.original_json,pipeline_mode=mode,config_directory=root)
                        self.assertEqual(result['isotopes'],{'p':['D']})

    def test_dataset_null_source_semantics_match_each_ranker(self):
        root=self.root/'null';layout=pack(root,'D')
        config={'db':{'catalog_csv':str(layout.catalog_csv),'profiles_dir':str(layout.profiles_dir),
            'cif_map_json':str(layout.cif_map_json),'original_json':str(layout.original_json)},
            'datasets':[{'cif_map_json':None}]}
        full=validate_custom_neutron_library(root,config,pipeline_mode='full')
        rapid=validate_custom_neutron_library(root,config,pipeline_mode='rapid')
        self.assertIsNone(full['source_bindings'][0]['cif_map_json'])
        self.assertEqual(rapid['source_bindings'][0]['cif_map_json'],str(layout.cif_map_json.resolve()))
        self.assertEqual(full['isotopes'],rapid['isotopes'])

    def test_legitimate_augmented_base_metadata_is_retained_without_relabeling(self):
        root=self.root/'augmented';layout=pack(root,'D')
        base=pack(self.root/'base','H',legacy=True)
        raw=(base.cifs_dir/'p.cif').read_text()
        base.original_json.write_text(json.dumps({'q':{'cif_content':raw}}))
        catalog=pd.read_csv(base.catalog_csv);catalog['id']='q';catalog['npz']='phases/q.npz'
        pd.concat([pd.read_csv(layout.catalog_csv),catalog],ignore_index=True).to_csv(layout.catalog_csv,index=False)
        shutil.copyfile(base.phases_dir/'p.npz',layout.phases_dir/'q.npz')
        with np.load(layout.profiles_npz) as value: bundle={key:value[key] for key in value.files}
        with np.load(base.profiles_npz) as value: bundle['profiles']=np.concatenate([bundle['profiles'],value['profiles']])
        np.savez_compressed(layout.profiles_npz,**bundle)
        pd.DataFrame({'id':['p','q'],'row':[0,1]}).to_csv(layout.profiles_index_csv,index=False)
        manifest=json.loads(layout.manifest_json.read_text());manifest['kind']='augmented'
        layout.manifest_json.write_text(json.dumps(manifest));layout.original_json.unlink()
        config={'db_source':'neutron','db_neutron':{'catalog_csv':str(layout.catalog_csv),
            'profiles_dir':str(layout.profiles_dir),'cif_map_json':str(layout.cif_map_json),
            'original_json':str(base.original_json)}}
        result=validate_custom_neutron_library(root,config,original_json=base.original_json)
        self.assertEqual(result['isotopes'],{'p':['D']})
        self.assertEqual(result['io']['original_json_reads'],1)
        with np.load(layout.profiles_npz) as value:
            self.assertEqual(set(json.loads(str(value['fingerprint_provenance'].item()))['phases']),{'p'})

    def test_both_runner_modes_fail_before_pipeline_dispatch(self):
        for mode in ('full','rapid'):
            with self.subTest(mode=mode):
                base=self.root/mode;base.mkdir();work=base/'work'
                root=work/'custom_database';pack(root,legacy=True)
                data=base/'data.gsa';data.write_text('1 2 3\n')
                inst=base/'profile.instprm';inst.write_text('test')
                config=base/'config.json';config.write_text(json.dumps({'$schema':ndip_runner.CONFIG_SCHEMA,
                    'analysis':{'mode':mode,'radiation':'neutron'}}))
                args=ndip_runner.build_parser().parse_args(['analyze','--mode',mode,'--config',str(config),
                    '--data',str(data),'--instrument',str(inst),'--db-pack',str(root),
                    '--work-dir',str(work),'--output-dir',str(base/'portal'),'--dry-run'])
                with patch.object(ndip_runner,'_resolve_instrument_mode',return_value=('tof','test')), \
                     patch.object(ndip_runner,'_materialize_contract_config',return_value={'allowed_elements':['H','O']}), \
                     patch.object(ndip_runner,'_pipeline_command') as dispatch:
                    with self.assertRaises(LibraryFingerprintError):ndip_runner.command_analyze(args)
                    dispatch.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
