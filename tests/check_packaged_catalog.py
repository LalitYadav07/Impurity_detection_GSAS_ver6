"""Cloud-image contract using packaged data and a synthetic public CIF only."""
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ndip_catalog_bundle import verify_installed_catalog, require_runtime_catalog
from ndip_runner import _db_root_for, main as runner_main
from db_pack_builder import resolve_simulation_settings
from ratio_filter import _load_profiles64_metadata

def main():
    package = Path(os.environ['RADAR_PD_DB_ROOT'])
    receipt = verify_installed_catalog(package)
    assert receipt['version'] == os.environ['RADAR_PD_DATABASE_VERSION']
    for radiation in ('neutron', 'xray'):
        root = _db_root_for(radiation, None)
        assert root == package / ('database_' + radiation)
        require_runtime_catalog(root, radiation)
        profiles = _load_profiles64_metadata(str(root / 'profiles64'))
        assert profiles['profiles'].shape[0] > 0
        print('Verified packaged fingerprint grid', radiation, profiles['profiles'].shape)
    root = _db_root_for('neutron', None)
    with (root / 'catalog_deduplicated.csv').open() as stream:
        erbium = [r for r in csv.DictReader(stream) if str(r.get('pretty_formula', '')).replace(' ', '') == 'Er2O3']
    assert erbium, 'Packaged catalog lacks the Er2O3 validation candidate'
    print('Verified built-in Er2O3 entries', len(erbium))

    with TemporaryDirectory() as directory:
        folder = Path(directory)
        cif = folder / 'Synthetic_catalog_contract.cif'
        cif.write_text('''data_synthetic_catalog_contract
    _cell_length_a 4.137
    _cell_length_b 5.273
    _cell_length_c 6.491
    _cell_angle_alpha 74.31
    _cell_angle_beta 83.17
    _cell_angle_gamma 97.23
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
    Si1 Si .127 .238 .349 1
    Ti1 Ti .543 .654 .765 1
    O1 O .836 .417 .592 1
    ''')
        expected = resolve_simulation_settings('neutron', base_db_root=root).to_manifest_dict()
        default = resolve_simulation_settings('neutron').to_manifest_dict()
        for mode in ('mini', 'augmented'):
            output = folder / mode
            manifest = folder / (mode + '.json')
            assert runner_main(['build-library', '--cif', str(cif), '--library-mode', mode,
                         '--radiation', 'neutron', '--base-db-root', str(root),
                         '--output-root', str(output), '--manifest-output', str(manifest),
                         '--archive', str(folder / (mode + '.zip'))]) == 0
            result = json.loads(manifest.read_text())
            assert result['simulation'] == expected
            assert result['phase_count'] > 0 and not result['failures']
            if mode == 'augmented':
                assert result['n_base_phases'] > 0 and result['n_added_phases'] == 1
            print('Verified packaged library builder', mode, result['phase_count'])
        print('Mini settings with packaged base', json.dumps(expected, sort_keys=True))
        print('Mini settings without base', json.dumps(default, sort_keys=True))
    print('Packaged catalog and library entrypoint contracts passed')


if __name__ == "__main__":
    main()
