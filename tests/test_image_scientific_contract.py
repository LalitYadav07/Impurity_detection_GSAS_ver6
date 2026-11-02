"""Built-image scientific contracts using only the frozen runtime dependencies."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator as UpstreamND
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import get_el_sp

from cif_occupancy import load_structure,make_review,parse_structure,policy_hash,species_inventory
import database_catalog_builder as catalog
from lattice_nudger import NDSticksQ
from neutron_fingerprints import NeutronFingerprintCalculator,GENERATOR_VERSION

HEADER='''data_contract
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
ROUNDED=(HEADER+'C1 C .12 .23 .34 .6\nLi1 Li .12 .23 .34 .40066\n').encode()
MIXED=(HEADER+'H1 H .12 .23 .34 .4\nD1 D .12 .23 .34 .6\nO1 O .67 .51 .19 1\n').encode()


class BuiltImageScientificContract(unittest.TestCase):
    def test_unreviewed_excess_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'occupancy total'):
            parse_structure(ROUNDED)

    def test_reviewed_excess_retains_source_values(self):
        review=make_review(ROUNDED,.001,'Image contract: explicitly preserve rounded values')
        inventory=species_inventory(parse_structure(ROUNDED,review))
        self.assertAlmostEqual(inventory['C'],.6,places=12)
        self.assertAlmostEqual(inventory['Li'],.40066,places=12)
        self.assertAlmostEqual(sum(inventory.values()),1.00066,places=12)

    def test_forged_receipt_is_rejected_even_after_rehashing(self):
        review=make_review(ROUNDED,.001,'Image contract: source-bound review')
        review['composition']={'C':999,'Li':1}
        review['policy_sha256']=policy_hash(review)
        with self.assertRaisesRegex(ValueError,'evidence'):
            parse_structure(ROUNDED,review)

    def test_source_file_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.cif';path.write_bytes(ROUNDED)
            review=make_review(ROUNDED,.001,'Image contract: retain exact source')
            self.assertAlmostEqual(species_inventory(load_structure(path,review))['Li'],.40066,places=12)
            self.assertEqual(path.read_bytes(),ROUNDED)

    def test_natural_pattern_matches_installed_upstream(self):
        s=Structure(Lattice.cubic(4.2),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
        a=UpstreamND(wavelength=1.54).get_pattern(s,scaled=False,two_theta_range=(5,100))
        b=NeutronFingerprintCalculator(wavelength=1.54).get_pattern(s,scaled=False,two_theta_range=(5,100))
        self.assertTrue(np.array_equal(a.x,b.x) and np.array_equal(a.y,b.y))

    def test_hydrogen_and_deuterium_have_distinct_neutron_intensity(self):
        lat=Lattice.orthorhombic(4.11,5.23,6.37)
        a=Structure(lat,['H','O'],[[.12,.23,.34],[.67,.51,.19]])
        b=Structure(lat,['D','O'],a.frac_coords)
        calc=NeutronFingerprintCalculator(wavelength=1.54)
        h=calc.get_pattern(a,scaled=False,two_theta_range=(5,100))
        d=calc.get_pattern(b,scaled=False,two_theta_range=(5,100))
        self.assertEqual(len(h.x),len(d.x))
        families=lambda pattern: [sorted((tuple(row['hkl']),row['multiplicity']) for row in peak) for peak in pattern.hkls]
        self.assertEqual(families(h),families(d))
        np.testing.assert_array_equal(h.d_hkls,d.d_hkls)
        # Upstream natural-species angles use NumPy vectorized trig; the
        # isotope-preserving path uses scalar math. Their final bits can
        # differ across CPU/libm implementations. Require the same physical
        # reflections at far below the calculator's peak-merging tolerance.
        maximum_offset=float(np.max(np.abs(h.x-d.x))) if len(h.x) else 0.
        print(f'H/D geometry contract: {len(h.x)} identical reflection families; maximum angle offset {maximum_offset:.17g} degrees')
        np.testing.assert_allclose(h.x,d.x,rtol=0,atol=1e-10)
        self.assertFalse(np.allclose(h.y,d.y))

    def test_mixed_loader_reaches_actual_builder_and_nudger_hooks(self):
        self.assertIs(catalog.NDCalculator,NeutronFingerprintCalculator)
        self.assertIsInstance(NDSticksQ()._calc(),NeutronFingerprintCalculator)
        inventory=species_inventory(parse_structure(MIXED))
        self.assertEqual(inventory,{'O':1.,'H':.4,'D':.6})
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'phases').mkdir()
            row,error,_=catalog.process_one_phase('mixed',{'cif_content':MIXED.decode()},folder,
                5,100,50,False,False,False,False,'neutron',1.54)
            self.assertIsNone(error)
            explicit=Structure(Lattice.orthorhombic(4.11,5.23,6.37),
                [{get_el_sp('H'):.4,get_el_sp('D'):.6},'O'],[[.12,.23,.34],[.67,.51,.19]])
            _,_,_,expected=catalog.simulate_topM_peaks(explicit,5,100,50,'neutron',1.54)
            with np.load(Path(folder)/row['npz']) as data:
                np.testing.assert_allclose(data['I0'],expected,rtol=1e-6,atol=0)
                receipt=json.loads(str(data['fingerprint_provenance'].item()))
                self.assertEqual(receipt['generator']['generator'],GENERATOR_VERSION)
                self.assertEqual(receipt['source_sha256'],hashlib.sha256(MIXED).hexdigest())


if __name__=='__main__':unittest.main(verbosity=2)
