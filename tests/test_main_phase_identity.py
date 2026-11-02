"""Known-main exclusion must survive charge-label normalization, not chemistry changes."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species, get_el_sp

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from main_phase_identity import main_phase_comparison_pair
from aniso_db_loader import build_mask


def mixed_structure(charged=True):
    fe, mg, oxygen = ('Fe3+', 'Mg2+', 'O2-') if charged else ('Fe', 'Mg', 'O')
    return Structure(Lattice.orthorhombic(8, 9, 11),
                     [{fe: .79, mg: .21}, {fe: .61, mg: .39}, oxygen, oxygen],
                     [[.03,.12,.24], [.45,.61,.78], [.73,.22,.53], [.32,.86,.08]])


def matches(a, b):
    pair = main_phase_comparison_pair(a, b)
    return pair is not None and StructureMatcher(
        primitive_cell=True, scale=True, attempt_supercell=False).fit(*pair)


class MainPhaseIdentityTests(unittest.TestCase):
    def test_ionic_and_neutral_same_mixed_sites_match_without_mutation(self):
        a, b = mixed_structure(), mixed_structure(False)
        before = json.dumps([a.as_dict(), b.as_dict()], sort_keys=True)
        self.assertTrue(matches(a, b))
        self.assertEqual(before, json.dumps([a.as_dict(), b.as_dict()], sort_keys=True))

    def test_changed_total_occupancy_is_not_equivalent(self):
        a, b = mixed_structure(), mixed_structure(False)
        b.replace(0, {'Fe':.78, 'Mg':.22})
        self.assertIsNone(main_phase_comparison_pair(a,b))

    def test_same_global_composition_with_different_site_fractions_is_not_equivalent(self):
        a, b = mixed_structure(), mixed_structure(False)
        b.replace(0, {'Fe':.78, 'Mg':.22})
        b.replace(1, {'Fe':.62, 'Mg':.38})
        self.assertFalse(matches(a,b))

    def test_changed_positions_are_not_equivalent(self):
        a, b = mixed_structure(), mixed_structure(False)
        b.translate_sites([1], [.45,.35,.35], frac_coords=True)
        b.translate_sites([2], [.15,.45,.35], frac_coords=True)
        self.assertFalse(matches(a,b))

    def test_polymorph_lattice_shape_is_not_equivalent(self):
        a = mixed_structure()
        b = Structure(Lattice.orthorhombic(3, 5, 24),
                      [site.species for site in a], a.frac_coords)
        self.assertFalse(matches(a,b))

    def test_matching_preserves_existing_uniform_cell_scale_tolerance(self):
        a, b = mixed_structure(), mixed_structure(False)
        b.scale_lattice(a.volume*1.01)
        self.assertTrue(matches(a,b))

    def test_hydrogen_isotopes_do_not_match_each_other(self):
        for a, b in [('H','D'), ('H','T'), ('D','T')]:
            with self.subTest(a=a,b=b):
                first = Structure(Lattice.cubic(4), [get_el_sp(a),'O'], [[0,0,0],[.31,.42,.19]])
                second = Structure(first.lattice, [get_el_sp(b),'O'], first.frac_coords)
                self.assertIsNone(main_phase_comparison_pair(first,second))

    def test_isotope_charge_normalization_keeps_mixed_hydrogen_and_deuterium_distinct(self):
        a = Structure(Lattice.orthorhombic(4,5,7),
                      [{get_el_sp('H1+'):.4, get_el_sp('D1+'):.6}, 'O2-'],
                      [[.1,.2,.3],[.62,.44,.77]])
        b = Structure(a.lattice, [{get_el_sp('H'):.4,get_el_sp('D'):.6},'O'], a.frac_coords)
        self.assertTrue(matches(a,b))
        pair = main_phase_comparison_pair(a,b)
        self.assertEqual(len(pair[0][0].species),2)
        self.assertEqual(sorted(pair[0][0].species.values()),[.4,.6])
        b.replace(0,{get_el_sp('H'):.6,get_el_sp('D'):.4})
        self.assertFalse(matches(a,b))

    def test_swapped_isotope_sites_do_not_merge_during_primitive_reduction(self):
        a = Structure(Lattice.orthorhombic(4,7,9),
                      ['H',get_el_sp('D'),'O','Li'],
                      [[.1,.2,.3],[.45,.61,.78],[.73,.22,.53],[.32,.86,.08]])
        b = a.copy(); b.replace(0,get_el_sp('D')); b.replace(1,'H')
        self.assertFalse(matches(a,b))

    def test_explicit_spin_labels_are_not_erased(self):
        a = Structure(Lattice.cubic(4), [Species('Fe',3,spin=1),'O2-'], [[0,0,0],[.3,.4,.2]])
        b = Structure(a.lattice, [Species('Fe',0,spin=1),'O'], a.frac_coords)
        self.assertTrue(matches(a,b))
        b.replace(0, Species('Fe',0,spin=-1))
        self.assertFalse(matches(a,b))


class MainPhasePipelineExclusionTests(unittest.TestCase):
    def _run(self, reference, candidate, candidate_sg=1):
        import gsas_complete_pipeline_nomain as pipeline
        elements = {sp.symbol for site in reference for sp in site.species}
        hi, lo = build_mask(elements)
        catalog = pd.DataFrame([dict(id='candidate', elements_mask_hi=hi,
                                     elements_mask_lo=lo, space_group=candidate_sg)])
        loader = SimpleNamespace(catalog=catalog, load_structure=lambda pid:candidate)
        pipe = pipeline.UnifiedPipeline.__new__(pipeline.UnifiedPipeline)
        pipe.db_loader = loader; pipe._main_phase_match_cache = {}
        with patch('cif_occupancy.load_authorized_structure',return_value=reference), \
             patch('cif_occupancy.cache_identity',return_value='source-key'), \
             patch.object(pipeline,'_parse_cif_metadata',return_value=('known','P 1 (1)')):
            return pipe._matching_db_ids_for_main_phase('main.cif')

    def test_charged_main_matches_neutral_catalog_entry(self):
        self.assertEqual(self._run(mixed_structure(),mixed_structure(False)),{'candidate'})

    def test_different_declared_space_group_is_not_excluded(self):
        self.assertEqual(self._run(mixed_structure(),mixed_structure(False),225),set())

    def test_different_site_occupancy_is_not_excluded(self):
        candidate = mixed_structure(False)
        candidate.replace(0,{'Fe':.78,'Mg':.22})
        self.assertEqual(self._run(mixed_structure(),candidate),set())

    def test_same_base_element_masks_do_not_allow_hydrogen_isotope_merge(self):
        first = Structure(Lattice.cubic(4),['H','O'],[[0,0,0],[.3,.4,.2]])
        second = Structure(first.lattice,[get_el_sp('D'),'O'],first.frac_coords)
        self.assertEqual(self._run(first,second),set())


if __name__ == '__main__':
    unittest.main()
