import unittest

from scripts.gsas_main_phase_refiner import (
    histogram_supports_light_instrument_calibration,
    pick_refinable_instrument_terms,
)


class _FakeHistogram:
    def __init__(self, inst_type="PXC", sample_type="Bragg-Brentano"):
        self._inst = {
            "Type": [inst_type, inst_type, False],
            "Zero": [0.0, 0.0, False],
            "U": [2.0, 2.0, False],
            "V": [-2.0, -2.0, False],
            "W": [5.0, 5.0, False],
            "Lam1": [1.5405, 1.5405, False],
        }
        self._sample = {"Type": sample_type}

    def getHistEntryValue(self, key):
        if key == ['Instrument Parameters']:
            return [self._inst]
        if key == ['Sample Parameters']:
            return self._sample
        raise KeyError(key)


class LightCalibrationHelperTests(unittest.TestCase):
    def test_pick_refinable_instrument_terms_keeps_supported_subset(self):
        inst = {
            "Zero": [0.0, 0.0, False],
            "U": [2.0, 2.0, False],
            "V": [-2.0, -2.0, False],
            "W": [5.0, 5.0, False],
            "Lam1": [1.5405, 1.5405, False],
        }

        chosen = pick_refinable_instrument_terms(inst, ["Zero", "U", "W", "Shift"])

        self.assertEqual(chosen, ("Zero", "U", "W"))

    def test_light_calibration_support_requires_bragg_brentano_pxrd(self):
        self.assertTrue(
            histogram_supports_light_instrument_calibration(_FakeHistogram("PXC", "Bragg-Brentano"))
        )
        self.assertFalse(
            histogram_supports_light_instrument_calibration(_FakeHistogram("PNC", "Debye-Scherrer"))
        )
        self.assertFalse(
            histogram_supports_light_instrument_calibration(_FakeHistogram("PXC", "Debye-Scherrer"))
        )


if __name__ == "__main__":
    unittest.main()
