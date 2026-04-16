import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import gsas_core_infrastructure as gci


class _FakeHistogram:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getHistEntryValue(self, keys):
        node = self.data
        for key in keys:
            node = node[key]
        return node


class _FakeProject:
    def __init__(self):
        self.data = {}
        self.names = [["Notebook"], ["Phases"]]

    def histograms(self):
        return [_FakeHistogram(name, data) for name, data in self.data.items() if name.startswith("PWDR ")]

    def histogram(self, name):
        return _FakeHistogram(name, self.data[name])

    def update_ids(self):
        return None


class _FakeReader:
    def __init__(self, name, ext_flag, contents_ok, powderdata, inst_type="PXC", validator_error="", read_error=""):
        self.formatName = name
        self.longFormatName = name
        self._ext_flag = ext_flag
        self._contents_ok = contents_ok
        self._powderdata = powderdata
        self._inst_type = inst_type
        self._validator_error = validator_error
        self._read_error = read_error
        self.read_calls = 0
        self.contents_calls = 0
        self.errors = ""
        self.powderentry = ["", None, 1]
        self.pwdparms = {}
        self.Sample = {}
        self.comments = []
        self.repeat = False
        self.idstring = name.replace(" ", "_")

    def __deepcopy__(self, memo):
        return self

    def ExtensionValidator(self, filename):
        return self._ext_flag

    def ReInitialize(self):
        self.errors = ""
        self.repeat = False

    def ContentsValidator(self, filename):
        self.contents_calls += 1
        self.errors = self._validator_error
        return self._contents_ok

    def Reader(self, filename, buffer=None, blocknum=1):
        self.read_calls += 1
        if self._read_error:
            self.errors = self._read_error
            return False
        x, y, w = self._powderdata
        self.powderdata = [
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(w, dtype=float),
            np.zeros(len(x)),
            np.zeros(len(x)),
            np.zeros(len(x)),
        ]
        self.powderentry[0] = filename
        self.Sample = {"Temperature": 300.0}
        self.comments = []
        return True


class _FakeG2sc:
    def __init__(self, readers):
        self.Readers = {"Pwdr": readers}

    def LoadG2fil(self):
        return None

    @staticmethod
    def load_pwd_from_reader(reader, instprm_file, existingnames=None, bank=None):
        histname = f"PWDR {reader.idstring}"
        inst_type = reader._inst_type
        if "TOF" in inst_type:
            inst = {"Type": [inst_type], "difC": [0.0, 5200.0, False], "difA": [0.0, 0.0, False], "difB": [0.0, 0.0, False], "Zero": [0.0, 0.0, False]}
        else:
            inst = {"Type": [inst_type], "Lam": [1.5406, 1.5406, False], "Zero": [0.0, 0.0, False]}
        pwdrdata = {
            "Comments": [],
            "Limits": [(float(reader.powderdata[0][0]), float(reader.powderdata[0][-1])), [float(reader.powderdata[0][0]), float(reader.powderdata[0][-1])]],
            "Background": [['chebyschev-1', False, 3, 1.0, 0.0, 0.0], {'nDebye': 0, 'debyeTerms': [], 'nPeaks': 0, 'peaksList': [], 'background PWDR': ['', 1.0, False]}],
            "Instrument Parameters": [inst, {}],
            "Sample Parameters": {"Temperature": 300.0},
            "Peak List": {"peaks": [], "sigDict": {}},
            "Index Peak List": [[], []],
            "Unit Cells List": [],
            "Reflection Lists": {},
            "data": [
                {"wtFactor": 1.0, "Dummy": False, "ranId": 1, "Offset": [0.0, 0.0], "delOffset": 0.0, "refOffset": 0.0, "refDelt": 0.0, "Yminmax": [float(np.min(reader.powderdata[1])), float(np.max(reader.powderdata[1]))]},
                reader.powderdata,
                histname,
            ],
        }
        new_names = [histname, "Comments", "Limits", "Background", "Instrument Parameters", "Sample Parameters", "Peak List", "Index Peak List", "Unit Cells List", "Reflection Lists"]
        return histname, new_names, pwdrdata


class HistogramImportTests(unittest.TestCase):
    def _make_manager(self):
        manager = gci.GSASProjectManager("/tmp/test_import")
        manager.project = _FakeProject()
        return manager

    def test_skips_readers_rejected_by_contents_validator(self):
        bad = _FakeReader(
            "Bad Primary",
            ext_flag=True,
            contents_ok=False,
            powderdata=([1, 2, 3], [1, 2, 3], [1, 1, 1]),
            validator_error="validator says no",
        )
        good = _FakeReader(
            "Topas xye/qye or 2th Fit2D chi/qchi",
            ext_flag=None,
            contents_ok=True,
            powderdata=(np.linspace(5.0, 50.0, 64), np.linspace(10.0, 100.0, 64), np.ones(64)),
        )
        fake_g2 = _FakeG2sc([bad, good])
        manager = self._make_manager()

        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
            tmp.write("5 10\n6 12\n7 14\n8 16\n")
            path = tmp.name

        with patch.object(gci, "G2sc", fake_g2), patch.object(gci, "GSAS_AVAILABLE", True):
            ok = manager.add_histogram(path, "cw.instprm", instrument_type="CW")

        self.assertTrue(ok)
        self.assertEqual(bad.read_calls, 0)
        self.assertEqual(good.read_calls, 1)

    def test_prefers_xye_reader_for_two_column_dat_files(self):
        csv = _FakeReader(
            "comma/tab/semicolon separated",
            ext_flag=False,
            contents_ok=True,
            powderdata=(np.linspace(5.0, 50.0, 64), np.linspace(10.0, 100.0, 64), np.ones(64)),
        )
        xye = _FakeReader(
            "Topas xye/qye or 2th Fit2D chi/qchi",
            ext_flag=None,
            contents_ok=True,
            powderdata=(np.linspace(5.0, 50.0, 64), np.linspace(10.0, 100.0, 64), np.ones(64)),
        )
        fake_g2 = _FakeG2sc([csv, xye])
        manager = self._make_manager()

        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
            for i in range(20):
                tmp.write(f"{5.0 + i * 0.05:.3f} {100 + i:.3f}\n")
            path = tmp.name

        with patch.object(gci, "G2sc", fake_g2), patch.object(gci, "GSAS_AVAILABLE", True):
            ok = manager.add_histogram(path, "cw.instprm", instrument_type="CW")

        self.assertTrue(ok)
        self.assertEqual(xye.read_calls, 1)
        self.assertEqual(csv.read_calls, 0)

    def test_rejects_unhealthy_histogram_and_tries_next_reader(self):
        bad = _FakeReader(
            "Topas xye/qye or 2th Fit2D chi/qchi",
            ext_flag=None,
            contents_ok=True,
            powderdata=([5.0, 4.0, 3.0, 2.0], [10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0]),
        )
        good = _FakeReader(
            "Rigaku .txt exported",
            ext_flag=None,
            contents_ok=True,
            powderdata=(np.linspace(5.0, 50.0, 64), np.linspace(10.0, 100.0, 64), np.ones(64)),
        )
        fake_g2 = _FakeG2sc([bad, good])
        manager = self._make_manager()

        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp:
            for i in range(20):
                tmp.write(f"{5.0 + i * 0.05:.3f} {100 + i:.3f}\n")
            path = tmp.name

        with patch.object(gci, "G2sc", fake_g2), patch.object(gci, "GSAS_AVAILABLE", True):
            ok = manager.add_histogram(path, "cw.instprm", instrument_type="CW")

        self.assertTrue(ok)
        self.assertGreaterEqual(bad.read_calls, 1)
        self.assertEqual(good.read_calls, 1)


if __name__ == "__main__":
    unittest.main()
