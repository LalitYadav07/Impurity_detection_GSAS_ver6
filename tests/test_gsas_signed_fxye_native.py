"""Check signed SNAP data through the actual scientific-runtime importer."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('GSASII'), 'requires the GSAS-II scientific runtime')
class SignedSnapImportTest(unittest.TestCase):
    def test_native_density_conversion_preserves_signed_zero_and_masked_bins(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            x=1000*np.exp(.01*np.arange(40));y=np.full(40,5.);sigma=np.full(40,2.)
            y[15]=-3.25;y[20]=0.;sigma[20]=1.;y[10]=np.nan;sigma[10]=0.
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'SNAP'}
            profile=root/'profile.instprm'
            profile.write_text('#Bank 1: GSAS-II instrument parameter file\nType:PNT\nBank:1\n'
                'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nalpha:.99\n'
                'beta-0:.02\nbeta-1:.03\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\n')
            data=root/'signed.gsa'
            data.write_text('Sample Run: 42\n# RADAR-PD input provenance: '+json.dumps(marker)+
                '\nBANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+
                ''.join(f'{a:.15g} {b:.15g} {c:.15g}\n' for a,b,c in zip(x,y,sigma)))
            original=sha256(data.read_bytes()).hexdigest()
            manager=GSASProjectManager(str(root),'signed-import')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            histogram=manager.main_histogram
            actual_y=np.asarray(histogram.getdata('yobs'));actual_w=np.asarray(histogram.getdata('yweight'))
            raw=np.loadtxt(data,skiprows=3);widths=np.diff(raw[:,0]);valid=raw[:-1,2]>0
            expected_y=np.where(valid,raw[:-1,1],0.)/widths
            expected_w=np.zeros(39);expected_w[valid]=(widths[valid]/raw[:-1,2][valid])**2
            np.testing.assert_allclose(histogram.getdata('x'),raw[:-1,0]+widths/2,rtol=1e-12)
            np.testing.assert_allclose(actual_y,expected_y,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(actual_w,expected_w,rtol=1e-12,atol=1e-12)
            self.assertLess(actual_y[15],0);self.assertGreater(actual_w[15],0)
            self.assertEqual(actual_w[10],0)
            self.assertEqual(actual_y[20],0);self.assertGreater(actual_w[20],0)
            self.assertEqual(sha256(data.read_bytes()).hexdigest(),original)


if __name__=='__main__':
    unittest.main()
