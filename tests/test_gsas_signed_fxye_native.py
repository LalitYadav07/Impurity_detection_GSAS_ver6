"""Check staged signed TOF data through the actual production import path."""
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

        for instrument, family in [('NOM','NOMAD individual GSAS/SLOG'),
                                   ('PG3','POWGEN individual GSAS/FXYE'),
                                   ('SNAP','SNAP individual GSAS/SLOG')]:
          with self.subTest(instrument=instrument), tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            x=1000*np.exp(.01*np.arange(40));y=np.full(40,5.);sigma=np.full(40,2.)
            y[15]=-3.25;y[20]=0.;sigma[20]=1.;y[10]=np.nan;sigma[10]=0.
            marker={'schema':'radar-pd-input-provenance/v1','instrument':instrument,
                    'reduction_family':family,'invalid_count_bin_indices':[7]}
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
            expected_w[7]=0.
            np.testing.assert_allclose(histogram.getdata('x'),raw[:-1,0]+widths/2,rtol=1e-12)
            np.testing.assert_allclose(actual_y,expected_y,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(actual_w,expected_w,rtol=1e-12,atol=1e-12)
            self.assertLess(actual_y[15],0);self.assertGreater(actual_w[15],0)
            self.assertEqual(actual_w[10],0)
            self.assertEqual(actual_w[7],0)
            self.assertTrue(np.ma.getmaskarray(histogram.getdata('yweight'))[7])
            self.assertEqual(actual_y[20],0);self.assertGreater(actual_w[20],0)
            self.assertEqual(sha256(data.read_bytes()).hexdigest(),original)

    def test_native_positive_import_is_bit_stable(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            profile=root/'profile.instprm'
            profile.write_text('#Bank 1: GSAS-II instrument parameter file\nType:PNT\nBank:1\n'
                'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nalpha:.99\n'
                'beta-0:.02\nbeta-1:.03\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\n')
            x=1000*np.exp(.01*np.arange(40))
            rows=''.join(f'{t:.15g} {5+i/7:.15g} {2+i/13:.15g}\n' for i,t in enumerate(x))
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'NOM',
                    'reduction_family':'NOMAD individual GSAS/SLOG'}
            values=[]
            for name,comment in [('native',''),('marked','# RADAR-PD input provenance: '+json.dumps(marker)+'\n')]:
                data=root/(name+'.gsa')
                data.write_text('Sample Run: 42\n'+comment+'BANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+rows)
                manager=GSASProjectManager(str(root),name)
                self.assertTrue(manager.create_project())
                self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
                values.append([np.asarray(manager.main_histogram.getdata(key)).tobytes() for key in ('x','yobs','yweight')])
            self.assertEqual(values[0],values[1])

    def test_marked_tof_cannot_use_a_constant_wavelength_profile(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            profile=root/'cw.instprm'
            profile.write_text('#Bank 1: GSAS-II instrument parameter file\nType:PXC\nBank:1\n'
                               'Lam:1.54\nU:.1\nV:0\nW:.1\nZero:0\nX:0\nY:0\n')
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'NOM',
                    'reduction_family':'NOMAD individual GSAS/SLOG'}
            data=root/'marked.gsa'
            data.write_text('Sample Run: 42\n# RADAR-PD input provenance: '+json.dumps(marker)+
                '\nBANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+''.join(f'{1000+i*10} 5 2\n' for i in range(40)))
            manager=GSASProjectManager(str(root),'wrong-profile')
            self.assertTrue(manager.create_project())
            self.assertFalse(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            self.assertEqual(manager.project.histograms(),[])


if __name__=='__main__':
    unittest.main()
