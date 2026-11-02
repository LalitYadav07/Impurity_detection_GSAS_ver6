"""Check staged signed TOF data through the actual production import path."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import copy

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('GSASII'), 'requires the GSAS-II scientific runtime')
class SignedSnapImportTest(unittest.TestCase):
    @staticmethod
    def profile(root, bank=1, kind='PNT'):
        profile=root/f'profile-{bank}-{kind}.instprm'
        if kind=='PNT':
            values='difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nalpha:.99\n' \
                   'beta-0:.02\nbeta-1:.03\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\n'
        else:
            values='Lam:1.54\nU:.1\nV:0\nW:.1\nZero:0\nX:0\nY:0\n'
        profile.write_text(f'#Bank {bank}: GSAS-II instrument parameter file\nType:{kind}\nBank:{bank}\n'+values)
        return profile

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
        from GSASII import GSASIIscriptable as G2
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
            direct_data=root/'direct.gsa'
            direct_data.write_text('Sample Run: 42\nBANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+rows)
            direct=G2.G2Project(newgpx=str(root/'direct.gpx')).add_powder_histogram(str(direct_data),str(profile))
            native=[np.asarray(direct.getdata(key)).tobytes() for key in ('x','yobs','yweight')]
            for name,comment in [('native',''),('marked','# RADAR-PD input provenance: '+json.dumps(marker)+'\n')]:
                data=root/(name+'.gsa')
                data.write_text('Sample Run: 42\n'+comment+'BANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+rows)
                manager=GSASProjectManager(str(root),name)
                self.assertTrue(manager.create_project())
                self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
                values.append([np.asarray(manager.main_histogram.getdata(key)).tobytes() for key in ('x','yobs','yweight')])
            self.assertEqual(values[0],values[1])
            self.assertEqual(values[0],native)

    def test_unmarked_single_pattern_and_nonfacility_sources_keep_signed_density(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        for bank in [1,2,3,'001']:
          with self.subTest(bank=bank), tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,bank)
            x=1000*np.exp(.01*np.arange(40));y=np.full(40,5.);sigma=np.full(40,2.)
            y[15]=-3.25;y[20]=0.;sigma[20]=1.;y[10]=np.nan;sigma[10]=0.
            data=root/'unmarked.gsa'
            data.write_text('User supplied TOF source, no facility identity\n'+
                f'BANK {bank} 40 40 SLOG 1000 1500 .01 0 FXYE\n'+
                ''.join(f'{a:.15g} {b:.15g} {c:.15g}\n' for a,b,c in zip(x,y,sigma)))
            before=data.read_bytes()
            manager=GSASProjectManager(str(root),'unmarked')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            hist=manager.main_histogram;raw=np.loadtxt(data,skiprows=2);width=np.diff(raw[:,0])
            expected_y=np.where(raw[:-1,2]>0,raw[:-1,1],0.)/width
            expected_w=np.zeros(39);valid=raw[:-1,2]>0
            expected_w[valid]=(width[valid]/raw[:-1,2][valid])**2
            np.testing.assert_allclose(hist.getdata('yobs'),expected_y,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(hist.getdata('yweight'),expected_w,rtol=1e-12,atol=1e-12)
            self.assertGreater(hist.getdata('yweight')[20],0)
            self.assertEqual(hist.getdata('yweight')[10],0)
            self.assertEqual(data.read_bytes(),before)

    def test_actual_native_multibank_selection_uses_bank_label_not_position(self):
        from GSASII import GSASIIscriptable as G2
        from scripts.gsas_signed_fxye import read_staged_tof_source,restore_staged_tof_observations
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,5)
            data=root/'multi.gsa'
            data.write_text('User TOF\nBANK 2 3 3 SLOG 1000 1200 .01 0 FXYE\n1000 9 3\n1100 -4 2\n1200 0 1\n'
                            'BANK 5 3 3 SLOG 2000 2200 .01 0 FXYE\n2000 16 4\n2100 -7 3\n2200 0 2\n')
            G2.LoadG2fil()
            reader=copy.deepcopy(next(r for r in G2.Readers['Pwdr'] if r.formatName=='GSAS powder data'))
            self.assertTrue(reader.ContentsValidator(str(data)))
            reader.selections=[1]
            self.assertTrue(reader.Reader(str(data)))
            self.assertEqual(reader.powderentry[2],5)
            self.assertEqual(reader.numbanks,2)
            params=G2.load_iprms(str(profile),reader)
            self.assertTrue(restore_staged_tof_observations(reader,read_staged_tof_source(data),params))
            np.testing.assert_array_equal(reader.powderdata[0],[20.,21.,22.])
            np.testing.assert_array_equal(reader.powderdata[1],[16.,-7.,0.])
            np.testing.assert_array_equal(reader.powderdata[2],[1./16,1./9,1./4])

    def test_unmarked_constant_wavelength_retains_native_reader_semantics(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        from GSASII import GSASIIscriptable as G2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,kind='PXC');data=root/'cw.gsa'
            data.write_text('Unmarked CW\nBANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+
                            ''.join(f'{1000+10*i} {(-3 if i==15 else 0 if i==20 else 5)} 2\n' for i in range(40)))
            native=G2.G2Project(newgpx=str(root/'native.gpx')).add_powder_histogram(str(data),str(profile))
            manager=GSASProjectManager(str(root),'cw')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='CW',fmthint='gsas'))
            for key in ['x','yobs','yweight']:
                self.assertEqual(np.asarray(native.getdata(key)).tobytes(),np.asarray(manager.main_histogram.getdata(key)).tobytes())

    def test_claimed_acquisition_does_not_assign_native_sample_metadata(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root)
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'unverified',
                    'acquisition':{'temperature':10,'sample_id':'forged'}}
            data=root/'unverified.gsa'
            data.write_text('Unknown specimen\n# RADAR-PD input provenance: '+json.dumps(marker)+
                            '\nBANK 1 40 40 SLOG 1000 1500 .01 0 FXYE\n'+
                            ''.join(f'{1000+10*i} {(-3 if i==15 else 0 if i==20 else 5)} 2\n' for i in range(40)))
            manager=GSASProjectManager(str(root),'unknown')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            self.assertLess(manager.main_histogram.getdata('yobs')[15],0)
            self.assertEqual(manager.main_histogram.data['Sample Parameters']['Temperature'],300.)

    def test_unsupported_signed_tof_bank_does_not_fall_back_to_another_reader(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root);data=root/'unsupported.gsa'
            data.write_text('Unmarked TOF\nBANK 1 40 40 CONST 1000 10 0 0 FXYE\n'+
                            ''.join(f'{1000+10*i} {(-3 if i==15 else 5)} 2\n' for i in range(40)))
            manager=GSASProjectManager(str(root),'unsupported')
            self.assertTrue(manager.create_project())
            with self.assertLogs('scripts.gsas_core_infrastructure',level='WARNING') as messages:
                self.assertFalse(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            self.assertTrue(any('requires supported SLOG/FXYE' in m for m in messages.output))
            self.assertEqual(manager.project.histograms(),[])

    def test_native_leading_zero_std_and_positive_const_contracts_are_unchanged(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        from GSASII import GSASIIscriptable as G2
        for kind in ['STD','FXYE']:
          with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root);data=root/'legacy.gsa'
            if kind=='STD':
                records=[''.join(f'{1:2d}{100+i:6d}' for i in range(j,j+10)) for j in range(0,40,10)]
                text='Legacy STD\nBANK 001 40 4 CONST 1000 10 0 0 STD\n'+'\n'.join(records)+'\n'
            else:
                text='Positive CONST\nBANK 001 40 40 CONST 1000 10 0 0 FXYE\n'+''.join(f'{1000+10*i} {100+i} 2\n' for i in range(40))
            data.write_text(text)
            native=G2.G2Project(newgpx=str(root/'native.gpx')).add_powder_histogram(str(data),str(profile))
            manager=GSASProjectManager(str(root),'legacy')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='TOF',fmthint='gsas'))
            for key in ['x','yobs','yweight']:
                self.assertEqual(np.asarray(native.getdata(key)).tobytes(),np.asarray(manager.main_histogram.getdata(key)).tobytes())

    def test_actual_cw_zero_negative_coordinates_and_native_counts_remain_unchanged(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        from GSASII import GSASIIscriptable as G2
        for first,declared in [(0,40),(-100,40),(-100,999)]:
          with self.subTest(first=first,declared=declared), tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,kind='PXC');data=root/'cw.gsa'
            data.write_text(f'Native CW\nBANK 001 {declared} {declared} CONST {first} 10 0 0 FXYE\n'+
                            ''.join(f'{first+10*i} {(-3 if i==15 else 5)} 2\n' for i in range(40)))
            native=G2.G2Project(newgpx=str(root/'native.gpx')).add_powder_histogram(str(data),str(profile))
            manager=GSASProjectManager(str(root),'cw')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='CW',fmthint='gsas'))
            for key in ['x','yobs','yweight']:
                self.assertEqual(np.asarray(native.getdata(key)).tobytes(),np.asarray(manager.main_histogram.getdata(key)).tobytes())

    def test_marked_hb2a_unbanked_xye_keeps_native_cw_import_and_masks(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        from GSASII import GSASIIscriptable as G2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,kind='PNC');data=root/'hb2a.xye'
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'HB2A',
                    'reduction_family':'HB2A individual reduced XYE','source_bank':1}
            data.write_text('# RADAR-PD input provenance: '+json.dumps(marker)+'\n# temperature = 2.5\n'+
                ''.join(f'{10+i*.1:.3f} {0 if i==15 else 20+i} {0 if i==15 else 2}\n' for i in range(40)))
            before=data.read_bytes()
            native=G2.G2Project(newgpx=str(root/'native.gpx')).add_powder_histogram(str(data),str(profile))
            manager=GSASProjectManager(str(root),'hb2a')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='CW',fmthint='xye'))
            for key in ['x','yobs','yweight']:
                expected=native.getdata(key);actual=manager.main_histogram.getdata(key)
                self.assertEqual(np.ma.getdata(expected).tobytes(),np.ma.getdata(actual).tobytes())
                np.testing.assert_array_equal(np.ma.getmaskarray(expected),np.ma.getmaskarray(actual))
            self.assertEqual(manager.main_histogram.data['Sample Parameters']['Temperature'],2.5)
            self.assertEqual(data.read_bytes(),before)

    def test_non_tof_marked_banked_cw_uses_its_native_profile(self):
        from scripts.gsas_core_infrastructure import GSASProjectManager
        from GSASII import GSASIIscriptable as G2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profile=self.profile(root,kind='PNC');data=root/'marked-cw.gsa'
            marker={'schema':'radar-pd-input-provenance/v1','instrument':'other',
                    'reduction_family':'constant wavelength'}
            data.write_text('# RADAR-PD input provenance: '+json.dumps(marker)+
                '\nBANK 001 40 40 CONST 0 10 0 0 FXYE\n'+
                ''.join(f'{10*i} {20+i} 2\n' for i in range(40)))
            native=G2.G2Project(newgpx=str(root/'native.gpx')).add_powder_histogram(str(data),str(profile))
            manager=GSASProjectManager(str(root),'cw')
            self.assertTrue(manager.create_project())
            self.assertTrue(manager.add_histogram(str(data),str(profile),instrument_type='CW',fmthint='gsas'))
            for key in ['x','yobs','yweight']:
                self.assertEqual(np.asarray(native.getdata(key)).tobytes(),np.asarray(manager.main_histogram.getdata(key)).tobytes())

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
