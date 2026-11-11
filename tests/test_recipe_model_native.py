"""Native public GPXs: cell changes transfer; added phases do not."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nova/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gpx_recipe.compatibility import capture, assess


class NativeRecipeModelTest(unittest.TestCase):
    def test_public_cw_and_tof_model_boundaries(self):
        try:
            G = importlib.import_module('GSASII.GSASIIscriptable')
        except ImportError:
            if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE') == '1':
                raise
            self.skipTest('GSAS-II native runtime is required')
        for kind in ('PNC','PNT'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root=Path(folder); path=root/'source.gpx'
                profile=root/'profile.instprm'
                parameters=('Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\nX:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n' if kind=='PNC' else
                    'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\nbeta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n')
                profile.write_text('#GSAS-II instrument parameter file\nType:'+kind+'\nBank:1\n'+parameters)
                project=G.G2Project(newgpx=str(path))
                phase=project.add_phase(phasename='Silicon',spacegroup='F d -3 m',cell=[5.431,5.431,5.431,90,90,90])
                phase.add_atom(.125,.125,.125,'Si','Si1')
                project.add_simulated_powder_histogram('Public boundary control',str(profile),
                    *((12.,125.) if kind=='PNC' else (3.,40.)),Npoints=301,scale=100.,phases=[phase])
                project.save()
                sha=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
                original=sha(path); ref=capture(path,expected_sha256=original)
                moved=root/'moved.gpx';shutil.copy2(path,moved)
                project=G.G2Project(gpxfile=str(moved))
                project.phases()[0].data['General']['Cell'][1:4]=[5.44,5.44,5.44]
                project.save()
                value=capture(moved,expected_sha256=sha(moved))
                report=assess(dict(source_sha256=sha(moved),project={'recipe_model':value}),ref)
                self.assertEqual(report['status'],'compatible')
                added=root/'with-container.gpx';shutil.copy2(moved,added)
                project=G.G2Project(gpxfile=str(added))
                aluminium=project.add_phase(phasename='Container Al',spacegroup='F m -3 m',cell=[4.05,4.05,4.05,90,90,90])
                aluminium.add_atom(0.,0.,0.,'Al','Al1');project.save()
                value=capture(added,expected_sha256=sha(added))
                report=assess(dict(source_sha256=sha(added),project={'recipe_model':value}),ref)
                self.assertEqual(report['status'],'incompatible')
                self.assertIn('Container Al',report['reason'])
                self.assertEqual(sha(path),original)
                # Exercise Analyze publication and NOVA's exact indexed-source
                # resolver, not just the standalone descriptor functions.
                from ndip_outputs import collect_outputs
                from radar_pd_nova.refinement_sources import source_from_record
                analysis=root/'analysis';final=analysis/'Technical/GSAS_Projects/control_seq_final_accepted_model.gpx'
                final.parent.mkdir(parents=True);shutil.copy2(path,final)
                portal=root/'portal';summary=collect_outputs(analysis,portal,mode='full',run_name='public control')
                index=json.loads((portal/'gpx_index.json').read_text())
                record=SimpleNamespace(local_output_dir=portal,output_dataset_ids={},uid='public-control',
                                       galaxy_job_id=None,name='public-control')
                loaded=source_from_record(record,dict(summary=summary,gpx_index=index),None,'public-history')
                published=loaded['project']['recipe_model']
                self.assertEqual(published['source_sha256'],loaded['source_sha256'])
                self.assertEqual(assess(loaded,published)['status'],'compatible')


if __name__=='__main__':
    unittest.main()
