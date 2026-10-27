"""Result publication must retain optional uncertainty fields on every phase."""
import csv
import json
import pickle
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ndip_outputs import collect_outputs


class HeterogeneousLatticeExport(unittest.TestCase):
    def test_fixed_and_refined_phase_rows_export_in_either_order(self):
        fixed=dict(phase_id='fixed-model',phase='Fixed model',space_group='P 1',
            source_gpx='Technical/GSAS_Projects/seq_final_joint_polished.gpx',
            a=4.0,a_esd=None,uncertainty_status='not_estimated')
        refined=dict(phase_id='refined-model',phase='Refined model',space_group='P 1',
            source_gpx=fixed['source_gpx'],a=4.1,a_esd=.001,uncertainty_status='estimated',
            uncertainty_method='J C J^T from final GSAS-II reciprocal-cell covariance',
            gsas_esd_comparison='different',gsas_api_cell_esds={'a':.002})
        for rows in ([fixed,refined],[refined,fixed]):
            with self.subTest(first=rows[0]['phase_id']),TemporaryDirectory() as folder:
                root=Path(folder);run=root/'run';gpx=run/fixed['source_gpx']
                gpx.parent.mkdir(parents=True)
                with gpx.open('wb') as stream:pickle.dump([['Phases',None]],stream,protocol=1)
                portal=root/'portal'
                with patch('ndip_lattice.extract_lattice',return_value=(rows,1.25)):
                    result=collect_outputs(run,portal,mode='full',include_archive=False)
                with (portal/'tables/Final_lattice_parameters.csv').open(newline='') as stream:
                    reader=csv.DictReader(stream);published=list(reader)
                    self.assertEqual(set(reader.fieldnames),set(fixed)|set(refined))
                by_id={row['phase_id']:row for row in published}
                self.assertEqual(by_id['fixed-model']['uncertainty_method'],'')
                self.assertEqual(by_id['refined-model']['uncertainty_method'],refined['uncertainty_method'])
                self.assertEqual(by_id['refined-model']['gsas_api_cell_esds'],str(refined['gsas_api_cell_esds']))
                self.assertEqual(by_id['refined-model']['a_esd'],'0.001')
                self.assertEqual(result['lattice_parameters'],rows)
                self.assertEqual(json.loads((portal/'summary.json').read_text())['lattice_parameters'],rows)


if __name__=='__main__':unittest.main()
