"""Offline evidence retains measured failures; no comparison tab is exposed."""
import io
import shutil
import zipfile

import numpy as np
import pytest

from radar_pd_nova import planning_evidence as evidence

pytestmark = pytest.mark.skipif(not (evidence.ROOT / "manifest.json").is_file(),
    reason="Private measured development evidence is tested from an authorized external audit folder")


def test_complete_native_measurements_reproduce_the_archived_signed_failures():
    metadata, arrays, _, _ = evidence.load_evidence()
    assert metadata['independent_absolute_acceptance_cases'] == 0
    assert not metadata['target_scale_or_mass_fitted']
    assert not metadata['raw_count_rebinning_performed']
    assert sum(r['native_bins'] for r in metadata['records']) == 16994
    assert sum(r['peak_contrasts'] for r in metadata['records']) == 30
    # Frozen audit outcomes, independent of the viewer's metric labels.
    expected = {
        1: [-.26448232441889263, -.5871214012766162, -.15830367187469885],
        2: [-.13084035968351293, -.3135282583935879, -.11501189686623803],
        3: [-.054257297152102923, -.14426398607595037, -.04670040977770684],
    }
    for band, gold in expected.items():
        data = evidence.comparison(metadata, arrays, band)
        np.testing.assert_allclose(list(data['metrics'].values()), gold, rtol=0, atol=2e-13)
        assert np.count_nonzero(data['sample'] < 0) == data['record']['negative_can_subtracted_bins'] > 0
        figure = evidence.comparison_figure(data, 'sample')
        np.testing.assert_array_equal(figure['data'][0]['y'], data['sample'])
        assert len(figure['data'][0]['x']) == data['record']['native_bins']


@pytest.mark.parametrize('filename', ['manifest.json', 'arrays.npz'])
def test_changed_evidence_cannot_be_viewed_or_exported(tmp_path, filename):
    for name in ('manifest.json', 'arrays.npz'):
        shutil.copyfile(evidence.ROOT / name, tmp_path / name)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b' changed')
    with pytest.raises(ValueError, match='changed'):
        evidence.load_evidence(tmp_path)
    with pytest.raises(ValueError, match='changed'):
        evidence.export_evidence(tmp_path)


def test_download_retains_every_integer_count_bin_and_signed_subtraction():
    metadata, arrays, manifest_bytes, array_bytes = evidence.load_evidence()
    with zipfile.ZipFile(io.BytesIO(evidence.export_evidence())) as archive:
        assert archive.read('manifest.json') == manifest_bytes
        assert archive.read('arrays.npz') == array_bytes
        assert 'not full predictive intervals' in archive.read('README.txt').decode()
        for row in metadata['records']:
            band = row['band']
            csv = np.genfromtxt(io.BytesIO(archive.read(f'band{band}-native-bins.csv')), delimiter=',', names=True)
            assert len(csv) == row['native_bins']
            np.testing.assert_array_equal(csv['measured_sample_counts'], arrays[f'band{band}_sample_counts'])
            np.testing.assert_array_equal(csv['measured_can_counts'], arrays[f'band{band}_can_counts'])
            measured = arrays[f'band{band}_sample_counts'] / row['charge_uah'] - arrays[f'band{band}_can_counts'] / row['can_charge_uah']
            np.testing.assert_array_equal(csv['measured_sample_minus_can_counts_per_uah_bin'], measured)
