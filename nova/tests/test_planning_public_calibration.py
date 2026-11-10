"""Public calibration provenance, immutable physics arrays and three-band use."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from gpx_recipe.planning import current_response as module
from gpx_recipe.planning.instrument import selected_profile
from gpx_recipe.planning.response import verify_calibration_asset


ARRAY_HASHES = {
    'calibration/pg3-53998/response.npz': 'c8f789e6b5d97095dd93c1ed9d8f787160058a2f4f5b4c57cee05a2d56e6cb12',
    'calibration/pg3-53999/response.npz': '7b3e8459d84562d38dff879da7bb214c65540a94a30bdf136623260fe11f9c6b',
    'calibration/pg3-54000/response.npz': '75bdbe14f3d75540ba61e6cf1792fa06d895a587440dd70074aedce82701d744',
    'current-2026b-hr-pac-b1/PG3_PAC_d63561_2026-08-4.h5': 'cc75eb0cfed744c78d5d0c7a3d596a100cb118b91b46dab416fe2a3f1a0e7a31',
    'current-2026b-hr-pac-b1/counts.npz': '9a18a75f04e35f8da48e0def1916a915e529e03f8288bac98feb96e3b012cff2',
    'current-2026b-hr-pac-b1/reduction.npz': '9f14f30dcf48cfd03d390ba8067fb052063a4766aabdba1a0f317740294586c4',
    'current-2026b-hr-pac-b2/PG3_PAC_d63561_2026-08-4.h5': 'cc75eb0cfed744c78d5d0c7a3d596a100cb118b91b46dab416fe2a3f1a0e7a31',
    'current-2026b-hr-pac-b2/counts.npz': 'cc8f8b680abc49bb229fb2a59e9fc3a27b8b07c03165d0dbe8de6184befa29b7',
    'current-2026b-hr-pac-b2/reduction.npz': 'b6e87c585614d8dafb650d015a9cdc8a905ff7817ec967a9e25e9d2e5cd2605f',
    'current-2026b-hr-pac-b2/standard-calibration.npz': 'ae229db00283fe7bc5834064e29af0fcc812bbcb1c2b272df5d6253aca095f96',
    'current-2026b-hr-pac-b3/PG3_PAC_d63561_2026-08-4.h5': 'cc75eb0cfed744c78d5d0c7a3d596a100cb118b91b46dab416fe2a3f1a0e7a31',
    'current-2026b-hr-pac-b3/counts.npz': 'bf6183963e70d2ec0bcaaa7fa6f5c8a24fc579319d7a38be84a824326c76c92c',
    'current-2026b-hr-pac-b3/reduction.npz': 'eff5c59472ba5bd53d9ecd913867353878ff91633b624f3959dd5c4ee19aa5ee',
}


def test_public_metadata_contains_no_local_paths_or_transport_details():
    root=module.DATA.parent
    files=list(root.rglob('*.json'))+[root.parent/'source-provenance.json']
    private=re.compile(r'/(?:Users|private|home|tmp|var|workspace|output|work|dev/shm)/|[A-Za-z]:\\|neutrons-dgx|stock_readonly_transport_commands|ControlPath')
    changed=0
    for path in files:
        text=path.read_text()
        assert not private.search(text),path
        value=json.loads(text)
        if 'public_metadata_provenance' in value:
            changed+=1
            assert re.fullmatch('[a-f0-9]{64}',value['public_metadata_provenance']['original_metadata_sha256'])
    assert changed==12


def test_calibration_array_files_are_identical_to_the_reviewed_inputs():
    root=module.DATA.parent
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.suffix in {'.npz','.h5'}}==set(ARRAY_HASHES)
    for name,expected in ARRAY_HASHES.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    for path in (root/'calibration').rglob('*'):
        if path.suffix in {'.json','.npz'} and path.name!='cross-sections.json':
            verify_calibration_asset(path,root)


def example_inputs(profile_id):
    config=selected_profile(profile_id)
    scalar={}
    for line in Path(config['profile']).read_text().splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1)
        try:scalar[key]=float(value)
        except ValueError:pass
    lower,upper={.8:(.3,3.1),1.5:(.55,7.24),2.665:(1.1,7.24)}[config['wavelength']]
    d=np.geomspace(lower*1.01,upper*.99,64)
    x=scalar['difC']*d+scalar['difA']*d*d+scalar['difB']/d+scalar['Zero']
    arrays=dict(d=d,x=x,nuclear=.01*np.exp(-((np.log(d)-np.log(np.sqrt(lower*upper)))/.1)**2),magnetic=np.zeros_like(d))
    metadata=dict(profile_sha256=config['sha256'],cell_mass_da=224.684,cell_volume_a3=160.16,
        composition={'Si':8.},tof_calibration=scalar)
    sample=dict(mass_g=.95,radius_cm=.295,packing_fraction=.3730165664701342,beam_width_cm=1.,beam_height_cm=4.)
    return config,metadata,arrays,sample


@pytest.mark.parametrize('profile_id',list(module.BUNDLES))
def test_public_metadata_keeps_three_band_count_and_shape_paths(profile_id):
    meta,observed=module.load_bundle(profile_id)
    assert meta['profile_id']==profile_id and meta['experimental_accuracy_certified'] is False
    assert observed['on_counts'].shape==observed['off_counts'].shape
    config,metadata,arrays,sample=example_inputs(profile_id)
    campaign,components,detail=module.current_response(config,metadata,arrays,sample,seed=9382)
    early=campaign.advance(25.);late=campaign.advance(125.)
    np.testing.assert_array_equal(early.counts+late.increment,late.counts)
    np.testing.assert_allclose(early.counting_variance*25.,late.counting_variance*125.,rtol=2e-14)
    np.testing.assert_array_equal(early.reference_variance,late.reference_variance)
    assert np.isfinite(components['nuclear']).all() and np.isfinite(late.sigma).all()
    assert np.all(late.counts>=0) and late.counts.dtype.kind=='i'
    assert detail['experimental_accuracy_certified'] is False


def test_sanitized_asset_is_still_checked_by_the_bundle_loader(tmp_path,monkeypatch):
    directory=tmp_path/'bundle';shutil.copytree(module.DATA,directory)
    monkeypatch.setitem(module.BUNDLES,module.DEFAULT_PROFILE,(directory,module.BUNDLE_SHA256))
    module.load_bundle()
    path=directory/'can-reference.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='asset is missing or changed'):module.load_bundle()
