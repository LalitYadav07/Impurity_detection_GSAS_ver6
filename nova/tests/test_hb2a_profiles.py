"""The public mapping is exact, closed at both dates and checksum verified."""
from datetime import datetime,timedelta
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from radar_pd_nova import hb2a_profiles

CATALOG=json.loads((hb2a_profiles.ROOT/'catalog.json').read_text())


def setup(row, *, start=None, end=None):
    return SimpleNamespace(metadata={
        'start_time':(start or row['start_date'])+'T12:00:00-04:00',
        'end_time':(end or row['end_date'])+'T13:00:00-04:00',
        'ge_reflection':row['ge_reflection'],'collimator':row['collimator'],
        'nominal_wavelength_angstrom':{113:2.41,115:1.54,117:1.12}[row['ge_reflection']]},
        two_theta_min=5.,two_theta_max=155.)


@pytest.mark.parametrize('row',CATALOG['profiles'],ids=lambda p:p['filename'])
def test_all_published_mappings_have_exact_identity_checksum_and_physical_variance(row):
    path,receipt=hb2a_profiles.resolve_profile(setup(row))
    assert path.name==row['filename'] and sha256(path.read_bytes()).hexdigest()==row['sha256']
    assert receipt['registry_sha256']==hb2a_profiles.catalog_digest()
    assert receipt['ge_reflection']==row['ge_reflection']


@pytest.mark.parametrize('date',['2023-08-29','2025-09-16','2026-05-17','2020-01-01'])
def test_unknown_missing_or_contradictory_cycle_has_no_nearest_fallback(date):
    with pytest.raises(ValueError,match='No unique published'):
        hb2a_profiles.resolve_profile(setup(CATALOG['profiles'][0],start=date,end=date))


def test_no_unpublished_configuration_or_cross_cycle_acquisition():
    row=CATALOG['profiles'][0]
    s=setup(row);s.metadata['collimator']='OUT' # Cycle501 omits Ge113 OUT.
    with pytest.raises(ValueError,match='No unique published'):hb2a_profiles.resolve_profile(s)
    s=setup(row,end='2023-06-17')
    with pytest.raises(ValueError,match='No unique published'):hb2a_profiles.resolve_profile(s)


def test_no_naive_timestamp_ambiguous_registry_or_modified_profile(tmp_path):
    row=CATALOG['profiles'][0];s=setup(row)
    s.metadata['start_time']='2023-05-23T12:00:00'
    with pytest.raises(ValueError,match='timezone-qualified'):hb2a_profiles.resolve_profile(s)
    copied=json.loads(json.dumps(CATALOG));copied['profiles'].append(row)
    (tmp_path/'catalog.json').write_text(json.dumps(copied))
    with pytest.raises(ValueError,match='No unique published'):hb2a_profiles.resolve_profile(setup(row),root=tmp_path)
    (tmp_path/'catalog.json').write_text(json.dumps(CATALOG))
    (tmp_path/row['filename']).write_text('modified')
    with pytest.raises(ValueError,match='checksum'):hb2a_profiles.resolve_profile(setup(row),root=tmp_path)
