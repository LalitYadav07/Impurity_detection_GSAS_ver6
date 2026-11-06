"""Real long-flight-path TOF ranges must survive preparation admission."""
import pytest

from gpx_recipe.preparation.reference import validate_grid, MAXIMUM_TOF_MICROSECONDS
from gpx_recipe.preparation.worker import validate_request


@pytest.mark.parametrize('bounds', [(11301.913503367501, 239409.), (220000., 300000.),
                                  (3000., MAXIMUM_TOF_MICROSECONDS)])
def test_long_tof_range_survives_the_same_worker_admission(bounds):
    grid = validate_grid(kind='PNT', lower=bounds[0], upper=bounds[1], bins=3818, bank=1)
    request = dict(source='parent.cif', instrument='bank.instprm', grid=grid,
                   settings={}, destination='prepared')
    accepted = validate_request('reference', request)
    assert accepted['grid'] == grid
    assert accepted['grid']['upper'] == bounds[1]


@pytest.mark.parametrize('kind,lower,upper', [('PNT', 3000., float('inf')),
    ('PNT', 3000., MAXIMUM_TOF_MICROSECONDS+1), ('PNT', 3., 3.), ('PNC', 10., 180.)])
def test_extended_tof_support_retains_finite_and_physical_admission(kind, lower, upper):
    with pytest.raises(ValueError):
        validate_grid(kind=kind, lower=lower, upper=upper)
