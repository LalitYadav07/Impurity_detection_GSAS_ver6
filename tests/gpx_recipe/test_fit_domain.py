from copy import deepcopy

from gpx_recipe.fit_domain import (DOMAIN_SCHEMA, PREVIOUS_DOMAIN_SCHEMA,
    NATIVE_FIT_CONVENTION, native_powder_fit_mask, fit_domain_digest)


def test_native_range_includes_the_first_bin_above_a_between_bin_upper_limit():
    assert native_powder_fit_mask([0.,1.,2.,3.],[.2,1.5]).tolist()==[False,True,True,False]
    assert native_powder_fit_mask([0.,1.,2.,3.],[.2,2.]).tolist()==[False,True,True,False]
    assert native_powder_fit_mask([0.,1.,2.,3.],[.2,1.5],[(1.9,2.1)]).tolist()==[False,True,False,False]


def test_legacy_hash_projection_does_not_rewrite_the_native_domain():
    old=dict(coordinate='tof',x=[0.,1.,2.,3.],observed=[1.,2.,3.,4.],weights=[1.]*4,
        weight_factor=1.,valid=[True]*4,fitted=[False,True,False,False],limits=[.2,1.5],exclusions=[])
    new=deepcopy(old);new.update(fitted=[False,True,True,False],fit_range_convention=NATIVE_FIT_CONVENTION)
    before=deepcopy(new)
    assert fit_domain_digest(old,schema=DOMAIN_SCHEMA) is None
    assert fit_domain_digest(old)==fit_domain_digest(new,schema=PREVIOUS_DOMAIN_SCHEMA)
    assert fit_domain_digest(new)!=fit_domain_digest(old)
    assert new==before
