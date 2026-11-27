"""Fail-closed scalar diagnostics without a GSAS-II or private-data fixture."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.symmetry.profiled_magnetic import ProfiledMagneticObjective
from gpx_recipe.symmetry.profiled_solver import _apply_starting_point


def context(*,error=None,native=None):
    obj=NativeMagneticStartingPoint.__new__(NativeMagneticStartingPoint)
    obj.model=SimpleNamespace(offset=np.array([2.001,-4.,1.]),nuclear=np.zeros(3),
        quadratic=np.zeros((3,1,1)),background=np.empty((3,0)))
    obj.model.heldout_error=lambda *args: (error if error is not None else
        ProfiledMagneticObjective.heldout_error(obj.model,*args))
    obj.residual=lambda *args: np.array([2.,-4.,1.]) if native is None else native
    obj.moment_names=['2::AMx:0']
    obj.build_receipt=dict(compiler='native_reflection_amplitudes',cache_hit=True)
    return obj


def fit():
    return SimpleNamespace(success=True,coefficients=np.array([1.25]),scale=5.,
        background=np.empty(0),maximum_moment=1.25,iterations=2,evaluations=3,
        scaled_stationarity_norm=1e-11,local_descent_found=False)


class UntouchedProject:
    def phase(self,*args):raise AssertionError('A rejected point must not mutate the project')
    def histogram(self,*args):raise AssertionError('A rejected point must not mutate the project')


def test_mismatch_records_exact_scalar_error_and_never_applies(tmp_path):
    obj=context()
    with pytest.raises(ValueError,match='final starting point disagrees') as caught:
        _apply_starting_point(obj,UntouchedProject(),fit(),tmp_path/'model.gpx')
    data=json.loads((tmp_path/'native-objective-mismatch.json').read_text())
    assert type(caught.value) is ValueError
    assert data==caught.value.native_objective_diagnostics
    assert data['check']=='final_starting_point_native_weighted_residual_identity'
    assert data['threshold']==1e-8
    assert data['relative_maximum_error']==pytest.approx(.001/4.)
    assert data['absolute_maximum_error']==pytest.approx(.001)
    assert data['normalization']==data['native_infinity_norm']==4.
    assert data['native_l2_norm']==pytest.approx(np.sqrt(21.))
    assert data['native_weighted_cost']==10.5
    assert data['predicted_weighted_cost']==pytest.approx(.5*(2.001**2+17.))
    assert data['independent_magnetic_variables']==['2::AMx:0']
    assert data['magnetic_coefficients']==[1.25] and data['scale']==5.
    assert data['background_coefficients']==[] and data['weighted_residual_points']==3
    assert data['cache_hit'] is True and data['compiler']=='native_reflection_amplitudes'
    assert data['candidate_applied'] is False and data['raw_residuals_saved'] is False
    assert data['tolerance_changed'] is False and data['acceptance_status_changed'] is False
    assert 'native_residual' not in data and 'predicted_residual' not in data


def test_existing_value_error_catch_and_threshold_are_preserved(tmp_path):
    obj=context(error=np.nextafter(1e-8,np.inf))
    with pytest.raises(ValueError) as caught:
        _apply_starting_point(obj,UntouchedProject(),fit(),tmp_path/'model.gpx')
    assert str(caught.value)=='The final starting point disagrees with the native weighted objective'
    assert type(caught.value) is ValueError
    assert caught.value.native_objective_diagnostics['threshold']==1e-8


def test_boundary_pass_does_not_write_failure_receipt(tmp_path):
    obj=context(error=1e-8)
    obj._all_moments=lambda:np.empty((0,3))
    obj.atoms=[];obj.histogram_name='synthetic';obj.scale_index=None;obj.background_indices=[]
    project=SimpleNamespace(histogram=lambda name:SimpleNamespace(data={}))
    result=_apply_starting_point(obj,project,fit(),tmp_path/'model.gpx')
    assert result['final_native_relative_error']==1e-8
    assert result['native_convergence_declared'] is False
    assert result['covariance_supplied'] is False
    assert not (tmp_path/'native-objective-mismatch.json').exists()


def test_unrelated_failure_does_not_create_identity_receipt(tmp_path):
    def fail(*args):raise ValueError('Independent failure')
    obj=SimpleNamespace(apply_to_project=fail)
    with pytest.raises(ValueError,match='Independent failure'):
        _apply_starting_point(obj,UntouchedProject(),fit(),tmp_path/'model.gpx')
    assert not (tmp_path/'native-objective-mismatch.json').exists()


@pytest.mark.parametrize('error_type',[OSError,TypeError,ValueError])
def test_receipt_storage_failure_preserves_rejection(monkeypatch,tmp_path,error_type):
    from gpx_recipe import execution
    def fail(*args):raise error_type('Diagnostic write failed')
    monkeypatch.setattr(execution,'write_json',fail)
    with pytest.raises(ValueError,match='final starting point disagrees') as caught:
        _apply_starting_point(context(),UntouchedProject(),fit(),tmp_path/'model.gpx')
    assert type(caught.value) is ValueError
    if callable(getattr(caught.value,'add_note',None)):
        assert caught.value.__notes__==['Could not save native identity diagnostics: '+error_type.__name__]


def test_receipt_storage_failure_preserves_rejection_without_exception_notes(monkeypatch,tmp_path):
    from gpx_recipe import execution
    class RejectedWithoutNotes(ValueError):
        add_note=None  # Python 3.10 does not supply BaseException.add_note.
    rejected=RejectedWithoutNotes('Original native rejection')
    rejected.native_objective_diagnostics={'candidate_applied':False}
    def apply(*args):raise rejected
    def storage(*args):raise OSError('Diagnostic write failed')
    monkeypatch.setattr(execution,'write_json',storage)
    with pytest.raises(RejectedWithoutNotes) as caught:
        _apply_starting_point(SimpleNamespace(apply_to_project=apply),UntouchedProject(),fit(),tmp_path/'model.gpx')
    assert caught.value is rejected
    assert not (tmp_path/'native-objective-mismatch.json').exists()


def test_overflowing_norms_remain_json_safe_and_fail_closed(tmp_path):
    obj=context(native=np.array([1e308,-1e308,1e308]))
    with pytest.raises(ValueError):
        _apply_starting_point(obj,UntouchedProject(),fit(),tmp_path/'model.gpx')
    data=json.loads((tmp_path/'native-objective-mismatch.json').read_text())
    assert data['native_l2_norm'] is None and data['native_weighted_cost'] is None
    assert 'native_weighted_cost' in data['nonfinite_diagnostic_fields']
    assert data['candidate_applied'] is False


def test_diagnostic_overflow_does_not_replace_original_failure(tmp_path):
    obj=context(error=1.)
    obj.model.nuclear[:]=1e308
    with np.errstate(all='raise'):
        with pytest.raises(ValueError,match='final starting point disagrees') as caught:
            _apply_starting_point(obj,UntouchedProject(),fit(),tmp_path/'model.gpx')
    assert type(caught.value) is ValueError
    data=json.loads((tmp_path/'native-objective-mismatch.json').read_text())
    assert data['predicted_weighted_cost'] is None
    assert data['candidate_applied'] is False
