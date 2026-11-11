"""Instrument-bound input ranges must preserve the exact native grid scope."""
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.preparation_coordinates import convert_limits
from test_preparation_controller import app
from test_preparation_workspace import workspace
from test_model_runtime import remote


def profile(tmp_path,kind='PNT'):
    path=tmp_path/'profile.instprm'
    text=('Type: PNT\nBank: 1\ndifC: 12000\ndifA: 0\ndifB: 0\nZero: 0\n'
          '# Bank 3\nType: PNT\nBank: 3\ndifC: 22000\ndifA: 4\ndifB: 35\nZero: -7\n'
          if kind=='PNT' else 'Type: PNC\nBank: 1\nLam: 2.41\nZero: 0.02\n')
    path.write_text(text)
    return path,sha256(path.read_bytes()).hexdigest()


def convert(path,checksum,source,target,values,*,kind='PNT',bank=3):
    return convert_limits(path,kind=kind,bank=bank,source=source,target=target,
                          lower=values[0],upper=values[1],expected_sha256=checksum)


def test_tof_and_q_limits_use_selected_bank_and_all_coordinate_terms(tmp_path):
    path,checksum=profile(tmp_path);spacing=np.array([.5,4.])
    expected=-7+22000*spacing+4*spacing**2+35/spacing
    native=convert(path,checksum,'d_angstrom','native',spacing)
    np.testing.assert_allclose(native,expected,rtol=1e-13)
    np.testing.assert_allclose(convert(path,checksum,'native','d_angstrom',native),spacing,rtol=1e-13)
    q=convert(path,checksum,'d_angstrom','q_inv_angstrom',spacing)
    np.testing.assert_allclose(q,2*np.pi/spacing[::-1],rtol=1e-13)
    np.testing.assert_allclose(convert(path,checksum,'q_inv_angstrom','native',q),native,rtol=1e-13)
    other=convert(path,checksum,'d_angstrom','native',spacing,bank=1)
    np.testing.assert_allclose(other,12000*spacing)
    assert native!=other


def test_cw_limits_use_wavelength_zero_and_bragg_domain(tmp_path):
    path,checksum=profile(tmp_path,'PNC')
    d=np.array([1.3,5.]);theta=2*np.degrees(np.arcsin(2.41/(2*d)))+.02
    native=convert(path,checksum,'d_angstrom','native',d,kind='PNC',bank=1)
    np.testing.assert_allclose(native,theta[::-1],rtol=1e-13)
    np.testing.assert_allclose(convert(path,checksum,'native','d_angstrom',native,kind='PNC',bank=1),d,rtol=1e-13)
    with pytest.raises(ValueError,match='Bragg'):
        convert(path,checksum,'d_angstrom','native',[.5,5.],kind='PNC',bank=1)


@pytest.mark.parametrize('limits',[[0,1],[1,1],[3,2],[float('nan'),3],[1,float('inf')],[True,3]])
def test_invalid_input_limits_are_not_relabelled(tmp_path,limits):
    path,checksum=profile(tmp_path)
    with pytest.raises(ValueError,match='positive, increasing'):
        convert(path,checksum,'d_angstrom','native',limits)


def test_profile_mutation_missing_bank_and_wrong_branch_cannot_convert(tmp_path):
    path,checksum=profile(tmp_path)
    with pytest.raises(ValueError,match='absent'):
        convert(path,checksum,'d_angstrom','native',[.5,4],bank=2)
    path.write_text(path.read_text().replace('22000','23000'))
    with pytest.raises(ValueError,match='checksum'):
        convert(path,checksum,'d_angstrom','native',[.5,4])
    path.write_text('Type: PNT\nBank: 3\ndifC: 100\ndifA: -1\ndifB: 0\nZero: 0\n')
    checksum=sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='physical TOF branch'):
        convert(path,checksum,'d_angstrom','native',[40,60])


def test_range_toggle_and_capture_preserve_native_scope(app,tmp_path):
    application,store,_,_=app;s=application.server.state
    path,checksum=profile(tmp_path)
    application._prep_profile_upload=SimpleNamespace(current_path=path)
    s.prep_profile_path=str(path);s.prep_profile_bank='3'
    s.prep_lower='11065';s.prep_upper='88065.75'
    native=[float(s.prep_lower),float(s.prep_upper)]
    for coordinate in ['d_angstrom','q_inv_angstrom','native','d_angstrom']:
        s.prep_range_coordinate=coordinate
        application._preparation_range_coordinate_changed()
        assert s.prep_lower and s.prep_upper
    s.prep_bank_label='Physical bank 2';s.prep_bins='2601'
    application.add_preparation_bank()
    assert not s.prep_error
    bank=application._prep_banks['Physical bank 2']
    np.testing.assert_allclose([bank['grid']['lower'],bank['grid']['upper']],native,rtol=1e-13)
    assert bank['grid']['bank']==3 and bank['grid']['bins']==2601
    assert bank['instrument'].read_bytes()==path.read_bytes()
    assert s.prep_bank_rows[0]['range'].endswith('µs')
    assert 'Captured native calculation range' in s.prep_range_message


def test_failed_unit_change_clears_old_numbers_without_changing_captured_banks(app):
    application,_,_,service=app;s=application.server.state
    application._prep_profile_upload=SimpleNamespace(current_path=None)
    s.prep_range_coordinate='d_angstrom'
    application._preparation_range_coordinate_changed()
    assert s.prep_lower==s.prep_upper==''
    assert 'Enter both limits' in s.prep_range_message
    assert not application._prep_banks and not service.submissions


def test_incomplete_cw_profile_gives_a_user_error_not_an_unhandled_key(app,tmp_path):
    application,_,_,_=app;s=application.server.state
    path=tmp_path/'incomplete.instprm';path.write_text('Type: PNC\nBank: 1\n')
    application._prep_profile_upload=SimpleNamespace(current_path=path)
    s.prep_profile_path=str(path);s.prep_grid_kind='PNC'
    s.prep_range_coordinate='d_angstrom'
    application._preparation_range_coordinate_changed()
    assert s.prep_lower==s.prep_upper==''
    assert 'coordinate parameters' in s.prep_range_message


def test_capture_rejects_a_profile_changed_after_range_conversion(app,tmp_path,monkeypatch):
    from radar_pd_nova import preparation_controller
    application,_,_,_=app;s=application.server.state
    path,_=profile(tmp_path)
    application._prep_profile_upload=SimpleNamespace(current_path=path)
    s.prep_profile_path=str(path);s.prep_profile_bank='3'
    s.prep_range_coordinate='d_angstrom';s.prep_lower='.8';s.prep_upper='5'
    original=preparation_controller.capture_file
    def changed(source,*args,**kwargs):
        source.write_text(source.read_text().replace('22000','23000'))
        return original(source,*args,**kwargs)
    monkeypatch.setattr(preparation_controller,'capture_file',changed)
    application.add_preparation_bank()
    assert 'changed' in s.prep_error
    assert not application._prep_banks
