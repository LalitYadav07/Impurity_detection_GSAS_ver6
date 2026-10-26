from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.diffraction_coordinates import TofCalibration,coordinate_values,display_pattern
from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
from test_pattern_transition import observed


def profile(tmp_path):
    path=tmp_path/"banks.instprm"
    path.write_text("# Bank 1\nType:PNT;Bank:1;difC:1000;difA:0;difB:0;Zero:0\n"
        "# Bank 7\nType:PNT;Bank:7;difC:22000;difA:-4.5;difB:12;Zero:-7\n")
    return path,sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("coefficients",[(22000,-4.5,12,-7),(10000,4,-12,80),(1200,0,0,0),(22000,-1e-10,0,2)])
def test_full_tof_polynomial_round_trip_on_positive_physical_branch(coefficients):
    c=TofCalibration(*coefficients)
    d=np.geomspace(.3,12,1001)
    tof=c.zero+c.difc*d+c.difa*d*d+c.difb/d
    np.testing.assert_allclose(c.spacing(tof),d,rtol=1e-9,atol=1e-10)
    q=coordinate_values(tof,"tof_us","q_inv_angstrom",c)
    np.testing.assert_allclose(q,2*np.pi/d,rtol=1e-9)
    np.testing.assert_allclose(coordinate_values(q,"q_inv_angstrom","tof_us",c),tof,rtol=1e-9)


def test_nonphysical_tof_range_is_withheld_instead_of_selecting_a_second_root():
    c=TofCalibration(100,-5)
    with pytest.raises(ValueError,match="outside"):
        c.spacing([501]) # maximum t=500 at d=10 on increasing branch
    with pytest.raises(ValueError):
        coordinate_values([20],"d_angstrom","tof_us",c)


def test_profile_bank_labels_and_original_checksum_are_mandatory(tmp_path):
    path,checksum=profile(tmp_path)
    c=TofCalibration.from_profile(path,source_bank=2,profile_bank=7,expected_sha256=checksum)
    assert c.difc==22000 and c.source_bank==2 and c.profile_bank==7
    with pytest.raises(ValueError,match="absent"):
        TofCalibration.from_profile(path,source_bank=2,profile_bank=2,expected_sha256=checksum)
    path.write_text(path.read_text().replace("difC:22000","difC:23000"))
    with pytest.raises(ValueError,match="checksum"):
        TofCalibration.from_profile(path,source_bank=2,profile_bank=7,expected_sha256=checksum)


def test_actual_packaged_powgen_bank_mapping_uses_offset_and_nonlinear_terms():
    path=Path(__file__).parents[1]/"src/radar_pd_nova/data/powgen_calibrations/2026B_HighRes_60HzB2_CWL1p5.instprm"
    c=TofCalibration.from_profile(path,source_bank=2,profile_bank=1,expected_sha256=sha256(path.read_bytes()).hexdigest())
    assert c.difa!=0 and c.difb!=0 and c.zero!=0
    d=np.array([.5,1,2,4,8,11])
    np.testing.assert_allclose(c.spacing(c.tof(d)),d,rtol=1e-10)
    assert np.max(abs(c.spacing(c.tof(d))-c.tof(d)/c.difc))>.01


def test_legacy_tof_icons_are_read_as_difc_difa_zero_not_difb():
    path=Path(__file__).parents[1]/"src/radar_pd_nova/data/powgen_calibrations/PGHR_60-2016A.prm"
    c=TofCalibration.from_profile(path,source_bank=1,profile_bank=1,expected_sha256=sha256(path.read_bytes()).hexdigest())
    assert (c.difc,c.difa,c.difb,c.zero)==(22589.16,6.55,0,1.43)
    np.testing.assert_allclose(c.spacing(c.tof([.4,1,5])),[.4,1,5],rtol=1e-10)


def test_display_conversion_reorders_q_without_changing_native_samples_or_masks():
    original=observed();c=TofCalibration(22000,-4,1,-7,source_bank=2)
    source_x=original.x.copy();source_y=original.intensity.copy()
    converted=display_pattern(original,"q_inv_angstrom",c)
    np.testing.assert_allclose(converted.x,2*np.pi/c.spacing(original.x[::-1]))
    np.testing.assert_array_equal(converted.intensity,source_y[::-1])
    np.testing.assert_array_equal(converted.valid,original.valid[::-1])
    np.testing.assert_array_equal(original.x,source_x)
    np.testing.assert_array_equal(original.intensity,source_y)
    with pytest.raises(ValueError,match="another physical bank"):
        display_pattern(original,"d_angstrom",replace(c,source_bank=1))
    with pytest.raises(ValueError,match="exact mapped"):
        display_pattern(original,"d_angstrom")


def test_d_and_q_interconvert_without_an_unneeded_tof_profile():
    np.testing.assert_allclose(coordinate_values([1,2,4],"d_angstrom","q_inv_angstrom"),[2*np.pi,np.pi,np.pi/2])


def test_saved_job_receipt_restores_the_explicit_mapping_without_controller_or_network(tmp_path):
    path,checksum=profile(tmp_path)
    receipt={"source_bank":2,"profile":{"path":str(path),"profile_bank":7,"sha256":checksum}}
    record=SimpleNamespace(uid="job",galaxy_job_id="job",provenance={"powgen_bank":receipt})
    controller=SimpleNamespace(records={"run":record})
    c=mapped_calibration({"run_id":"run","bank":2},controller,{})
    assert c.difc==22000
    with pytest.raises(ValueError,match="another physical bank"):
        mapped_calibration({"run_id":"run","bank":1},controller,{})


def test_nominal_export_geometry_is_not_accepted_as_a_calibrated_profile():
    with pytest.raises(ValueError,match="not available"):
        mapped_calibration({"run_id":"run","bank":2,"metadata":{"nomad_setup":{"difc":22000}}},None,{})
