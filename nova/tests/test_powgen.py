from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from radar_pd_nova.powgen import (
    AmbiguousPowgenInputError,
    PowgenResolutionError,
    UnknownPowgenInputError,
    parse_pg3_run_number,
    parse_powgen_reduced_filename,
    resolve_packaged_powgen_profile_path,
    resolve_powgen_profile,
    select_preferred_reduced_file,
)


def test_parses_run_number_from_raw_and_reduced_paths() -> None:
    assert parse_pg3_run_number("PG3_63764.nxs.h5") == 63764
    assert parse_pg3_run_number("/SNS/PG3/IPTS-38000/shared/PG3_63764.gsa") == 63764
    assert parse_pg3_run_number(r"C:\reduced\PG3_63764-2.xye") == 63764


@pytest.mark.parametrize(
    ("filename", "run_number", "bank", "suffix"),
    [
        ("PG3_63764.gsa", 63764, None, ".gsa"),
        ("PG3_63764-2.xye", 63764, 2, ".xye"),
        ("pg3_63764-2.DAT", 63764, 2, ".dat"),
        ("PG3_63764-1.GSA", 63764, 1, ".gsa"),
    ],
)
def test_parses_supported_reduced_filenames(
    filename: str, run_number: int, bank: int | None, suffix: str
) -> None:
    parsed = parse_powgen_reduced_filename(filename)

    assert parsed.run_number == run_number
    assert parsed.bank == bank
    assert parsed.suffix == suffix
    assert parsed.filename == filename


@pytest.mark.parametrize(
    "filename",
    [
        "PG3_63764.nxs.h5",
        "PG3-63764.gsa",
        "PG3_63764.gsas",
        "PG3_63764.csv",
        "PG3_0.gsa",
        "PG3_63764-0.dat",
        "notes.dat",
    ],
)
def test_rejects_non_reduced_or_nonstandard_filenames(filename: str) -> None:
    with pytest.raises(UnknownPowgenInputError):
        parse_powgen_reduced_filename(filename)


def test_prefers_gsa_over_xye_and_dat_for_same_run() -> None:
    selected = select_preferred_reduced_file(
        [
            "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764-2.dat",
            "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764-2.xye",
            "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa",
            "/SNS/PG3/IPTS-38000/shared/autoreduce/README.txt",
        ]
    )

    assert selected.filename == "PG3_63764.gsa"
    assert selected.run_number == 63764


def test_can_select_one_run_from_mixed_candidates_when_explicit() -> None:
    selected = select_preferred_reduced_file(
        ["PG3_63763.gsa", "PG3_63764-2.xye", "PG3_63764-2.dat"],
        run_number=63764,
    )

    assert selected.filename == "PG3_63764-2.xye"


def test_fails_closed_for_mixed_runs_without_explicit_run() -> None:
    with pytest.raises(AmbiguousPowgenInputError, match="multiple PG3 runs"):
        select_preferred_reduced_file(["PG3_63763.gsa", "PG3_63764.gsa"])


def test_fails_closed_for_equally_preferred_files() -> None:
    with pytest.raises(AmbiguousPowgenInputError, match="equally preferred"):
        select_preferred_reduced_file(
            ["/first/PG3_63764.gsa", "/second/PG3_63764.gsa"]
        )


@pytest.mark.parametrize(
    ("run_number", "wavelength", "canonical_wavelength", "expected"),
    [
        (63383, "0.8", "0.8", "2026B_HighRes_60HzB1_CWL0p8.instprm"),
        (63764, "1.50", "1.5", "2026B_HighRes_60HzB2_CWL1p5.instprm"),
        (70000, "2.665", "2.665", "2026B_HighRes_60HzB3_CWL2p665.instprm"),
    ],
)
def test_resolves_2026b_profile_by_exact_run_wavelength_and_frequency(
    run_number: int,
    wavelength: str,
    canonical_wavelength: str,
    expected: str,
) -> None:
    result = resolve_powgen_profile(
        run_number=run_number,
        wavelength_angstrom=wavelength,
        frequency_hz=60,
    )

    assert result.profile_filename == expected
    assert result.profile_resource_filename == f"powgen_profiles/{expected}"
    assert len(result.profile_sha256) == 64
    assert result.provenance.cycle == "2026B"
    assert result.provenance.run_min == 63383
    assert result.provenance.run_max is None
    assert result.provenance.run_number == run_number
    assert result.provenance.wavelength_angstrom == canonical_wavelength
    assert result.provenance.frequency_hz == "60"
    assert result.provenance.registry_schema_version == 1
    assert (
        result.provenance.registry_source_url
        == "https://neutrons.ornl.gov/powgen/users"
    )
    assert len(result.provenance.registry_sha256) == 64
    assert result.as_dict()["provenance"]["rule_id"].startswith("PG3-2026B-")
    profile_path = resolve_packaged_powgen_profile_path(result)
    assert profile_path.name == expected
    assert sha256(profile_path.read_bytes()).hexdigest() == result.profile_sha256


def test_resolves_and_verifies_packaged_profile_path() -> None:
    result = resolve_powgen_profile(
        run_number=63764,
        wavelength_angstrom="1.5",
        frequency_hz=60,
    )

    profile_path = resolve_packaged_powgen_profile_path(result)

    assert profile_path.name == "2026B_HighRes_60HzB2_CWL1p5.instprm"
    assert profile_path.is_file()
    assert sha256(profile_path.read_bytes()).hexdigest() == result.profile_sha256


def test_packaged_profile_path_rejects_checksum_mismatch() -> None:
    result = resolve_powgen_profile(
        run_number=63764,
        wavelength_angstrom="1.5",
        frequency_hz=60,
    )
    tampered = replace(result, profile_sha256="0" * 64)

    with pytest.raises(PowgenResolutionError, match="checksum mismatch"):
        resolve_packaged_powgen_profile_path(tampered)


def test_packaged_profile_path_rejects_unsafe_resource() -> None:
    result = resolve_powgen_profile(
        run_number=63764,
        wavelength_angstrom="1.5",
        frequency_hz=60,
    )
    unsafe = replace(result, profile_resource_filename="../profile.instprm")

    with pytest.raises(PowgenResolutionError, match="Unsafe packaged"):
        resolve_packaged_powgen_profile_path(unsafe)


@pytest.mark.parametrize(
    ("run_number", "wavelength", "frequency"),
    [
        (63382, "1.5", "60"),
        (63764, "1.4999", "60"),
        (63764, "1.5", "30"),
    ],
)
def test_profile_resolution_has_no_cycle_or_nearest_setting_fallback(
    run_number: int, wavelength: str, frequency: str
) -> None:
    with pytest.raises(UnknownPowgenInputError, match="No explicit POWGEN profile"):
        resolve_powgen_profile(
            run_number=run_number,
            wavelength_angstrom=wavelength,
            frequency_hz=frequency,
        )


def test_profile_resolution_rejects_fractional_run_number() -> None:
    with pytest.raises(UnknownPowgenInputError, match="run_number must be an integer"):
        resolve_powgen_profile(
            run_number=63764.5,
            wavelength_angstrom="1.5",
            frequency_hz="60",
        )


def test_profile_resolution_rejects_ambiguous_exact_registry_rules() -> None:
    registry = {
        "schema_version": 1,
        "instrument": "PG3",
        "source": {"url": "https://example.invalid/registry"},
        "profiles": [
            {
                "id": "first",
                "cycle": "2026B",
                "run_min": 63383,
                "run_max": None,
                "wavelength_angstrom": "1.5",
                "frequency_hz": "60",
                "filename": "first.instprm",
                "resource_filename": "powgen_profiles/first.instprm",
                "sha256": "1" * 64,
            }
        ],
    }
    duplicate = deepcopy(registry["profiles"][0])
    duplicate["id"] = "second"
    duplicate["filename"] = "second.instprm"
    duplicate["resource_filename"] = "powgen_profiles/second.instprm"
    registry["profiles"].append(duplicate)

    with pytest.raises(AmbiguousPowgenInputError, match="first, second"):
        resolve_powgen_profile(
            run_number=63764,
            wavelength_angstrom="1.5",
            frequency_hz="60",
            registry=registry,
        )


def test_profile_resolution_rejects_malformed_registry() -> None:
    registry = {
        "schema_version": 1,
        "instrument": "PG3",
        "source": {"url": "https://example.invalid/registry"},
        "profiles": [
            {
                "id": "bad-range",
                "cycle": "2026B",
                "run_min": 63383,
                "run_max": 63382,
                "wavelength_angstrom": "1.5",
                "frequency_hz": "60",
                "filename": "profile.instprm",
                "resource_filename": "powgen_profiles/profile.instprm",
                "sha256": "0" * 64,
            }
        ],
    }

    with pytest.raises(PowgenResolutionError, match="run_max < run_min"):
        resolve_powgen_profile(
            run_number=63764,
            wavelength_angstrom="1.5",
            frequency_hz="60",
            registry=registry,
        )


def test_profile_resolution_rejects_missing_resource_metadata() -> None:
    registry = {
        "schema_version": 1,
        "instrument": "PG3",
        "source": {"url": "https://example.invalid/registry"},
        "profiles": [
            {
                "id": "missing-resource",
                "cycle": "2026B",
                "run_min": 63383,
                "run_max": None,
                "wavelength_angstrom": "1.5",
                "frequency_hz": "60",
                "filename": "profile.instprm",
            }
        ],
    }

    with pytest.raises(PowgenResolutionError, match="profile rule"):
        resolve_powgen_profile(
            run_number=63764,
            wavelength_angstrom="1.5",
            frequency_hz="60",
            registry=registry,
        )


@pytest.mark.parametrize(
    ("resource_filename", "digest", "message"),
    [
        ("../profile.instprm", "0" * 64, "resource_filename"),
        ("powgen_profiles/profile.instprm", "not-a-digest", "sha256"),
    ],
)
def test_profile_resolution_rejects_unsafe_resource_metadata(
    resource_filename: str, digest: str, message: str
) -> None:
    registry = {
        "schema_version": 1,
        "instrument": "PG3",
        "source": {"url": "https://example.invalid/registry"},
        "profiles": [
            {
                "id": "unsafe-resource",
                "cycle": "2026B",
                "run_min": 63383,
                "run_max": None,
                "wavelength_angstrom": "1.5",
                "frequency_hz": "60",
                "filename": "profile.instprm",
                "resource_filename": resource_filename,
                "sha256": digest,
            }
        ],
    }

    with pytest.raises(PowgenResolutionError, match=message):
        resolve_powgen_profile(
            run_number=63764,
            wavelength_angstrom="1.5",
            frequency_hz="60",
            registry=registry,
        )
