from hashlib import sha256
import json
import os

import pytest

from radar_pd_nova.battery import powgen
from radar_pd_nova.battery_preparation import run_sources
from radar_pd_nova.battery.reduction_evidence import literal_effects, merge_effects


def source(tmp_path, *, invalid=False, banks=(2,)):
    path = tmp_path / 'PG3_43124.gsa'
    path.write_text('Sample Run: 43124 Vanadium Run: 42839 Wavelength: 1.5 A\n'
        '# synthetic battery adapter control\n# Instrument: POWGEN\n'
        f'# {len(banks)} Histograms\n# with Y multiplied by the bin widths.\n'
        '# Sample Temperature: 300 K Freq: 60 Hz Guide: -54 mm\n' + ''.join(
        '# Total flight path 63.18m, tth 90deg, DIFC 22585.8\n'
        f'BANK {b} 4 4 SLOG 10000 10030 .001 0 FXYE\n'
        f'10000 20 2\n10010 -5 1\n10020 0 3\n10030 7 {0 if invalid else 1}\n' for b in banks))
    path.with_suffix('.py').write_text("SNSPowderReduction(CalibrationFile='/SNS/PG3/shared/CALIBRATION/2019_1_11A_CAL/PG3_PAC.h5', "
        "CharacterizationRunsFile='/SNS/PG3/shared/CALIBRATION/2019_1_11A_CAL/PG3_char-HighRes.txt', SaveAs='gsas topas')\n")
    return path


def test_current_mapping_retains_original_records_and_separate_bank_labels(tmp_path):
    path = source(tmp_path)
    setup = powgen.inspect_reduced_setup(path)
    profile = powgen.catalog_profile(setup, 2)
    data, inst, receipt = powgen.stage_bank(path, setup, 2, profile.path, profile.profile_bank,
        tmp_path / 'staged', acquisition={'sample_id': 'synthetic'})
    assert setup.banks[0].bank == 2 and profile.profile_bank == 1
    assert receipt['source_numeric_sha256'] == receipt['numeric_sha256'] == setup.banks[0].numeric_sha256
    assert '10010 -5 1\n10020 0 3\n' in data.read_text()
    assert 'BANK 1 4 4' in data.read_text() and 'Bank:1.0' in inst.read_text()
    marker = next(line for line in data.read_text().splitlines() if line.startswith('# RADAR-PD input'))
    evidence = json.loads(marker.split(': ', 1)[1])
    assert evidence['instrument'] == 'PG3' and evidence['source_bank'] == 2
    assert evidence['reduction_evidence'][0]['sha256'] == sha256(path.with_suffix('.py').read_bytes()).hexdigest()
    assert 'scientific' in evidence['warnings'][0]['message']


def test_lookup_does_not_mix_same_number_across_instruments_or_summed_files(tmp_path):
    source(tmp_path)
    for name in ('NOM43124tof.gsa', 'PG3_43124_sum.gsa', 'PG3_43125.gsa'):
        (tmp_path / name).write_text('not this exposure')
    rows = run_sources(tmp_path, 43124, instrument='PG3')
    assert len(rows) == 1 and rows[0]['path'].endswith('/PG3_43124.gsa')


def test_invalid_uncertainties_require_explicit_exclusion_and_retain_original_evidence(tmp_path):
    path = source(tmp_path, invalid=True)
    with pytest.raises(ValueError, match='nonpositive uncertainty'):
        powgen.inspect_reduced_setup(path)
    setup = powgen.inspect_reduced_setup(path, inventory_only=True)
    profile = powgen.catalog_profile(setup, 2)
    with pytest.raises(ValueError, match='invalid numerical'):
        powgen.stage_bank(path, setup, 2, profile.path, 1, tmp_path / 'no', acquisition={})
    data, _, receipt = powgen.stage_bank(path, setup, 2, profile.path, 1, tmp_path / 'yes',
        acquisition={}, allow_invalid_counts=True)
    assert receipt['invalid_count_bins'] == [dict(record_index=3, intensity='7', uncertainty='0')]
    assert '10030 0 1\n' in data.read_text() and '10030 7 0\n' in path.read_text()
    assert receipt['source_numeric_sha256'] != receipt['numeric_sha256']


@pytest.mark.parametrize('change,message', [
    (lambda text: text.replace('# with Y multiplied by the bin widths.\n', ''), 'bin-width-multiplied'),
    (lambda text: text.replace('# Instrument: POWGEN', '# Instrument: NOMAD'), 'different or ambiguous instrument'),
    (lambda text: text.replace('10010 -5 1', '10000 -5 1'), 'increase strictly'),
    (lambda text: text.replace('SLOG', 'CONST'), 'requires explicit SLOG'),
    (lambda text: text.replace('Sample Run: 43124', 'Sample Run: 43125'), 'Sample Run header'),
])
def test_unknown_or_conflicting_observation_conventions_rejected(tmp_path, change, message):
    path = source(tmp_path)
    path.write_text(change(path.read_text()))
    with pytest.raises(ValueError, match=message):
        powgen.inspect_reduced_setup(path)


@pytest.mark.parametrize('target', ['source', 'sidecar'])
def test_same_size_and_timestamp_replacements_fail_before_staging(tmp_path, target):
    path = source(tmp_path)
    setup = powgen.inspect_reduced_setup(path)
    profile = powgen.catalog_profile(setup, 2)
    changed = path if target == 'source' else path.with_suffix('.py')
    stat = changed.stat()
    changed.write_text(changed.read_text().replace('10010 -5', '10010 -6') if target == 'source'
                       else changed.read_text().replace('2019_1', '2020_1'))
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='changed'):
        powgen.stage_bank(path, setup, 2, profile.path, 1, tmp_path / 'staged', acquisition={})
    assert not (tmp_path / 'staged').exists()


def test_bad_coordinates_cannot_be_hidden_by_outside_fit_option(tmp_path):
    path = source(tmp_path)
    path.write_text(path.read_text().replace('10030 7 1', 'nan 7 1'))
    setup = powgen.inspect_reduced_setup(path, inventory_only=True)
    profile = powgen.catalog_profile(setup, 2)
    with pytest.raises(ValueError, match='Invalid TOF coordinates'):
        powgen.stage_bank(path, setup, 2, profile.path, 1, tmp_path / 'staged',
                          acquisition={}, allow_invalid_counts=True)


def test_new_reduction_receipt_invalidates_inspection_even_when_data_is_unchanged(tmp_path):
    path=source(tmp_path)
    setup=powgen.inspect_reduced_setup(path)
    path.with_suffix('.xml').write_text('<Reduction/>')
    with pytest.raises(ValueError,match='evidence changed'):
        powgen.verify_evidence(path,setup)


def test_multiple_banks_are_staged_from_one_verified_exposure(tmp_path):
    path = source(tmp_path, banks=(2, 3))
    setup = powgen.inspect_reduced_setup(path)
    profile = powgen.catalog_profile(setup, 2)
    for bank in (2, 3):
        data, _, receipt = powgen.stage_bank(path, setup, bank, profile.path, 1, tmp_path / f'b{bank}', acquisition={})
        assert data.read_text().count('BANK ') == 1
        assert receipt['source_bank'] == bank
        assert receipt['source_sha256'] == setup.source_sha256
    # Metadata has no supported bank-3 mapping at this wavelength; staging
    # an explicit profile above is an unqualified numerical control only.
    with pytest.raises(ValueError, match='No validated'):
        powgen.catalog_profile(setup, 3)


def test_reduction_effects_are_literal_evidence_and_not_executable(tmp_path):
    marker = tmp_path/'never-created'
    raw = (f"__import__('pathlib').Path({str(marker)!r}).touch()\n"
           "SNSPowderReduction(Filename='/SNS/PG3/IPTS-1/nexus/PG3_43124.nxs.h5', "
           "PushDataPositive='AddMinimum', ScaleData=100, OffsetData=-1, SaveAs='gsas')\n"
           "SNSPowderReduction(Filename='other', ScaleData=2, SaveAs='topas')").encode()
    values = literal_effects(raw, '.py', 43124)
    assert values == {'PushDataPositive':'AddMinimum','ScaleData':100,'OffsetData':-1}
    assert not marker.exists()
    assert merge_effects([values, {'ScaleData':'100.0'}])['ScaleData'] == '100.0'
    with pytest.raises(ValueError, match='disagree on intensity'):
        merge_effects([values, {'ScaleData':'1.0'}])


@pytest.mark.parametrize('filename', ['43124,43125', '/a/PG3_43125.nxs.h5',
    '/a/PG3_43125.nxs.h5,/b/PG3_43124.nxs.h5'])
def test_a_matching_last_filename_cannot_hide_a_summed_reduction(filename):
    with pytest.raises(ValueError, match='summed or names a different'):
        literal_effects(f'SNSPowderReduction(Filename={filename!r}, SaveAs="gsas")'.encode(), '.py', 43124)


@pytest.mark.parametrize('arguments',["'PG3_43125.nxs.h5'", '*source_runs'])
def test_positional_reduction_sources_require_an_explicit_reviewable_mapping(arguments):
    with pytest.raises(ValueError,match='Positional reduction arguments'):
        literal_effects(f'SNSPowderReduction({arguments}, SaveAs="gsas")'.encode(), '.py', 43124)


def test_xml_sum_and_duplicate_intensity_declarations_fail_closed():
    with pytest.raises(ValueError, match='summed'):
        literal_effects(b'<Reduction><runnumber>43124</runnumber><sum>1</sum></Reduction>', '.xml', 43124)
    with pytest.raises(ValueError, match='Duplicate intensity'):
        literal_effects(b'<Reduction><scaledata>1</scaledata><scaledata>2</scaledata></Reduction>', '.xml', 43124)


def test_positivity_shift_is_carried_to_the_user_and_staged_provenance(tmp_path):
    path = source(tmp_path)
    script = path.with_suffix('.py')
    script.write_text(script.read_text().replace("SaveAs='gsas topas'", "SaveAs='gsas topas', PushDataPositive='AddMinimum'"))
    setup = powgen.inspect_reduced_setup(path)
    assert any('PushDataPositive=AddMinimum' in note for note in setup.warnings)
    assert setup.intensity_effects['PushDataPositive'] == 'AddMinimum'
