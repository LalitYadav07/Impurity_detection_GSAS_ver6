"""Mode-bearing CIFs permit name-only repair, never an implicit basis repair."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gsas_isodistort_import import _replace_inline_names


def test_name_only_rewrite_preserves_native_mode_and_operator_text():
    source = ('# Public test model\r\n'
              'data_child\r\n'
              '_symmetry_space_group_name_H-M "F 4/m -3 2/m"\r\n'
              '_space_group_IT_number 225\r\n'
              '_space_group_symop_operation_xyz "x,y,z"\r\n'
              '_iso_displacivemode_number 1\r\n'
              'loop_\r\n_iso_displacivemode_ID\r\n_iso_displacivemode_label\r\n'
              '1 "Pm-3m[1/2,1/2,1/2]R1+(a)[o:d:dsp]A1g(a)"\r\n')
    rewritten = _replace_inline_names(source, 'F m -3 m')
    expected = source.replace('"F 4/m -3 2/m"', '"F m -3 m"')
    assert rewritten == expected


def test_name_rewrite_retains_inline_provenance_comment():
    source = "  _space_group_name_H-M_alt 'F 4/m -3 2/m'  # ISODISTORT output\n"
    assert _replace_inline_names(source, 'F m -3 m') == '  _space_group_name_H-M_alt "F m -3 m"  # ISODISTORT output\n'


@pytest.mark.parametrize('source', ['data_child\n', '_symmetry_space_group_name_H-M\n"F m -3 m"\n'])
def test_unsupported_names_require_review(source):
    with pytest.raises(ValueError):
        _replace_inline_names(source, 'F m -3 m')


@pytest.mark.parametrize('symbol', ['', 'P 1\n_atom_site_occupancy 0', 'P "1"', "P '1'"])
def test_symbol_cannot_introduce_cif_content(symbol):
    with pytest.raises(ValueError):
        _replace_inline_names('_symmetry_space_group_name_H-M "P 1"\n', symbol)
