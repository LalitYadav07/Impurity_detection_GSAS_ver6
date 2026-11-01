import pytest

from gpx_recipe.symmetry.kernel_patch import corrected_source,function_source,replace_once,corrected_nuclear_derivative


@pytest.mark.parametrize('source',['','def MagStructureFactor2():\n    pass\n',
    'print("an unreviewed upstream update")\n'])
def test_unknown_source_fails_closed_without_executing_it(source):
    with pytest.raises(ValueError,match='Unreviewed GSAS-II source'):
        corrected_source(source)


def test_source_replacement_requires_exactly_one_expression():
    assert replace_once('before target after','target','reviewed')=='before reviewed after'
    for source in ['no match','target target']:
        with pytest.raises(ValueError,match='source review required'):
            replace_once(source,'target','reviewed')


@pytest.mark.parametrize('source', ['', 'def StructureFactorDerv2():\n    pass\n',
                                   'raise RuntimeError("unreviewed code must not run")\n'])
def test_nuclear_derivative_requires_the_independently_reviewed_source(source):
    with pytest.raises(ValueError, match='pinned nuclear powder derivative changed'):
        corrected_nuclear_derivative(source)


def test_function_capture_preserves_exact_whitespace_and_ignores_neighbor_code():
    source='''# module header
def first():
    """Preserve this docstring."""
    return 1

def second():
    return 2
'''
    assert function_source(source,'first')=='def first():\n    """Preserve this docstring."""\n    return 1\n'
    with pytest.raises(ValueError,match='unique pinned'):
        function_source(source,'absent')
    with pytest.raises(ValueError,match='unique pinned'):
        function_source(source+'\ndef first():\n    return 3\n','first')
