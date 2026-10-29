import pytest
from gpx_recipe.symmetry.worker import require_runtime,TrialDisposition,VALIDATED_MAGNETIC_FUNCTIONS
from gpx_recipe.symmetry.runtime import VALIDATED_STACKS


def test_matching_an_unvalidated_runtime_hash_does_not_approve_that_runtime():
    old='42e5cad268a08efa8bee0a9dc0f99dcf8a2d0765b22a70e2146ac5ec50a688d0'
    with pytest.raises(TrialDisposition) as e:require_runtime({'magnetic_function_sha256':old},old)
    assert e.value.status=='runtime_incompatible'


def test_validated_runtime_must_also_match_the_approved_protocol():
    actual=next(iter(VALIDATED_MAGNETIC_FUNCTIONS))
    require_runtime({'magnetic_function_sha256':actual,'native_stack_sha256':next(iter(VALIDATED_STACKS))},actual)
    with pytest.raises(TrialDisposition):require_runtime({'magnetic_function_sha256':actual},'a'*64)


def test_one_matching_function_does_not_approve_a_modified_native_stack():
    actual=next(iter(VALIDATED_MAGNETIC_FUNCTIONS))
    with pytest.raises(TrialDisposition):require_runtime({'magnetic_function_sha256':actual,'native_stack_sha256':'f'*64},actual)
