"""Language variation, ambiguity and conservative interpretation contracts."""

from dataclasses import replace

import pytest

from radar_pd_nova.live_assistant_language import (
    ELEMENT_NAMES, ReplyChoice, ReplyContext, SampleOption,
    interpret_reply, pending_answer,
)
from radar_pd_nova.uploads import _ELEMENT_SYMBOLS


SAMPLES = (SampleOption("A", "A — oxide"), SampleOption("B", "B — reference"),
           SampleOption("C", "C — catalyst"))
SELECTION = ReplyContext(step="selection", samples=SAMPLES, shown_samples=SAMPLES, banks=(1, 2))
CHEMISTRY = ReplyContext(step="chemistry")


@pytest.mark.parametrize("text", [
    "POWGEN IPTS-12345", "PG3, IPTS 12345", "powgen ipts:12345",
    "I am on POWGEN, IPTS-12345", "I'm working at POWGEN with IPTS 12345",
    "Our beamline is POWGEN and the experiment is IPTS-12345",
    "  POWGEN   IPTS : 12345  ", "actually use POWGEN, IPTS number is 12345",
])
def test_experiment_phrases_extract_the_same_explicit_values(text):
    reply = interpret_reply(text)
    assert reply.action == "patch"
    assert reply.value == {"live_beamline": "PG3", "powgen_ipts": "IPTS-12345"}


@pytest.mark.parametrize("text", ["12345", "please use 12345", "my proposal number is 12345"])
def test_bare_ipts_is_a_value_only_for_the_experiment_question(text):
    assert interpret_reply(text).value == {"powgen_ipts": "IPTS-12345"}
    assert interpret_reply(text, SELECTION).action != "patch"


def spelling_variants(word):
    return sorted({word[:i] + word[i+1:] for i in range(len(word))} |
                  {word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)} - {word})


@pytest.mark.parametrize("typo", spelling_variants("powgen"))
def test_beamline_spelling_edits_only_propose_and_preserve_exact_ipts(typo):
    reply = interpret_reply(f"I'm on {typo}, IPTS-12345")
    assert reply.action == "clarify"
    assert reply.value is None
    assert any(c.value == {"live_beamline": "PG3", "powgen_ipts": "IPTS-12345"} for c in reply.choices)


@pytest.mark.parametrize("bad", ["IPTS-12O45", "IPTS-12345a", "IPTS--12345-extra", "IPTS-0", "IPTS-012345", "IPTS-12.345"])
def test_malformed_identifiers_never_become_a_different_valid_identifier(bad):
    reply = interpret_reply("POWGEN " + bad)
    assert reply.action != "patch"
    assert all("powgen_ipts" not in (choice.value or {}) for choice in reply.choices if choice.action == "patch")


@pytest.mark.parametrize("goal", ["balanced", "rapid", "thorough"])
@pytest.mark.parametrize("phrase", ["{}", "please use {}", "I'd like {} please", "can we go with {}?", "apply {} analysis", "actually {}", "{} mode"])
def test_goal_phrasings(goal, phrase):
    reply = interpret_reply(phrase.format(goal))
    assert (reply.action, reply.value) == ("goal", goal)


@pytest.mark.parametrize("typo", spelling_variants("balanced") + spelling_variants("rapid") + spelling_variants("thorough"))
def test_goal_spelling_edits_require_confirmation(typo):
    reply = interpret_reply("please use " + typo)
    assert reply.action == "clarify"
    assert reply.choices and all(c.action == "goal" for c in reply.choices)


@pytest.mark.parametrize("text", ["why background", "what does background mean?", "explain the baseline", "can you explain backgroudn?", "please explain the background parameters"])
def test_help_variations_do_not_create_an_edit(text):
    reply = interpret_reply(text)
    assert (reply.action, reply.value) == ("help", "background")


@pytest.mark.parametrize("text", ["second", "the second one", "option 2", "the 2nd sample", "the 2nd sample.", "use sample 2", "select sample two", "I want reference", "sample ID B"])
def test_sample_replies_resolve_to_displayed_identity(text):
    reply = interpret_reply(text, SELECTION)
    assert reply.action == "patch"
    assert reply.value["powgen_processing_sample_ids"] == ["B"]


def test_numeric_sample_id_and_numbered_position_are_disambiguated():
    samples = (SampleOption("2", "2 — oxide"), SampleOption("99", "99 — reference"))
    context = replace(SELECTION, samples=samples, shown_samples=samples)
    reply = interpret_reply("2", context)
    assert reply.action == "clarify" and len(reply.choices) == 2
    assert interpret_reply("sample ID 2", context).value["powgen_processing_sample_ids"] == ["2"]
    assert interpret_reply("option 2", context).value["powgen_processing_sample_ids"] == ["99"]


def test_exact_sample_identity_retains_punctuation():
    samples = (SampleOption("A", "A — oxide"), SampleOption("A.", "A. — reference"))
    context = replace(SELECTION, samples=samples, shown_samples=samples)
    assert interpret_reply("A.", context).value["powgen_processing_sample_ids"] == ["A."]
    assert interpret_reply("sample ID A.", context).value["powgen_processing_sample_ids"] == ["A."]


@pytest.mark.parametrize('phrase',['Select sample 106208.','Use sample 106208!','sample 106208?'])
def test_sentence_punctuation_resolves_an_exact_discovered_numeric_sample_id(phrase):
    samples=(SampleOption('106208','106208 — oxide'),SampleOption('102085','102085 — nanoparticle'))
    context=replace(SELECTION,samples=samples,shown_samples=samples)
    reply=interpret_reply(phrase,context)
    assert reply.action=='patch' and reply.value['powgen_processing_sample_ids']==['106208']
    dotted=samples+(SampleOption('106208.','106208. — a distinct identity'),)
    literal=interpret_reply('Select sample 106208.',replace(context,samples=dotted,shown_samples=dotted))
    assert literal.value['powgen_processing_sample_ids']==['106208.']
    assert interpret_reply('sample ID 106208.',context).action=='clarify'
    assert interpret_reply('sample "106208."',context).action=='clarify'


def test_sample_names_with_reserved_words_use_the_active_question():
    samples = (SampleOption("42", "42 — NOMAD"), SampleOption("43", "43 — balanced"))
    context = replace(SELECTION, samples=samples, shown_samples=samples)
    assert interpret_reply("NOMAD", context).value["powgen_processing_sample_ids"] == ["42"]
    assert interpret_reply("balanced", context).value["powgen_processing_sample_ids"] == ["43"]
    assert interpret_reply("sample NOMAD", context).value["powgen_processing_sample_ids"] == ["42"]


def test_sample_typo_duplicate_names_and_digit_changes_never_auto_select():
    reply = interpret_reply("refernece", SELECTION)
    assert reply.action == "clarify" and reply.choices[0].value["powgen_processing_sample_ids"] == ["B"]
    samples = (SampleOption("A", "A — oxide 1"), SampleOption("B", "B — oxide 2"))
    context = replace(SELECTION, samples=samples, shown_samples=samples)
    assert interpret_reply("oxide 3", context).action != "patch"
    duplicate = (SampleOption("A", "A — oxide"), SampleOption("B", "B — oxide"))
    reply = interpret_reply("oxide", replace(SELECTION, samples=duplicate, shown_samples=duplicate))
    assert reply.action == "clarify" and len(reply.choices) == 2
    ids_only = (SampleOption("ABCDEF", "ABCDEF"),)
    reply = interpret_reply("ABDCEF", replace(SELECTION, samples=ids_only, shown_samples=ids_only))
    assert reply.action == "clarify" and not reply.choices


def test_grouped_sample_bank_answer_is_atomic_and_partial_answers_are_explicit():
    reply = interpret_reply("samples A and B, banks 1 and 2", SELECTION)
    assert reply.action == "patch"
    assert reply.value["powgen_processing_sample_ids"] == ["A", "B"]
    assert reply.value["powgen_processing_banks"] == [1, 2]
    bad = interpret_reply("sample B and bank 2.5", SELECTION)
    assert bad.action == "clarify"
    assert "not been applied" in bad.message
    assert bad.choices[0].value == {"powgen_processing_scope": "selected", "powgen_processing_sample_ids": ["B"]}


@pytest.mark.parametrize("bad", ["2.5", "2e0", "02", "-2", "7", "2 or 1", "2 and secret"])
def test_bank_numbers_must_be_exact_and_available(bad):
    assert interpret_reply("bank " + bad, SELECTION).action != "patch"


def test_nomad_bank_reply_cannot_confirm_calibration_or_choose_all_banks():
    context = replace(SELECTION, beamline="NOM")
    assert interpret_reply("all banks", context).action == "clarify"
    assert interpret_reply("banks 1 and 2", context).action == "clarify"
    assert interpret_reply("bank 2", context).value == {"nomad_physical_bank": 2, "nomad_profile_acknowledged": False}


@pytest.mark.parametrize("text", ["elements: Fe, O; can: Al", "sample: iron and oxygen; environment: aluminium", "my sample contains iron and oxygen, and the can is aluminum", "sample Fe, O; can Al"])
def test_chemical_role_and_name_variations(text):
    reply = interpret_reply(text, CHEMISTRY)
    assert reply.action == "patch"
    assert reply.value == {"live_sample_elements": "Fe, O", "live_environment_elements": "Al"}


def test_symbol_case_is_only_suggested_and_formulas_are_not_inferred():
    reply = interpret_reply("elements: fe, o; can: al", CHEMISTRY)
    assert reply.action == "clarify"
    assert reply.choices[0].value == {"live_sample_elements": "Fe, O", "live_environment_elements": "Al"}
    for bad in ("CO", "Fe2O3", "iron oxide", "Xx", "co2"):
        with pytest.raises(ValueError):
            interpret_reply(bad, CHEMISTRY)
    assert set(ELEMENT_NAMES.values()) == _ELEMENT_SYMBOLS
    assert interpret_reply("cobalt, carbon, oxygen", CHEMISTRY).value["live_sample_elements"] == "Co, C, O"


@pytest.mark.parametrize("text", ["not rapid", "don't use balanced", "all samples except A", "sample A without bank 2", "rapid and set background 999", "IPTS 12345 and start monitoring", "run /SNS/../secret"])
def test_unhandled_clauses_never_silently_apply_a_positive_fragment(text):
    assert interpret_reply(text, SELECTION).action not in {"patch", "goal", "continue"}


def test_explicit_positive_replacement_is_different_from_an_unspecified_negation():
    assert interpret_reply("not rapid, use balanced").value == "balanced"
    assert interpret_reply("not POWGEN but NOMAD").value == {"live_beamline": "NOM"}
    assert interpret_reply("not rapid, balanced and do more things").action == "clarify"


def test_pending_yes_cancel_and_numbers_need_a_specific_receipt():
    choice = ReplyChoice("Use POWGEN / IPTS-12345", "patch", {"live_beamline": "PG3", "powgen_ipts": "IPTS-12345"})
    assert pending_answer("yes please", (choice,)).value == choice.value
    assert pending_answer("POWGEN", (choice,)).value == choice.value
    assert pending_answer("cancel", (choice,)).action == "cancel"
    multiple = (ReplyChoice("Use balanced", "goal", "balanced"), ReplyChoice("Use rapid", "goal", "rapid"))
    assert pending_answer("yes", multiple).action == "clarify"
    assert pending_answer("second", multiple).value == "rapid"
    assert pending_answer("rapid", multiple).value == "rapid"
    assert interpret_reply("yes").action == "clarify"


def test_long_or_combinatorial_inputs_are_bounded():
    assert interpret_reply("balanced " + " " * 1000).action == "goal"  # Whitespace is normalized.
    assert interpret_reply("balanced " + "x" * 1000).action == "clarify"
    duplicate = tuple(SampleOption(str(i), str(i) + " — reference") for i in range(10))
    context = replace(SELECTION, samples=duplicate, shown_samples=duplicate)
    reply = interpret_reply("samples reference and reference and reference", context)
    assert reply.action == "clarify" and len(reply.choices) <= 6
