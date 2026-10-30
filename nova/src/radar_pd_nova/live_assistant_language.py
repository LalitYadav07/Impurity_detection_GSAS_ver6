"""Bounded, contextual Live setup language. No model, service, or file access.

Exact interpretations may edit the draft. Fuzzy matches only produce choices;
the controller binds those choices to a receipt over the current UI state.
Identifiers and numbers are never spell-corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
import unicodedata
from typing import Any

from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from .live_assistant import HELP, parse_elements
from .uploads import _ELEMENT_SYMBOLS


MAX_REPLY = 1000
MAX_SHOWN_SAMPLES = 12


@dataclass(frozen=True)
class SampleOption:
    value: str
    title: str


@dataclass(frozen=True)
class ReplyContext:
    step: str = "experiment"
    beamline: str = "PG3"
    samples: tuple[SampleOption, ...] = ()
    shown_samples: tuple[SampleOption, ...] = ()
    banks: tuple[int, ...] = ()
    focus: str = ""


@dataclass(frozen=True)
class ReplyChoice:
    label: str
    action: str
    value: Any = None


@dataclass(frozen=True)
class ParsedReply:
    action: str
    value: Any = None
    message: str = ""
    choices: tuple[ReplyChoice, ...] = ()


from .live_beamlines import BEAMLINES as LIVE_BEAMLINES
BEAMLINES = {alias.casefold(): code for code, adapter in LIVE_BEAMLINES.items() for alias in (code, adapter.name)}
BEAMLINE_WORDS = {a: a for a in BEAMLINES if a.isalpha() and len(a) >= 4}
GOALS = {
    "balanced": "balanced", "balanced discovery": "balanced",
    "rapid": "rapid", "rapid hypotheses": "rapid", "fast": "rapid",
    "thorough": "thorough", "thorough discovery": "thorough", "quick": "quick",
}
TOPICS = {
    "background": ("background", "baseline", "background terms", "background parameters", "coefficients"),
    "chemistry": ("chemistry", "elements", "sample elements", "environment", "container", "sample can", "can material"),
    "main": ("main", "main phase", "dominant phase", "main cif"),
    "library": ("library", "database", "catalog", "candidate structures", "cifs"),
    "banks": ("bank", "banks", "histogram", "histograms"),
    "profile": ("profile", "instrument", "instrument file", "instprm", "calibration", "beamline", "ipts"),
    "range": ("range", "fit range", "fit limits", "mask", "masks", "ignored regions", "units"),
    "goal": ("goal", "balanced", "rapid", "thorough", "analysis mode", "preset"),
    "magnetic": ("magnetic", "magnetism", "magnetic precheck"),
}
ORDINALS = {
    "first": 1, "one": 1, "second": 2, "two": 2, "third": 3, "three": 3,
    "fourth": 4, "four": 4, "fifth": 5, "five": 5, "sixth": 6, "six": 6,
    "seventh": 7, "seven": 7, "eighth": 8, "eight": 8, "ninth": 9, "nine": 9,
    "tenth": 10, "ten": 10, "eleventh": 11, "twelfth": 12,
}
ELEMENT_NAMES = dict(item.split(":") for item in """
hydrogen:H helium:He lithium:Li beryllium:Be boron:B carbon:C nitrogen:N oxygen:O
fluorine:F neon:Ne sodium:Na magnesium:Mg aluminium:Al aluminum:Al silicon:Si
phosphorus:P sulfur:S sulphur:S chlorine:Cl argon:Ar potassium:K calcium:Ca
scandium:Sc titanium:Ti vanadium:V chromium:Cr manganese:Mn iron:Fe cobalt:Co
nickel:Ni copper:Cu zinc:Zn gallium:Ga germanium:Ge arsenic:As selenium:Se
bromine:Br krypton:Kr rubidium:Rb strontium:Sr yttrium:Y zirconium:Zr niobium:Nb
molybdenum:Mo technetium:Tc ruthenium:Ru rhodium:Rh palladium:Pd silver:Ag
cadmium:Cd indium:In tin:Sn antimony:Sb tellurium:Te iodine:I xenon:Xe caesium:Cs
cesium:Cs barium:Ba lanthanum:La cerium:Ce praseodymium:Pr neodymium:Nd
promethium:Pm samarium:Sm europium:Eu gadolinium:Gd terbium:Tb dysprosium:Dy
holmium:Ho erbium:Er thulium:Tm ytterbium:Yb lutetium:Lu hafnium:Hf tantalum:Ta
tungsten:W rhenium:Re osmium:Os iridium:Ir platinum:Pt gold:Au mercury:Hg
thallium:Tl lead:Pb bismuth:Bi polonium:Po astatine:At radon:Rn francium:Fr
radium:Ra actinium:Ac thorium:Th protactinium:Pa uranium:U neptunium:Np
plutonium:Pu americium:Am curium:Cm berkelium:Bk californium:Cf einsteinium:Es
fermium:Fm mendelevium:Md nobelium:No lawrencium:Lr rutherfordium:Rf dubnium:Db
seaborgium:Sg bohrium:Bh hassium:Hs meitnerium:Mt darmstadtium:Ds roentgenium:Rg
copernicium:Cn nihonium:Nh flerovium:Fl moscovium:Mc livermorium:Lv tennessine:Ts
oganesson:Og deuterium:D tritium:T
""".split())


def normalize(text: str) -> str:
    # Preserve the original text separately for chemical symbols and identifiers.
    return " ".join(unicodedata.normalize("NFKC", text).casefold()
                    .replace("’", "'").replace("–", "-").replace("—", "-").split())


def _key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", normalize(text)).strip()


def _raw_body(text: str, *, trim_punctuation=True) -> str:
    trim = " .!?\"'" if trim_punctuation else " \"'"
    value = " ".join(text.replace("’", "'").split()).strip(trim)
    prefix = r"^(?:please|actually|instead|could you|can you|can we|let's|i would like to|i'd like to|i would like|i'd like|i want to|i want|i'll take|use|select|choose|apply|go with|switch to|change to|set the goal to|set goal to|set it to)\s+"
    while re.match(prefix, value, re.I):
        value = re.sub(prefix, "", value, count=1, flags=re.I)
    return re.sub(r"(?:\s+please|,?\s+thanks|\s+for me)[.!?]*$", "", value, flags=re.I).strip(trim)


def _body(text: str) -> str:
    return normalize(_raw_body(text))


def _clarify(message: str, choices=()) -> ParsedReply:
    return ParsedReply("clarify", message=message, choices=tuple(choices))


def _patch(values: dict, label: str, *, suggest=False) -> ParsedReply:
    if suggest:
        return _clarify("Please confirm: " + label + ".", [ReplyChoice(label, "patch", values)])
    return ParsedReply("patch", values, label)


def _near(query: str, choices: dict[str, str]) -> list[str]:
    """Small vocabulary, bounded edit distance, no matching across digits.

    A returned score is spelling distance, never a probability of correctness.
    Even a unique fuzzy result requires confirmation.
    """
    if len(query) < 4 or len(query) > 160:
        return []
    limit = 1 if len(query) < 9 else 2
    digits = re.findall(r"\d+", query)
    eligible = {key: value for key, value in choices.items()
                if re.findall(r"\d+", value) == digits}
    matches = process.extract(query, eligible, scorer=DamerauLevenshtein.distance,
                              score_cutoff=limit, limit=4)
    return [match[2] for match in matches]


def _ordinal(text: str) -> int | None:
    body = _body(text)
    body = re.sub(r"^(?:the\s+)?(?:option|number|no\.?|item)\s*#?\s*", "", body)
    body = re.sub(r"^the\s+|\s+(?:one|option|item)$", "", body)
    if body in ORDINALS:
        return ORDINALS[body]
    if re.fullmatch(r"[1-9]\d*(?:st|nd|rd|th)?", body):
        return int(re.match(r"\d+", body)[0])
    return None


def pending_answer(text: str, choices: tuple[ReplyChoice, ...]) -> ParsedReply | None:
    """Only the controller decides whether this pending receipt is still valid."""
    body = _body(text)
    if body in {"no", "nope", "cancel", "never mind", "nevermind", "neither", "none of these"}:
        return ParsedReply("cancel")
    if body in {"yes", "y", "yep", "yeah", "ok", "okay", "correct", "that's right", "confirm", "do that"}:
        if len(choices) == 1:
            choice = choices[0]
            return ParsedReply(choice.action, choice.value)
        return _clarify("Which choice do you mean? Select a button or its option number.", choices)
    aliases = []
    for choice in choices:
        if choice.action == "goal" and GOALS.get(body) == choice.value:
            aliases.append(choice)
        if choice.action == "patch" and body in BEAMLINES and choice.value.get("live_beamline") == BEAMLINES[body]:
            aliases.append(choice)
    if len(aliases) == 1:
        return ParsedReply(aliases[0].action, aliases[0].value)
    index = _ordinal(text)
    if index is not None:
        if 1 <= index <= len(choices):
            choice = choices[index - 1]
            return ParsedReply(choice.action, choice.value)
        return _clarify("That option is outside the displayed choices. Select one of the buttons.", choices)
    exact = [c for c in choices if _key(c.label) == _key(body)]
    if len(exact) == 1:
        return ParsedReply(exact[0].action, exact[0].value)
    return None


def _help(text: str) -> ParsedReply | None:
    lower = normalize(text)
    if not (lower in {"help", "why", "?"} or re.match(
            r"^(?:please\s+)?(?:why\b|help\b|explain\b|what\b|how\b|tell me about\b|can you explain\b|could you explain\b)", lower)):
        return None
    found = {topic for topic, aliases in TOPICS.items()
             if any(re.search(r"\b" + re.escape(alias) + r"\b", lower) for alias in aliases)}
    if len(found) == 1:
        return ParsedReply("help", found.pop())
    if len(found) > 1:
        return _clarify("Which setting would you like explained?", [
            ReplyChoice("Explain " + topic, "help", topic) for topic in sorted(found)])
    words = re.findall(r"[a-z]+", lower)
    vocabulary = {alias: alias for aliases in TOPICS.values() for alias in aliases if " " not in alias and len(alias) >= 4}
    near = {topic for word in words for match in _near(word, vocabulary)
            for topic, aliases in TOPICS.items() if match in aliases}
    if len(near) == 1:
        return ParsedReply("help", near.pop())  # An explanation cannot edit inputs.
    if lower in {"help", "why", "?"}:
        return ParsedReply("help")
    return _clarify("Choose a setting to explain.", [ReplyChoice("Explain " + t, "help", t) for t in HELP])


def _experiment(text: str, context: ReplyContext) -> ParsedReply | None:
    lower = normalize(text)
    matches = list(re.finditer(r"\bipts\s*(?:number|id)?\s*(?:is\s*)?[-:# ]*([1-9][0-9]{0,8})\b(?![\w/-]|\.[0-9])", lower))
    beams = list(re.finditer(r"\b(?:" + "|".join(re.escape(a) for a in sorted(BEAMLINES, key=len, reverse=True)) + r")\b", lower))
    tentative = []
    if context.step == "experiment" or matches:
        for match in re.finditer(r"\b[a-z]{4,9}\b", lower):
            if match[0] not in BEAMLINES and _near(match[0], BEAMLINE_WORDS):
                tentative.append(match)
    if not (matches or beams or tentative):
        if context.step == "experiment":
            bare = re.fullmatch(r"(?:my\s+)?(?:proposal(?:\s+number)?\s*(?:is\s*)?)?([1-9][0-9]{0,8})", _body(text))
            if bare:
                return _patch({"powgen_ipts": "IPTS-" + bare[1]}, "Use IPTS-" + bare[1])
            if re.search(r"\bipts\b", lower):
                return _clarify("Enter the IPTS digits exactly, for example IPTS-12345. I do not correct identifier digits.")
        return None
    ipts = list(dict.fromkeys("IPTS-" + m[1] for m in matches))
    beam_ids = list(dict.fromkeys(BEAMLINES[m[0]] for m in beams))
    if len(ipts) > 1 or len(beam_ids) > 1:
        return _clarify("I found more than one experiment or beamline. Specify one beamline and one IPTS.")
    values = {"powgen_ipts": ipts[0]} if ipts else {}
    if beam_ids:
        values["live_beamline"] = beam_ids[0]
    elif tentative:
        possible = {BEAMLINES[b] for m in tentative for b in _near(m[0], BEAMLINE_WORDS)}
        if len(possible) != 1:
            return _clarify("Which beamline did you mean?", [ReplyChoice(a.name, "patch", {**values, "live_beamline": code}) for code, a in LIVE_BEAMLINES.items()])
        values["live_beamline"] = possible.pop()
    residual = lower
    for match in sorted(matches + beams + tentative, key=lambda m: m.start(), reverse=True):
        residual = residual[:match.start()] + " " + residual[match.end():]
    fillers = {"i", "i'm", "am", "we", "we're", "are", "working", "work", "at", "on", "using", "use", "my", "our", "the", "beamline", "instrument", "is", "it's", "and", "with", "for", "experiment", "proposal", "number", "id", "please", "actually", "instead", "switch", "change", "to", "select", "choose", "set", "it", "this", "thanks"}
    unknown = [word for word in re.findall(r"[\w']+", residual) if word not in fillers]
    label = "Use " + " / ".join(([LIVE_BEAMLINES[values["live_beamline"]].name] if "live_beamline" in values else []) + ipts)
    if unknown:
        return _clarify("I understood the experiment values, but could not interpret the rest. Apply only the values shown here?", [ReplyChoice(label, "patch", values)])
    return _patch(values, label, suggest=bool(tentative))


def _sample_aliases(option: SampleOption) -> set[str]:
    aliases = {_key(option.title)} if _key(option.title) != _key(option.value) else set()
    parts = re.split(r"\s+[—–-]\s+", option.title, maxsplit=1)
    if len(parts) == 2:
        aliases.add(_key(parts[1]))
    return aliases


def _sample_names(option: SampleOption) -> set[str]:
    parts = re.split(r"\s+[—–-]\s+", option.title, maxsplit=1)
    name = _key(parts[1] if len(parts) == 2 else option.title)
    return {name} if name and name != _key(option.value) else set()


def _one_sample(text: str, context: ReplyContext) -> ParsedReply:
    raw = text.strip(" \"'")
    explicit_id = bool(re.match(r"^(?:with\s+)?ids?\s*[:# ]", raw, re.I))
    raw = re.sub(r"^(?:with\s+)?ids?\s*[:# ]\s*", "", raw, flags=re.I)
    exact_ids = [s for s in context.samples if s.value == raw]
    ordinal = None if explicit_id else _ordinal(raw)
    explicit_option = bool(re.search(r"\b(?:option|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b|\d(?:st|nd|rd|th)\b", raw, re.I))
    positional = context.shown_samples[ordinal - 1] if ordinal and 1 <= ordinal <= len(context.shown_samples) else None
    exact_names = [] if explicit_id else [s for s in context.samples if _key(raw) in _sample_aliases(s)]
    candidates = {s.value: s for s in exact_ids + exact_names}
    if positional and (not candidates or not explicit_id):
        if explicit_option:
            candidates = {positional.value: positional}
        else:
            candidates[positional.value] = positional
    if len(candidates) == 1:
        sample = next(iter(candidates.values()))
        return _patch({"powgen_processing_scope": "selected", "powgen_processing_sample_ids": [sample.value]}, "Sample " + sample.title)
    if candidates:
        return _clarify("That reply matches more than one sample. Which identity do you mean?", [ReplyChoice("Sample " + s.title, "patch", {"powgen_processing_scope": "selected", "powgen_processing_sample_ids": [s.value]}) for s in candidates.values()])
    # Fuzzy names only. Do not repair a sample ID or its digit sequence.
    choices = {}
    if not explicit_id and not ordinal:
        for i, sample in enumerate(context.samples):
            for j, alias in enumerate(sorted(_sample_names(sample))):
                choices[f"{i}:{j}"] = alias
        near = _near(_key(raw), choices)
        found = list(dict.fromkeys(int(key.split(":")[0]) for key in near))
        if found:
            return _clarify("Did you mean one of these samples?", [ReplyChoice("Sample " + context.samples[i].title, "patch", {"powgen_processing_scope": "selected", "powgen_processing_sample_ids": [context.samples[i].value]}) for i in found])
    return _clarify("I could not identify that sample. Use its exact ID/name, a displayed option number, or the sample control.")


def _merge(replies: list[ParsedReply], label: str) -> ParsedReply:
    """Keep a grouped answer atomic; never silently discard an invalid part."""
    variants = []
    for reply in replies:
        if reply.action == "patch":
            variants.append([ReplyChoice(reply.message, "patch", reply.value)])
        elif reply.choices and all(c.action == "patch" for c in reply.choices):
            variants.append(reply.choices)
        else:
            known = {key: value for r in replies if r.action == "patch" for key, value in r.value.items()}
            choices = [ReplyChoice("Apply only: " + "; ".join(r.message for r in replies if r.action == "patch"), "patch", known)] if known else []
            return _clarify(reply.message + " The grouped answer has not been applied.", choices)
    if all(r.action == "patch" for r in replies):
        values = {key: value for r in replies for key, value in r.value.items()}
        # A list of sample replies appends identities instead of overwriting.
        ids = [item for r in replies for item in r.value.get("powgen_processing_sample_ids", [])]
        if ids:
            values["powgen_processing_sample_ids"] = list(dict.fromkeys(ids))
        return _patch(values, label)
    count = 1
    for variant in variants:
        count *= len(variant)
    if count > 6:
        return _clarify("Several parts of that answer are ambiguous. Choose the samples and banks with the controls.")
    options = []
    for combination in product(*variants):
        values = {key: value for c in combination for key, value in c.value.items()}
        ids = [item for c in combination for item in c.value.get("powgen_processing_sample_ids", [])]
        if ids:
            values["powgen_processing_sample_ids"] = list(dict.fromkeys(ids))
        options.append(ReplyChoice("; ".join(c.label for c in combination), "patch", values))
    return _clarify("Please confirm the complete selection.", options)


def _samples(text: str, context: ReplyContext) -> ParsedReply:
    body = _body(text)
    literal = _one_sample(text, context) if context.samples else None
    if body in {"all", "all samples", "every sample", "all of them"}:
        if literal and literal.action == "patch":
            return _clarify("That is also a sample name. Do you mean this sample or every sample?", [ReplyChoice(literal.message, "patch", literal.value), ReplyChoice("All samples", "patch", {"powgen_processing_scope": "all", "powgen_processing_sample_ids": []})])
        return _patch({"powgen_processing_scope": "all", "powgen_processing_sample_ids": []}, "All samples, including unidentified scans")
    if not context.samples:
        return _clarify("Find samples for this experiment first, then choose from the returned sample list.")
    whole = literal
    if whole.action == "patch" or whole.choices:
        return whole
    parts = [part.strip() for part in re.split(r"\s*(?:,|;|\band\b|&)\s*", text, flags=re.I) if part.strip()]
    if len(parts) > 1:
        return _merge([_one_sample(p, context) for p in parts], "Select samples: " + ", ".join(parts))
    return whole


def _banks(text: str, context: ReplyContext) -> ParsedReply:
    body = _body(text)
    if body in {"all", "all banks", "every bank", "all of them"}:
        if context.beamline in {'NOM','SNAP'}:
            return _clarify(LIVE_BEAMLINES[context.beamline].name + " requires exactly one source histogram and an explicit profile-bank mapping. Select one bank.")
        return _patch({"powgen_bank_scope": "all", "powgen_processing_banks": []}, "All available banks, separately")
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", body)
    if not parts or any(not re.fullmatch(r"[1-9][0-9]*", p) for p in parts):
        return _clarify("Use exact bank numbers from the discovered list. I do not correct or round bank numbers.")
    banks = list(dict.fromkeys(int(p) for p in parts))
    if set(banks) - set(context.banks):
        return _clarify("A requested bank is not in the current bank list. Choose a discovered bank or refresh the experiment.")
    if context.beamline in {'NOM','SNAP'}:
        if len(banks) != 1:
            return _clarify("Choose exactly one NOMAD source histogram. Confirm its profile-bank mapping separately.")
        return _patch({"nomad_physical_bank": banks[0], "nomad_profile_acknowledged": False}, f"NOMAD source bank {banks[0]}; profile mapping requires review")
    return _patch({"powgen_bank_scope": "selected", "powgen_processing_banks": banks}, "Banks " + ", ".join(map(str, banks)))


def _selection(text: str, context: ReplyContext) -> ParsedReply | None:
    # Keep chemical symbols in their original case; selection names also retain IDs.
    body = _raw_body(text, trim_punctuation=False)
    # Parse a delimited pair as two fields before looking for keywords inside
    # its values. Exact IDs/names such as "oxide-sample" or "low T sample"
    # are values, not repeated commands. Unknown/ambiguous parts stay atomic.
    pair = re.fullmatch(r"samples?\s+(.*?)[;,]\s*(?:banks?|histograms?)\s+(.*)", body, re.I)
    if not pair and context.step=='selection':
        pair=re.fullmatch(r"(.+?)[;,]\s*(?:banks?|histograms?)\s+(.*)",body,re.I)
    if pair:
        return _merge([_samples(pair[1].strip(), context), _banks(pair[2].strip(), context)], 'Updated samples and banks')
    all_choices = re.fullmatch(r"(?:all|every)\s+(?:available\s+)?(samples?|banks?|histograms?)", body.strip(" .!?"), re.I)
    if all_choices:
        return _samples("all", context) if all_choices[1].lower().startswith("sample") else _banks("all", context)
    trailing_sample = re.fullmatch(r"((?:the\s+)?(?:\d+(?:st|nd|rd|th)?|" + "|".join(ORDINALS) + r"))\s+sample", body.strip(" .!?"), re.I)
    if context.step == "selection" and trailing_sample:
        return _samples(trailing_sample[1], context)
    matches = list(re.finditer(r"(?<![\w.-])(samples?|banks?|histograms?)(?![\w.-])\s*(?:are\b|is\b|:|=)?\s*", body, re.I))
    if not matches:
        if context.step != "selection":
            return None
        if _body(body) == "none":
            return _clarify("Which part do you want to change? A Live setup needs samples and banks to process.", [ReplyChoice("Choose samples", "focus", "samples"), ReplyChoice("Choose banks", "focus", "banks")])
        if _body(body) in {"all", "both"}:
            return _clarify("Do you mean samples or banks?", [ReplyChoice("All samples", "patch", {"powgen_processing_scope": "all", "powgen_processing_sample_ids": []}), ReplyChoice("Choose banks", "focus", "banks")])
        return _banks(body, context) if context.focus == "banks" else _samples(body, context)
    if body[:matches[0].start()].strip(" ,;"):
        return None
    replies = []
    kinds = []
    for i, match in enumerate(matches):
        kind = "samples" if match[1].lower().startswith("sample") else "banks"
        if kind in kinds:
            return _clarify("Specify the sample selection and bank selection once each, or use the controls.")
        kinds.append(kind)
        chunk = body[match.end():matches[i+1].start() if i+1 < len(matches) else len(body)].strip(" ,;")
        chunk = re.sub(r"\s+and$", "", chunk, flags=re.I).strip(" ,;")
        if not chunk:
            return ParsedReply("focus", kind, "Choose " + kind + " by name or displayed number.")
        replies.append(_samples(chunk, context) if kind == "samples" else _banks(chunk, context))
    return _merge(replies, "Updated " + " and ".join(kinds))


def _chemical_value(text: str, optional: bool) -> tuple[str, bool]:
    raw = text.strip(" .!?")
    if optional and normalize(raw) in {"none", "no", "nothing", "no environment", "no container", "no can"}:
        return "", False
    tokens = [t for t in re.split(r"[,;\s&]+", raw) if t and t.lower() != "and"]
    if not tokens:
        parse_elements("", optional=optional)
    result = []
    needs_confirmation = False
    for token in tokens:
        if token in _ELEMENT_SYMBOLS:
            result.append(token)
        elif token.casefold() in ELEMENT_NAMES:
            result.append(ELEMENT_NAMES[token.casefold()])
        elif token.islower() and len(token) <= 2 and token.capitalize() in _ELEMENT_SYMBOLS:
            # Lowercase symbols can be suggested, never silently reinterpreted.
            result.append(token.capitalize())
            needs_confirmation = True
        else:
            raise ValueError("Use separate element symbols or full element names; check: " + token + ". Formulas and ambiguous symbols are not guessed.")
    values = list(dict.fromkeys(result))
    parse_elements(", ".join(values), optional=optional)
    return ", ".join(values), needs_confirmation


def _chemistry(text: str, context: ReplyContext) -> ParsedReply | None:
    # 'can' is a role label only with punctuation or a chemical 'is/contains' verb.
    label = r"(?:sample(?:\s+elements)?|elements|environment(?:\s+elements)?|container|can)"
    delimiter = r"(?::|=|\bis\b|\bare\b|\bcontains\b|\bhas\b|\bconsists of\b)" + ("?" if context.step == "chemistry" else "")
    role = re.compile(r"\b(?:(?:the|my|our)\s+)?(" + label + r")\s*" + delimiter + r"\s*", re.I)
    raw = re.sub(r"^(?:(?:my|our|the|set)\s+)+", "", _raw_body(text), flags=re.I)
    matches = list(role.finditer(raw))
    if not matches:
        if context.step != "chemistry":
            return None
        if normalize(raw) in {"none", "no", "skip", "i don't know", "i do not know", "not sure", "unsure"}:
            return _clarify("Sample chemistry is required and may include impurities. You can use the CIF suggestion and review it; environment chemistry is optional.")
        value, suggest = _chemical_value(raw, optional=False)
        return _patch({"live_sample_elements": value}, "Sample elements: " + value, suggest=suggest)
    if raw[:matches[0].start()].strip(" ,;"):
        return None
    values = {}
    suggest = False
    for i, match in enumerate(matches):
        key = "live_sample_elements" if match[1].lower().startswith(("sample", "elements")) else "live_environment_elements"
        if key in values:
            return _clarify("Specify sample chemistry and environment chemistry once each.")
        chunk = raw[match.end():matches[i+1].start() if i+1 < len(matches) else len(raw)].strip(" ,;")
        chunk = re.sub(r"\s+and$", "", chunk, flags=re.I).strip(" ,;")
        values[key], correction = _chemical_value(chunk, optional=key == "live_environment_elements")
        suggest |= correction
    description = "; ".join(("Sample" if k == "live_sample_elements" else "Environment") + ": " + (v or "none") for k, v in values.items())
    return _patch(values, description, suggest=suggest)


def interpret_reply(text: str, context: ReplyContext | None = None) -> ParsedReply:
    context = context or ReplyContext()
    text = str(text or "").strip()
    if len(text) > MAX_REPLY:
        return _clarify("Please keep each reply under 1000 characters, or enter the settings with the controls.")
    if not text:
        return ParsedReply("empty")
    explanation = _help(text)
    if explanation:
        return explanation
    body = _body(text)
    if re.search(r"\b(?:start monitoring|submit|launch|execute|analy[sz]e (?:a |the )?(?:scan|run))\b", body):
        return _clarify("This assistant fills the Live setup. Use the ordinary monitoring controls after reviewing the configuration.")
    navigation = {"back": "back", "go back": "back", "previous": "back", "previous question": "back", "undo": "undo", "undo that": "undo", "undo last change": "undo", "continue": "continue", "next": "continue", "next question": "continue", "move on": "continue", "keep current settings": "continue", "keep my settings": "continue", "keep the current settings": "continue"}
    if body in navigation:
        return ParsedReply(navigation[body])
    if body in {"yes", "no", "ok", "okay", "yep", "yeah", "correct", "confirm", "cancel", "never mind", "nevermind"}:
        return _clarify("There is no pending yes/no question. Choose a setup option or say what you want to change.")
    if body in {"i don't know", "i do not know", "not sure", "unsure", "skip"}:
        return ParsedReply("help", message="You can use the controls and explanations to decide. Required inputs still need a value.")
    sections = {"experiment": "experiment", "beamline": "experiment", "ipts": "experiment", "samples": "selection", "banks": "selection", "structures": "structures", "library": "structures", "chemistry": "chemistry", "elements": "chemistry", "settings": "settings", "goal": "settings", "review": "review"}
    change = re.fullmatch(r"(?:change|edit|revisit|go to|back to)(?: the| my)?\s+(.+)", normalize(text).strip(" .!?"))
    if change and change[1] in sections:
        return ParsedReply("navigate", sections[change[1]])
    if body in {"no main phase", "no main cif", "no dominant phase", "no known main phase"}:
        return _patch({"powgen_main_cif_source": "none"}, "No supplied main phase")
    if context.step == "chemistry" and body in {"no environment", "no container", "no can"}:
        return _patch({"live_environment_elements": ""}, "No environment elements")
    replacement = re.fullmatch(r"(?:not|don't use|do not use)\s+([a-z]+)\s*(?:,\s*|\bbut\s+)(.+)", body)
    if replacement:
        positive = _body(replacement[2])
        if replacement[1] in GOALS and positive in GOALS:
            return ParsedReply("goal", GOALS[positive])
        if replacement[1] in BEAMLINES and positive in BEAMLINES:
            return _patch({"live_beamline": BEAMLINES[positive]}, "Use " + positive.upper())
    if re.search(r"\b(?:not|don't|dont|never|except|without|neither|nor)\b", body):
        return _clarify("I will not change the setup from a negative or exclusion instruction. State the value you do want, for example 'use balanced' or 'samples A and B'.")
    raw_body = _raw_body(text, trim_punctuation=False)
    explicit_chemistry = re.match(r"(?:my\s+|the\s+)?(?:sample(?:\s+elements)?|elements|environment|container|can)\s*(?::|=|\bis\b|\bare\b|\bcontains\b|\bhas\b)", raw_body, re.I)
    if explicit_chemistry:
        chemical = _chemistry(text, context)
        if chemical:
            return chemical
    if context.step != "chemistry" and re.match(r"(?:samples?|banks?|histograms?)\b", raw_body, re.I):
        selected = _selection(text, context)
        if selected:
            return selected
    if context.step == "selection" and any(raw_body == s.value or _key(raw_body) in _sample_aliases(s) for s in context.samples):
        return _selection(text, context)
    experiment = _experiment(text, context)
    if experiment:
        return experiment
    goal_body = re.sub(r"\s+(?:mode|goal|analysis|preset)$", "", body)
    if goal_body in GOALS:
        return ParsedReply("goal", GOALS[goal_body])
    near_goals = list(dict.fromkeys(GOALS[g] for g in _near(goal_body, {g: g for g in GOALS})))
    if near_goals:
        return _clarify("Did you mean this analysis goal?", [ReplyChoice("Use " + g, "goal", g) for g in near_goals])
    if body in {"default", "defaults"}:
        return _clarify("Keep the current settings, or apply the Balanced analysis goal?", [ReplyChoice("Keep current settings", "keep"), ReplyChoice("Use balanced", "goal", "balanced")])
    chemical = _chemistry(text, context)
    if chemical:
        return chemical
    selection = _selection(text, context)
    if selection:
        return selection
    if context.step == "structures" and body in {"built in", "built-in", "built in catalog", "built-in catalog", "mp/cod", "built-in database"}:
        return _patch({"live_database_source": "builtin"}, "Use the built-in MP/COD catalog")
    if context.step == "structures" and body in {"none", "no cif", "no cifs"}:
        return _clarify("Do you mean no dominant-phase CIF, or no candidate structures? A candidate catalog/library is still needed.", [ReplyChoice("No supplied main phase", "patch", {"powgen_main_cif_source": "none"})])
    nearby = _near(body, {word: word for word in ("continue", "back", "undo")})
    if nearby:
        return _clarify("Did you mean this setup action?", [ReplyChoice(word.capitalize(), word) for word in nearby])
    return ParsedReply("unknown", message="I could not interpret that reply. Use the choices for this question, give a sample name/number, or ask 'what does background mean?'.")
