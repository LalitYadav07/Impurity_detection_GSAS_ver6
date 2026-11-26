"""Bounded declared NOMAD correction metadata; never inferred chemistry."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import re


SCHEMA = "radar-pd-nomad-corrections/v1"
MAX_CORRECTION_BYTES = 4096
MAX_FORMULA_LENGTH = 1024
_SPECIES = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D T".split())
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?|\.\d+)?")
_FIELDS = ("Material", "DummyInfo", "PackingFraction", "MassDensity", "Mass", "Geometry",
           "Background", "AbsorptionCorrection", "MultipleScatteringCorrection",
           "InelasticCorrection", "PushPositiveLevel")


def declared_corrections(config):
    """Keep explicit values and missing flags distinct, bound to the sidecar."""
    result = {"schema": SCHEMA, "status": "declared_in_reduction_input_not_execution_verified"}
    for section in ("Sample", "Vanadium"):
        value = config.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Invalid NOMAD {section} correction metadata")
        result[section] = {key: deepcopy(value[key]) for key in _FIELDS if key in value}
        result[section]["DummyInfo_present"] = "DummyInfo" in value
    processing = {key: deepcopy(config[key]) for key in
                  ("HighQLinearFitRange", "Merging", "Normalization", "PushDataPositive") if key in config}
    if processing:
        result["Processing"] = processing
    # Omitted, false and null must not become the same review decision.
    if len(json.dumps(result, allow_nan=False, ensure_ascii=True).encode()) > MAX_CORRECTION_BYTES:
        raise ValueError("NOMAD declared correction metadata exceeds the bounded provenance limit")
    return result


def strict_formula_species(value):
    """Accept only plain formulas, not aliases, masses, prose or guessed groups.

    The returned species set supports a narrow contradiction test. Equal sets
    do not establish stoichiometry, phase identity or isotope occupancy.
    D and T remain distinct from H.
    """
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FORMULA_LENGTH:
        return None
    text = re.sub(r"\s+", "", value)
    tokens = list(_TOKEN.finditer(text))
    if not tokens or "".join(token.group() for token in tokens) != text:
        return None
    if any(token[1] not in _SPECIES or (token[2] is not None and
           (not math.isfinite(float(token[2])) or float(token[2]) <= 0))
           for token in tokens):
        return None
    return sorted({token[1] for token in tokens})


def material_comparison(corrections, source_metadata):
    """Compare only the source's explicit formula field; never its name/title."""
    declared = (corrections.get("Sample") or {}).get("Material")
    source = source_metadata.get("sample_formula")
    result = {"reduction_material": declared, "source_formula": source,
              "source_field": source_metadata.get("formula_source", ""),
              "source_path": source_metadata.get("nexus_path", ""),
              "source_signature": deepcopy(source_metadata.get("source_signature")),
              "formula_source_verified": source_metadata.get("formula_source_verified") is True,
              "status": "unavailable", "identity_claim": False}
    if declared is None or source in (None, ""):
        result["reason"] = source_metadata.get("formula_review_reason") or "Both explicit material and authoritative formula are required"
        return result
    if not isinstance(source, str) or len(source) > MAX_FORMULA_LENGTH:
        raise ValueError("NOMAD source formula exceeds the bounded provenance limit")
    if not result["formula_source_verified"]:
        result.update(status="unverified_source", reason=source_metadata.get("formula_review_reason") or "Formula source run/proposal identity is unverified")
        return result
    expected, actual = strict_formula_species(declared), strict_formula_species(source)
    result.update(reduction_species=expected, source_species=actual)
    if expected is None or actual is None:
        result.update(status="unassessable", reason="Formula is not a plain chemical formula; no aliases or free-form text were interpreted")
    elif expected != actual:
        result.update(status="different_species", reason="Strictly parsed element/isotope sets differ")
    else:
        result.update(status="same_species_not_identity", reason="Equal species sets do not establish stoichiometry, phase identity or isotope occupancy")
    return result


def with_material_review(setup, source_metadata):
    assessment = material_comparison(setup.declared_corrections, source_metadata)
    warnings = list(setup.warnings)
    if assessment["status"] == "different_species":
        warnings.append(f"Declared NOMAD reduction material {assessment['reduction_material']!r} and authoritative sample formula {assessment['source_formula']!r} have different element/isotope sets; review the declared correction settings and verify their execution before interpretation")
    elif assessment["status"] == "unassessable":
        warnings.append("Declared NOMAD reduction material and source formula include free-form or unsupported text; chemistry agreement is unverified")
    elif assessment["status"] == "unverified_source":
        warnings.append("Formula source run/proposal identity is unverified; review reduction chemistry provenance")
    return replace(setup, material_comparison=assessment, warnings=tuple(dict.fromkeys(warnings)))
