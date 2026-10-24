"""Small scientific observation artifact from trusted pipeline GPX outputs."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def observations_from_project(path: Path, lattice, records, *, mode, input_provenance):
    histogram_profiles = []
    for record in records:
        if record and str(record[0][0]).startswith("PWDR"):
            for name, payload in record[1:]:
                if name == "Instrument Parameters":
                    # GSAS stores [initial value, refined value, refine flag].
                    parameters = payload[0] if isinstance(payload, list) else payload
                    histogram_profiles.append(
                        {
                            key: (value[0] if isinstance(value, (tuple, list)) else value)
                            for key, value in parameters.items()
                        }
                    )
    profile = (
        hashlib.sha256(
            json.dumps(histogram_profiles, sort_keys=True, default=str).encode()
        ).hexdigest()
        if histogram_profiles
        else None
    )
    return {
        "$schema": "radar-pd-observables/v1",
        "extractor_version": 1,
        "source_gpx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_selection": "pipeline final-project precedence; scientific acceptance requires review",
        "fraction_basis": "normalized modeled-crystalline mass; omitted/amorphous material not quantified",
        "context": {
            "profile_sha256": profile,
            "recipe_key": f"radar-pd/{mode}/observables-v1",
            "reduction_key": input_provenance.get("reduction_key")
            or input_provenance.get("resolver"),
        },
        "phases": lattice,
    }
