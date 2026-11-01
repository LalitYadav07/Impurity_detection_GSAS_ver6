"""Build a new, immutable-input joint-bank model from reviewed sources.

This is preparation, not calibration acceptance or a scientific result. Every
bank, profile, phase role, link and starting scale is explicit. No fit or phase
selection happens here. Native refinement uses the existing typed recipe worker.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil

from ..contracts import exact_keys

SCHEMA = "radar-battery-preparation/v1"
ROLES = {"cathode", "anode", "current_collector", "electrolyte", "cell_component", "other"}


def validate_sample_widths(widths):
    """Explicit isotropic starts, with units independent of native field names."""
    exact_keys(widths, {"size_um", "microstrain_ustrain"}, set())
    for key, value in widths.items():
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("Sample-width starts must be finite numbers with the displayed units")
        if key == "size_um" and not .001 <= value <= 10:
            raise ValueError("Starting sample size must be between 0.001 and 10 µm, the supported native storage interval")
        if key == "microstrain_ustrain" and value < 0:
            raise ValueError("Starting sample microstrain must be nonnegative in µstrain")
    return deepcopy(widths)


def set_sample_widths(hap, widths):
    """Change only explicitly declared isotropic values; freezing is separate."""
    for key, value in validate_sample_widths(widths).items():
        native = "Size" if key == "size_um" else "Mustrain"
        if not hap.get(native) or hap[native][0] != "isotropic":
            raise ValueError("Explicit sample-width starts require an isotropic native model")
        hap[native][1][0] = float(value)


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def input_path(root, record):
    exact_keys(record, {"path", "sha256"}, {"path", "sha256"})
    name = record["path"]
    if not isinstance(name, str) or not name or "\\" in name or PurePosixPath(name).is_absolute() or any(p in {".", ".."} for p in name.split("/")):
        raise ValueError("Preparation sources must be confined relative file paths")
    root = Path(root).resolve()
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Preparation source is outside its bundle or exceeds 64 MiB")
    if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or file_hash(path) != record["sha256"]:
        raise ValueError("Preparation source changed: " + name)
    return path


def _text(value, label, maximum=300):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Provide a bounded " + label)


def validate_preparation(spec, root):
    from radar_pd_nova.battery.electrochemistry import utc_time
    exact_keys(spec, {"schema", "acquisition", "reduction_family", "banks", "phases", "review_note", "electrochemistry", "reduction_evidence", "catalog_context"},
                    {"schema", "acquisition", "reduction_family", "banks", "phases", "review_note"})
    if spec["schema"] != SCHEMA:
        raise ValueError("Unsupported battery preparation schema")
    acquisition = spec["acquisition"]
    exact_keys(acquisition, {"instrument", "ipts", "run", "sample_id", "cell_id", "start_utc", "end_utc", "metadata_sha256"},
                           {"instrument", "ipts", "run", "sample_id", "cell_id", "start_utc", "end_utc", "metadata_sha256"})
    if acquisition["instrument"] not in {"NOM", "PG3"}:
        raise ValueError("Joint-bank preparation supports individual NOMAD and POWGEN SLOG/FXYE products")
    if not re.fullmatch(r"IPTS-\d+", str(acquisition["ipts"])) or type(acquisition["run"]) is not int or acquisition["run"] < 1:
        raise ValueError("A proposal and positive integer run are required")
    for key in ("sample_id", "cell_id"):
        _text(acquisition[key], key, 128)
    if not re.fullmatch(r"[0-9a-f]{64}", str(acquisition["metadata_sha256"])):
        raise ValueError("Bind the reviewed acquisition metadata digest")
    if utc_time(acquisition["end_utc"]) <= utc_time(acquisition["start_utc"]):
        raise ValueError("A positive, timezone-qualified exposure is required")
    if 'electrochemistry' in spec:
        from .electrochemical_evidence import validate_electrochemistry
        validate_electrochemistry(spec['electrochemistry'],root,acquisition)
    if 'catalog_context' in spec:
        from .catalog_evidence import read_catalog
        read_catalog(spec['catalog_context'],root,acquisition)
    _text(spec["reduction_family"], "reduction family")
    _text(spec["review_note"], "preparation review note", 3000)
    if not isinstance(spec["banks"], list) or not 1 <= len(spec["banks"]) <= 6:
        raise ValueError("Select one to six banks for this exposure")
    keys, physical, source_digests = set(), set(), set()
    for bank in spec["banks"]:
        exact_keys(bank, {"key", "physical_bank", "data", "profile", "profile_bank", "fit_d", "profile_review", "invalid_bin_policy", "profile_origin"},
                         {"key", "physical_bank", "data", "profile", "profile_bank", "fit_d", "profile_review"})
        if not isinstance(bank["key"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", bank["key"]) or bank["key"] in keys:
            raise ValueError("Bank keys must be unique safe identifiers")
        keys.add(bank["key"])
        if type(bank["physical_bank"]) is not int or bank["physical_bank"] not in range(1, 7) or bank["physical_bank"] in physical:
            raise ValueError("A physical bank may appear only once in a joint observation")
        physical.add(bank["physical_bank"])
        if type(bank["profile_bank"]) is not int or bank["profile_bank"] < 1:
            raise ValueError("Select an explicit instrument Bank label")
        _text(bank["profile_review"], "profile mapping review", 3000)
        origin = bank.get('profile_origin', 'explicit')
        if not isinstance(origin, str) or origin not in {'explicit', 'powgen_catalog'} or (origin == 'powgen_catalog' and acquisition['instrument'] != 'PG3'):
            raise ValueError('Unsupported profile origin for this instrument')
        policy = bank.get('invalid_bin_policy', 'reject')
        if not isinstance(policy, str) or policy not in {'reject', 'outside_fit_only'}:
            raise ValueError('Invalid bins require rejection or explicit exclusion outside the fit range')
        interval = bank["fit_d"]
        if not isinstance(interval, list) or len(interval) != 2 or any(type(v) not in {int, float} or not math.isfinite(v) for v in interval) or not 0 < interval[0] < interval[1]:
            raise ValueError("Each bank needs increasing positive fit limits in Angstrom")
        input_path(root, bank["data"]); input_path(root, bank["profile"])
        source_digests.add(bank["data"]["sha256"])
    if len(source_digests) != 1:
        raise ValueError("Joint banks must come from the same reduced exposure and revision; do not mix reduction families")
    # Sidecars can select a different calibration even when diffraction bytes
    # are unchanged. New captures seal both their presence and content.
    if acquisition['instrument'] == 'PG3' or 'reduction_evidence' in spec:
        evidence = spec.get('reduction_evidence')
        if not isinstance(evidence, list) or len(evidence) > 2:
            raise ValueError('Record the finite reduction-evidence list, including an empty list when unavailable')
        paths = [input_path(root, row) for row in evidence]
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate reduction-evidence files')
        source_paths = {input_path(root, bank['data']) for bank in spec['banks']}
        if len(source_paths) != 1:
            raise ValueError('Captured reduction evidence requires one source file')
        source = next(iter(source_paths))
        suffixes = ('.py', '.xml') if acquisition['instrument'] == 'PG3' else ('.json',)
        present = {source.with_suffix(suffix).resolve() for suffix in suffixes if source.with_suffix(suffix).exists()}
        if set(paths) != present:
            raise ValueError('Reduction evidence is missing, unrecorded, or not adjacent to the captured observation')
    if not isinstance(spec["phases"], list) or not 1 <= len(spec["phases"]) <= 24:
        raise ValueError("Select one to 24 explicit phase hypotheses")
    names, phase_keys, linked = set(), set(), set()
    for phase in spec["phases"]:
        exact_keys(phase, {"key", "name", "role", "cif", "links", "hypothesis_note"},
                          {"key", "name", "role", "cif", "links", "hypothesis_note"})
        _text(phase["name"], "phase name", 128); _text(phase["hypothesis_note"], "phase hypothesis note", 3000)
        if not isinstance(phase["key"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", phase["key"]) or phase["key"] in phase_keys or phase["name"] in names:
            raise ValueError("Phase names and keys must be unique")
        phase_keys.add(phase["key"]); names.add(phase["name"])
        if phase["role"] not in ROLES:
            raise ValueError("Assign an explicit physical role to every phase")
        input_path(root, phase["cif"])
        if not isinstance(phase["links"], list) or not 1 <= len(phase["links"]) <= len(keys):
            raise ValueError("Each phase needs explicit bank links and starting HAP scales")
        seen = set()
        for link in phase["links"]:
            exact_keys(link, {"bank", "hap_scale", "sample_widths"}, {"bank", "hap_scale"})
            scale = link["hap_scale"]
            if link["bank"] not in keys or link["bank"] in seen or type(scale) not in {int, float} or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Each phase/bank link needs one positive starting HAP scale, not a mass fraction")
            if "sample_widths" in link:
                validate_sample_widths(link["sample_widths"])
            seen.add(link["bank"]); linked.add(link["bank"])
    if linked != keys:
        raise ValueError("Every bank must have at least one explicit phase link")
    return deepcopy(spec)


def array_fingerprints(histogram):
    import numpy as np
    result = {}
    for index, label in enumerate(("x", "observed", "weight")):
        array = np.ma.asarray(histogram.data["data"][1][index])
        values = np.ascontiguousarray(array.data, dtype="<f8")
        mask = np.ascontiguousarray(np.ma.getmaskarray(array), dtype="u1")
        if not np.all(np.isfinite(values)):
            raise ValueError("Native import produced nonfinite observed arrays")
        result[label] = dict(count=int(values.size), sha256=sha256(values.tobytes()).hexdigest(),
                             mask_sha256=sha256(mask.tobytes()).hexdigest())
    return result


def preserve_tof_observations(histogram, staged_data, invalid_bins=()):
    """Audit native TOF/bin-width conversion and retain signed observations.

    GSAS's general FXYE reader discards nonpositive intensities. A strictly
    checked reduced SLOG/FXYE observation with positive sigma remains usable.
    The supported bin convention is checked against the executing importer;
    an importer change fails visibly rather than shifting the data silently.
    """
    import numpy as np
    rows, banks = [], 0
    with Path(staged_data).open() as stream:
        for line in stream:
            if line.startswith("BANK"):
                banks += 1
                if not re.fullmatch(r"BANK\s+1\s+\d+\s+\d+\s+SLOG\s+.*\s+FXYE\s*", line):
                    raise ValueError("Expected one staged SLOG/FXYE bank")
            elif banks and line.strip() and not line.lstrip().startswith("#"):
                rows.append([float(v) for v in line.split()])
    raw = np.asarray(rows, dtype=float)
    if banks != 1 or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 3 or not np.all(np.isfinite(raw)) or np.any(raw[:, 2] <= 0):
        raise ValueError("Staged TOF observations need finite values and positive uncertainties")
    widths = np.diff(raw[:, 0])
    if np.any(widths <= 0):
        raise ValueError("Staged TOF coordinates must increase")
    expected = (raw[:-1, 0]+widths/2, raw[:-1, 1]/widths, (widths/raw[:-1, 2])**2)
    actual = histogram.data["data"][1]
    if np.shape(actual[0]) != expected[0].shape or not np.allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10):
        raise ValueError("Native importer changed the expected TOF bin convention")
    positive = raw[:-1, 1] > 0
    if any(not np.allclose(np.asarray(actual[i])[positive], expected[i][positive], rtol=1e-11, atol=1e-12) for i in (1, 2)):
        raise ValueError("Native importer applied an unexpected intensity/uncertainty transformation")
    for i in (1, 2):
        actual[i] = np.ma.array(expected[i], mask=np.zeros(expected[i].shape, dtype=bool))
    ignored=[row['record_index'] for row in invalid_bins if row['record_index']<len(actual[0])]
    if ignored:
        actual[1].mask[ignored]=True
        actual[2][ignored]=0.
    return dict(method="GSAS_TOF_bin_centres_and_density_with_signed_observations",
                source_records=len(raw), native_bins=len(widths),
                nonpositive_observations_retained=int(np.count_nonzero(~positive))-len(ignored),
                invalid_bin_indices_excluded=ignored,
                final_source_record="right bin edge; no following width, matching native GSAS import")


# Compatibility for previously imported preparation helpers.
preserve_nomad_observations = preserve_tof_observations


def verify_excluded_bins_outside_fit(histogram, invalid_bins):
    from ..fit_domain import native_powder_fit_mask
    coordinates=histogram.data['data'][1][0]
    fitted=native_powder_fit_mask(coordinates,histogram.data['Limits'][1],histogram.data['Limits'][2:])
    inside=[row['record_index'] for row in invalid_bins if row['record_index']<len(coordinates)
            and fitted[row['record_index']]]
    if inside:
        raise ValueError(f'{len(inside)} invalid intensity/uncertainty bins lie inside the selected fit range; narrow the limits or obtain a corrected reduction')


def prepare_model(spec, root, destination):
    from radar_pd_nova import nomad
    from radar_pd_nova.battery import powgen
    from ..advanced import native_interval
    from ..execution import freeze, write_json
    from ..snapshot import gsas, runtime_info, snapshot, json_safe
    from ..symmetry.adapters.gsas_cif_import import add_verified_phase
    from .profiles import native_profile
    spec = validate_preparation(spec, root)
    adapter = nomad if spec['acquisition']['instrument'] == 'NOM' else powgen
    out = Path(destination).resolve()
    # Never overwrite an earlier checkpoint or mix a new model into its bundle.
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="radar-battery-preparation-result/v1", status="preparing", specification=spec,
                   calibration_status="requires_scientific_validation", scientific_qualification="not_evaluated")
    try:
        if 'catalog_context' in spec:
            from .catalog_evidence import read_catalog, catalog_summary
            receipt['catalog_context']=catalog_summary(read_catalog(spec['catalog_context'],root,spec['acquisition']),spec['catalog_context'])
        if 'electrochemistry' in spec:
            from .electrochemical_evidence import reproduce_electrochemistry
            receipt['electrochemistry']=reproduce_electrochemistry(spec['electrochemistry'],root,spec['acquisition'])
        G2 = gsas()
        project = G2.G2Project(newgpx=str(out / "prepared.gpx"))
        histograms, bank_records = {}, []
        for bank in spec["banks"]:
            source, profile = input_path(root, bank["data"]), input_path(root, bank["profile"])
            allow_invalid=bank.get('invalid_bin_policy','reject')=='outside_fit_only'
            setup = adapter.inspect_reduced_setup(source, selected_bank=bank["physical_bank"],inventory_only=allow_invalid)
            profile_resolution = None
            if bank.get('profile_origin') == 'powgen_catalog':
                from radar_pd_nova.powgen_setup import text_hash
                resolved = powgen.catalog_profile(setup, bank['physical_bank'])
                if resolved.profile_bank != bank['profile_bank'] or text_hash(profile.read_bytes()) != resolved.sha256:
                    raise ValueError('Captured profile no longer matches the POWGEN reduction metadata and Bank mapping')
                profile_resolution = resolved.as_dict()
            profile, profile_conversion = native_profile(profile,bank['profile_bank'],out/'profiles'/(bank['key']+'.instprm'),G2)
            if setup.run_number != spec["acquisition"]["run"]:
                raise ValueError("Reduced file run conflicts with the reviewed acquisition")
            options = {'acknowledged': True} if adapter is nomad else {}
            data, inst, provenance = adapter.stage_bank(source, setup, bank["physical_bank"], profile,
                bank["profile_bank"], out / "banks" / bank["key"], **options,
                acquisition=spec["acquisition"],allow_invalid_counts=allow_invalid)
            histogram = project.add_powder_histogram(str(data), str(inst), fmthint="GSAS powder", databank=1, instbank=1)
            if histogram.InstrumentParameters["Type"][0] != "PNT":
                raise ValueError("The native importer did not preserve neutron TOF geometry")
            invalid_bins=provenance.get('invalid_count_bins',[])
            conversion = preserve_tof_observations(histogram, data,invalid_bins)
            histogram.set_refinements({"Limits": native_interval(histogram, *bank["fit_d"], "d", project)})
            verify_excluded_bins_outside_fit(histogram,invalid_bins)
            histograms[bank["key"]] = histogram
            bank_records.append(dict(key=bank["key"], histogram=histogram.name, physical_bank=bank["physical_bank"],
                profile_bank=bank["profile_bank"], fit_d=bank["fit_d"], arrays=array_fingerprints(histogram),
                instrument=json_safe(histogram.InstrumentParameters), provenance=provenance, conversion=conversion,
                profile_conversion=profile_conversion, profile_resolution=profile_resolution))
        phase_records = []
        for phase in spec["phases"]:
            path = input_path(root, phase["cif"])
            copied = out / "cifs" / (phase["key"] + ".cif")
            copied.parent.mkdir(exist_ok=True); shutil.copyfile(path, copied)
            native = add_verified_phase(project, copied, phasename=phase["name"],
                histograms=[histograms[link["bank"]] for link in phase["links"]], fmthint="CIF", useNet=False)
            for link in phase["links"]:
                hap = native.data["Histograms"][histograms[link["bank"]].name]
                hap["Scale"] = [float(link["hap_scale"]), False]
                set_sample_widths(hap, link.get("sample_widths", {}))
            phase_records.append(dict(key=phase["key"], name=native.name, role=phase["role"],
                                      hypothesis_note=phase["hypothesis_note"], links=phase["links"]))
        freeze(project)
        # Retain both explicit and imported width inputs, including model shape
        # and mixing terms. A save/reopen must not alter any of these starts.
        for record in phase_records:
            native = project.phase(record["name"])
            record["starting_hap"] = [dict(bank=link["bank"], **{
                key: json_safe(native.data["Histograms"][histograms[link["bank"]].name][key])
                for key in ("Scale", "Size", "Mustrain")}) for link in record["links"]]
        project.save()
        reopened = G2.G2Project(gpxfile=str(out / "prepared.gpx"))
        for record in bank_records:
            histogram = reopened.histogram(record["histogram"])
            if array_fingerprints(histogram) != record["arrays"] or json_safe(histogram.InstrumentParameters) != record["instrument"]:
                raise ValueError("Saved model changed imported observations, weights or instrument settings")
        for record in phase_records:
            native = reopened.phase(record["name"])
            for start in record["starting_hap"]:
                hap = native.data["Histograms"][histograms[start["bank"]].name]
                if any(json_safe(hap[key]) != start[key] for key in ("Scale", "Size", "Mustrain")):
                    raise ValueError("Saved model changed the reviewed phase scale or sample-width starts")
        validate_preparation(spec, root)  # All original inputs still match.
        receipt.update(status="prepared_needs_review", banks=bank_records, phases=phase_records,
                       runtime=runtime_info(G2, reopened), project_sha256=file_hash(out / "prepared.gpx"))
        write_json(out / "context.json", snapshot(out / "prepared.gpx"))
        write_json(out / "preparation.json", receipt)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        if (out / "prepared.gpx").exists():
            (out / "prepared.gpx").rename(out / "incomplete.gpx")
        write_json(out / "preparation.json", receipt)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest"); parser.add_argument("destination")
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Preparation manifest exceeds 2 MiB")
    result = prepare_model(json.loads(path.read_text()), path.parent, args.destination)
    print(json.dumps({"status": result["status"], "banks": len(result["banks"]), "phases": len(result["phases"])}))


if __name__ == "__main__":
    main()
