"""Read and expose immutable experimental development evidence, without a fit."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(os.environ.get("RADAR_PD_PLANNER_EVIDENCE_ROOT", Path(__file__).parent / "data" / "powgen_planning_evidence"))
MANIFEST_SHA256 = "0d97d1fa629a0203af20dfebed2d5d053983787e3824c39a918600cd158df795"


def load_evidence(root=ROOT):
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError("Private development evidence is not part of the application package. Supply the authorized audit folder explicitly.")
    manifest_bytes = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("The packaged experimental evidence manifest changed")
    metadata = json.loads(manifest_bytes)
    array_bytes = (root / "arrays.npz").read_bytes()
    if hashlib.sha256(array_bytes).hexdigest() != metadata["arrays_sha256"]:
        raise ValueError("The packaged experimental evidence arrays changed")
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    for row in metadata["records"]:
        prefix = f"band{row['band']}_"
        n = row["native_bins"]
        edges = arrays[prefix + "d_edges_a"]
        if edges.shape != (n + 1,) or not np.all(np.diff(edges) > 0):
            raise ValueError("Invalid native measured bin edges")
        for name in ("sample_counts", "can_counts"):
            value = arrays[prefix + name]
            if value.shape != (n,) or value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError("Measured integer counts must remain intact")
        for name in ("expected_total_rate", "expected_sample_rate"):
            value = arrays[prefix + name]
            if value.shape != (n,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError("Invalid frozen prediction")
        weights = arrays[prefix + "peak_weights"]
        if weights.shape != (row["peak_contrasts"], n) or not np.all(np.isfinite(weights)):
            raise ValueError("Invalid original peak-contrast weights")
        calculated = comparison(metadata, arrays, row["band"])
        for name, expected in row["expected_metrics"].items():
            if not np.isclose(calculated["metrics"][name], expected, rtol=0, atol=2e-13):
                raise ValueError("Packaged counts do not reproduce the archived comparison")
    return metadata, arrays, manifest_bytes, array_bytes


def comparison(metadata, arrays, band):
    row = next((item for item in metadata["records"] if item["band"] == int(band)), None)
    if row is None:
        raise ValueError("Choose one of the three packaged wavelength comparisons")
    prefix = f"band{row['band']}_"
    sample, can = arrays[prefix + "sample_counts"], arrays[prefix + "can_counts"]
    q, can_q = row["charge_uah"], row["can_charge_uah"]
    total = sample / q
    observed_sample = total - can / can_q
    expected_total, expected_sample = arrays[prefix + "expected_total_rate"], arrays[prefix + "expected_sample_rate"]
    weights = arrays[prefix + "peak_weights"]
    measured_peaks, expected_peaks = weights @ observed_sample, weights @ expected_sample
    total_variance = sample / q**2
    sample_variance = total_variance + can / can_q**2
    edges = arrays[prefix + "d_edges_a"]
    return dict(record=row, d=(edges[:-1] + edges[1:]) / 2,
                total=total, sample=observed_sample, expected_total=expected_total, expected_sample=expected_sample,
                total_sigma=np.sqrt(total_variance), sample_sigma=np.sqrt(sample_variance),
                peaks=measured_peaks, expected_peaks=expected_peaks,
                peaks_sigma=np.sqrt((weights**2) @ sample_variance),
                metrics=dict(total_relative_error=float(expected_total.sum() / total.sum() - 1),
                             sample_relative_error=float(expected_sample.sum() / observed_sample.sum() - 1),
                             peak_sum_relative_error=float(expected_peaks.sum() / measured_peaks.sum() - 1)))


def summary_rows(metadata, arrays):
    rows = []
    for record in metadata["records"]:
        data = comparison(metadata, arrays, record["band"])
        rows.append(dict(band=record["band"], wavelength=f"{record['wavelength_a']:g} Å",
                         run=f"PG3_{record['sample_run']}", bins=f"{record['native_bins']:,}",
                         peaks=str(record["peak_contrasts"]),
                         **{name: f"{100 * value:+.2f}%" for name, value in data["metrics"].items()}))
    return rows


def comparison_figure(data, quantity):
    if quantity not in {"total", "sample", "peaks"}:
        raise ValueError("Choose total counts, can-subtracted sample counts or peak contrasts")
    x = np.arange(1, len(data["peaks"]) + 1) if quantity == "peaks" else data["d"]
    observed, expected = data[quantity], data["expected_" + quantity]
    sigma = data[quantity + "_sigma"]
    name = {"total": "Total counts", "sample": "Can-subtracted sample", "peaks": "Fixed peak contrasts"}[quantity]
    traces = [dict(x=x.tolist(), y=observed.tolist(), name="Measured", mode="markers",
                   marker=dict(size=5 if quantity == "peaks" else 2, color="#406788"),
                   error_y=dict(type="data", array=sigma.tolist(), visible=True, thickness=.5)),
              dict(x=x.tolist(), y=expected.tolist(), name="Frozen prediction", mode="lines+markers" if quantity == "peaks" else "lines",
                   line=dict(color="#bd5b37", width=1.4)),
              dict(x=x.tolist(), y=(observed - expected).tolist(), name="Measured − predicted", mode="markers",
                   marker=dict(size=4 if quantity == "peaks" else 2, color="#555e68"), xaxis="x2", yaxis="y2", showlegend=False)]
    unit = "counts / μAh" if quantity == "peaks" else "counts / μAh / native bin"
    layout = dict(height=560, margin=dict(l=92, r=25, t=70, b=65), template="plotly_white",
                  title=dict(text=f"Silicon · {data['record']['wavelength_a']:g} Å · {name}", font=dict(size=15)),
                  xaxis=dict(domain=[0, 1], anchor="y", showticklabels=False),
                  yaxis=dict(domain=[.31, 1], title=dict(text=unit)),
                  xaxis2=dict(domain=[0, 1], anchor="y2", matches="x",
                              title=dict(text="Peak index" if quantity == "peaks" else "d spacing (Å)")),
                  yaxis2=dict(domain=[0, .22], anchor="x2", title=dict(text="Difference"), zeroline=True, zerolinecolor="#888"),
                  legend=dict(orientation="h", x=0, y=1.07),
                  uirevision=f"silicon-evidence-{data['record']['band']}-{quantity}")
    return {"data": traces, "layout": layout}


def export_evidence(root=ROOT):
    metadata, arrays, manifest_bytes, array_bytes = load_evidence(root)
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("arrays.npz", array_bytes)
        archive.writestr("README.txt", (
            "POWGEN 2026B silicon development comparisons\n\n"
            "Three acquisitions in one cycle; no independently verified absolute acceptance cases.\n"
            "These are frozen development observations and predictions, not a validation of an uploaded sample.\n"
            "No target mass or scale was fitted. Recorded sample mass/fill were not independently verified.\n"
            "Original measured integer counts, native d-bin edges and signed peak weights are in arrays.npz.\n"
            "The CSV files retain every native bin in the fixed comparison range, including negative can-subtracted values.\n"
            "Error bars are conditional measurement count errors, not full predictive intervals.\n"
            "See manifest.json for source hashes, doses, assumptions, metrics and outstanding physics/validation.\n"
            "With Python and NumPy installed, run: python reproduce.py\n"))
        archive.writestr("reproduce.py", '''import json
import os
import numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
m=json.loads((p/'manifest.json').read_text())
with np.load(p/'arrays.npz',allow_pickle=False) as a:
    for r in m['records']:
        k=f"band{r['band']}_"
        y=a[k+'sample_counts']/r['charge_uah']
        s=y-a[k+'can_counts']/r['can_charge_uah']
        t=a[k+'expected_total_rate'];f=a[k+'expected_sample_rate'];w=a[k+'peak_weights']
        print(r['wavelength_a'],dict(total_relative_error=float(t.sum()/y.sum()-1),sample_relative_error=float(f.sum()/s.sum()-1),peak_sum_relative_error=float((w@f).sum()/(w@s).sum()-1)))
''')
        for record in metadata["records"]:
            band = record["band"]
            data = comparison(metadata, arrays, band)
            csv = io.StringIO()
            columns = np.column_stack([data["d"], arrays[f"band{band}_sample_counts"], arrays[f"band{band}_can_counts"],
                                       data["total"], data["expected_total"], data["total_sigma"],
                                       data["sample"], data["expected_sample"], data["sample_sigma"]])
            np.savetxt(csv, columns, delimiter=",", comments="",
                       fmt=["%.17g", "%d", "%d"] + ["%.17g"] * 6,
                       header="d_a,measured_sample_counts,measured_can_counts,measured_total_counts_per_uah_bin,predicted_total_counts_per_uah_bin,measured_total_count_only_sigma,measured_sample_minus_can_counts_per_uah_bin,predicted_sample_counts_per_uah_bin,measured_sample_minus_can_count_only_sigma")
            archive.writestr(f"band{band}-native-bins.csv", csv.getvalue())
    return result.getvalue()
