# NOMAD Live integration

Status: development branch, 2026-10-23. Not yet deployed or scientifically
accepted for automatic profile selection or quantitative phase analysis.

## Supported workflow

Choose **Live → NOMAD**, enter an IPTS, then refresh sample IDs. The archive
instrument code is `NOM`; the interface calls it NOMAD. Select all samples or
specific sample IDs and names. The initial-scan limit applies after sample
filtering; later matching reductions use the same selection.

The default source is the immediate folder
`/SNS/NOM/<IPTS>/shared/autoreduce/multi_banks/GSAS`. Only individual
`NOM_<run>.gsa` files trigger analysis. The legacy `shared/autoreduce` folder
is a separate selection when present. The adapter never recursively combines
these folders, summed products, `GSAS_unnorm`, XYE, S(Q), PDF or raw events.

Select exactly one physical bank per watch and provide a matching GSAS-II
`.instprm`. Enter the **Bank label declared inside the profile**, which can
differ from the physical bank number and the ordinal position of a block.
Confirm this mapping explicitly. The interface shows available bank angles
and approximate d ranges after checking the experiment. Bank 5 is the
backscattering bank in the inspected six-bank products; bank 6 is forward.
Unsupported file layouts fail explicitly.

NOMAD uses broadband neutron TOF. A `Wavelength: 1.5 A` GSA header or chopper
request is not a monochromatic wavelength or a POWGEN profile selector. An
HDF5 detector calibration is not a GSAS-II peak-shape profile. No automatic
NOMAD profile registry is advertised in this version.

## Scientific safeguards

Preflight checks every selected initial run. The filename, Sample Run header,
individual reduction JSON, bank labels/counts and complete FXYE records must
agree. TOF must increase, observations and uncertainties must be finite, and
uncertainties must be positive. Negative observed intensities are preserved.
Truncated or changing files cannot be submitted successfully and remain in
the shared retry lifecycle.

The profile must contain a complete neutron TOF parameter set with unique
declared Bank labels. Its DIFC must agree with the selected bank export within
0.5%, and its scattering angle within 5 degrees. These are structural
compatibility checks, **not scientific calibration acceptance**. A profile
with duplicated Bank labels is rejected, rather than silently repaired.

The original reduction input JSON supplies calibration and environment
provenance. Placeholder/unconfirmed sample corrections, an intensity
positivity shift, or missing correction provenance produce visible notices
and require explicit exploratory-analysis acknowledgement. `Sample.Material`
is never used to infer chemistry: inspected reductions used `Si` as a
placeholder for other materials. `Instrument: Dummy` in the GSAS export is
an export convention and is distinct from `Sample.DummyInfo` in the JSON.

One supplied profile cannot silently span different detector calibrations or
recorded sample environments within the initial selection or later in the watch. Split those selections
and verify each mapping. Missing calibration provenance remains exploratory.
All NOMAD results currently carry `needs_review`, including durable batch
summary JSON and HTML reports. A closed NOVA session does not remove notices.

## Bank staging and provenance

Analyze's existing importer reads one data block. NOVA stages the selected
physical bank as local `BANK 1` and remaps only the chosen profile's Bank label
to 1. It preserves the selected bank's export geometry, SLOG/FXYE convention,
numeric records and uncertainties. No normalization, clipping, interpolation
or square-root uncertainty reconstruction occurs during staging.

Receipts retain the physical and profile bank labels, source and profile
SHA256 values, numeric-record hash, staged hashes, original reduction input
hash, calibration and environment. A bounded header marker carries review
notices through Analyze into normalized output provenance. It can add review
notices; it cannot certify a dataset or clear pipeline warnings.

Discovery never automatically resubmits a replaced reduction. Existing
results retain their original provenance and receive a source-revision
warning. A pending run whose data or JSON changed after discovery is blocked
until a separately checked watch is created. A new watch must use distinct
Galaxy settings/history if the intent is explicit reprocessing, since
matching checkpoints are deliberately restored. NOMAD Galaxy run names and
idempotency tokens include input identity, preventing retry recovery from
attaching an earlier reduction of the same run.

## Metadata, recovery and future beamlines

Sample IDs come from NeXus sample identifiers and `SampleId`; conflicting
identifiers remain unidentified. Repeated names do not merge different IDs.
Temperature comes from the selected `SampleTemp` mean and recorded units,
with minimum/maximum when available. Celsius is explicitly converted to
kelvin. Inactive furnace or levitator channels and run-title numbers are not
temperature fallbacks. Run and IPTS identifiers are cross-checked when read.
Only small metadata fields are read; event arrays are never loaded.

`live_beamlines.py` registers discovery, metadata, preflight, settings and
controller behavior. Shared orchestration retains retries, submission limits,
Galaxy idempotency, checkpoints and dashboards. NOMAD state uses the generic
v2 schema; old POWGEN v1 checkpoints remain readable. Cross-beamline recovery
is rejected. Switching beamlines clears incompatible selections/results and
requires new preflight; it is disabled while monitoring.

## Validation and remaining acceptance work

Synthetic tests cover malformed/partial files, profile-label ambiguity,
non-first-bank staging, sample identity, temperature units, source replacement,
checkpoint recovery, calibration changes, UI resets and durable review flags.
Existing POWGEN and batch contract suites also run.

Local real-file import checks used Si run 181278 and sample run 248736.
For physical bank 5, staged RADAR-PD imports exactly matched native GSAS-II
imports in TOF, observed intensities, weights and profile coefficients.
The historical profile used for this comparison is **not** accepted for
those acquisition dates. Import equivalence alone does not establish correct
peak widths, lattice parameters or phase fractions. Private facility files
and local import artifacts are not committed as test fixtures.

Before deployment:

1. Obtain a verified profile for the current calibration, bank and sample
   environment, including its validity range and uncertainty in coefficients.
2. Benchmark an appropriate standard, checking positions, widths, residuals,
   d range and known lattice parameters through the exact batch-image importer.
3. Test a representative multiphase sample before accepting phase fractions.
4. Verify NOMAD mount/ACL access from a fresh NDIP NOVA session and Galaxy job;
   DGX access alone does not prove that access.
5. Run an end-to-end selected-bank analysis, dashboard, GPX handoff and GSAS-II
   export on NDIP-test. Build and deploy both NOVA and Analyze: the batch change
   preserves durable NOMAD review notices. Use the normal full-SHA image pins
   and verify both GitLab deployment jobs.

Joint multi-bank refinement, simultaneous multiple-bank watches, automatic
raw-event reduction, PDF analysis and summed-product revision processing are
outside this first adapter.

Facility references: [NOMAD autoreduction documentation](https://powder.ornl.gov/auto_reduce/nomad_auto.html)
and [NOMAD instrument/profile resources](https://neutrons-user-facilities.ornl.gov/instruments/nomad/).
