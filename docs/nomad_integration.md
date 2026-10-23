# NOMAD Live integration

Status: deployed to NDIP-test on 2026-10-23 as Interactive 0.4.0 and Analyze
0.5.16. NOMAD remains exploratory and is not scientifically accepted for
automatic profile selection or quantitative phase analysis.

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
declared Bank labels. Calibrated DIFC is checked within 5% of nominal export
geometry, with the scattering angle within 5 degrees. Differences above 0.5%
produce a durable review notice. The published 2019B bank-2 and bank-3
profiles differ from nominal geometry by about 0.84% and 2.44%, respectively;
the previous 0.5% rejection threshold incorrectly excluded them. These are
wrong-bank sanity checks, **not scientific calibration acceptance**. A profile
with duplicated Bank labels is rejected, rather than silently repaired.

## Battery discovery safeguards

Carbon explicitly included in sample chemistry now admits elemental carbon
candidates such as graphite. The previous generic C-only exclusion could
silently remove a supplied graphite CIF. Oxygen-only phases remain excluded
by the generic policy unless explicitly overridden in the pipeline settings.

For neutron lattice-nudge exports, oxidation-state decorations are removed
from an exported structure copy; elements, occupancies and the source model
are preserved. This prevents GSAS-II from substituting hydrogen for unsupported
fractional ionic atom labels. X-ray ionic labels retain their existing behavior.

Joint scale/background refinement now keeps finite, non-worsening one-cycle
checkpoints, including the matching fraction listing. Failed solver iterates
cannot replace the last accepted model. Phase acceptance and final polishing
also reject Rwp regressions. Full mode can consider up to eight phases using
an automatic starting phase and seven phase-addition passes; candidates still
have to survive screening and refinement rather than being forced present.

Recovered Galaxy jobs read their exact saved configuration through the deployed
configuration-override parameter, rather than showing the UI defaults for
limits and pass counts. These safeguards apply beyond the battery dataset;
private patterns and CIFs are not added to the repository as test fixtures.

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

Before automatic profile selection or quantitative acceptance:

1. Obtain a verified profile for the current calibration, bank and sample
   environment, including its validity range and uncertainty in coefficients.
2. Benchmark an appropriate standard, checking positions, widths, residuals,
   d range and known lattice parameters through the exact batch-image importer.
3. Test a representative multiphase sample before accepting phase fractions.
4. Verify access for each user's proposal from NOVA. This adapter uploads the
   selected bank to Galaxy History, so Analyze consumes History datasets and
   does not require its own `/SNS` mount. Any future direct batch resolver
   needs a separate mount/ACL check; DGX readability alone does not establish it.
5. Repeat the end-to-end selected-bank analysis, dashboard, GPX handoff and
   GSAS-II export checks for each release. Deploy both NOVA and Analyze when
   the input/result contract changes, using successful full-SHA image pins and
   checking both GitLab deployment jobs.

## NDIP-test deployment verification

Both images were built and tested from source commit
`78f2dcedcd5eed1db1c0885a47dc47cac18d63e0`. The Analyze deployment commit was
`2fb50685fa563d739ab49943c55a0dc243f80e19`, followed by the Interactive
deployment commit `74509ac27fed0f4c33e98fb9764bcdb6162f5e28`. Tool sync and
Galaxy rollout succeeded for both stages. The live catalog and a fresh browser
session confirmed the new versions.

The deployed NOVA session discovered two sample IDs across all twelve modern
individual reductions in IPTS-38746, with the expected names and counts.
A Full-mode silicon bank-5 smoke analysis completed and published its fit,
tables, GPX and result archive. Its normalized summary and rendered results
retain the unverified-profile warning and `needs_review` status. The smoke
test used [COD 9008565](https://www.crystallography.net/cod/9008565.html) as its
single reference candidate, built through the existing Build CIF Library tool.
Analyze requires the built portable library output from that tool.

The existing POWGEN regression reproduced the previous release's Rwp, phase
fractions and lattice parameters exactly. These checks establish deployment
and data-handling behavior. They do not qualify the historical NOMAD 2019B
profile for the 2022 silicon acquisition or for current experiments.

The NOMAD GPX opened in the existing hosted GSAS-II desktop. Save project,
Quick CIF export into `Exports`, and Save and Exit completed successfully.
The finalized Galaxy GPX and ZIP were downloaded and checked; the archive
contains the silicon CIF with cell parameters.

During this check, closing GSAS-II's plot window caused a later phase-selection
callback to raise `RuntimeError: wrapped C/C++ object of type AuiNotebook has
been deleted`. Keep that window open or minimize it during editing. Saving
and export still completed; a fresh desktop can reopen the saved project.
The saved-project reopen test confirmed that the phase controls and unit cell
load correctly when the plot window is minimized rather than closed.
The hosted GSAS-II image was not changed by the NOMAD release.

Joint multi-bank refinement, simultaneous multiple-bank watches, automatic
raw-event reduction, PDF analysis and summed-product revision processing are
outside this first adapter.

Facility references: [NOMAD autoreduction documentation](https://powder.ornl.gov/auto_reduce/nomad_auto.html)
and [NOMAD instrument/profile resources](https://neutrons-user-facilities.ornl.gov/instruments/nomad/).
