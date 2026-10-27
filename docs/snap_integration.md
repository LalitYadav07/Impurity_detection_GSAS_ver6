# SNAP Live integration (local development)

NOVA 0.4.22 adds SNAP as an explicitly labelled development preview in the
existing Live interface and shared watch lifecycle.
It is not an NDIP release, and it does not certify automatic GSAS-II profile
selection. Real SNAP calibration-standard checks show why profile acceptance
must remain a separate scientific decision.

The combined review preserves current POWGEN/NOMAD support and HB-2A's distinct
CW/HFIR path. Restored completed SNAP watches load the original supplied
profile once, verify its original Bank label and hash, and distinguish that
receipt from the staged Bank-1 copy. Missing profiles leave completed results
readable and report the unavailable coordinate conversion. The combined NOVA
suite passed 912 tests; 43 adapter/import checks and a native GSAS-II signed-bin
comparison also passed. The real SNAP pipeline smoke reproduced Rwp 17.0844%
and retained `needs_review` in the frozen runtime. These are software checks,
not independent calibration acceptance.

## Select the actual reduction

Choose **Live → SNAP**, enter the IPTS, and refresh sample IDs. Choose an explicit
GSAS export folder and, where more than one exists, a **grouping / export
variant**. Refresh the samples for that choice. Candidate structures and all
scientific settings are prepared within Live, using the established library
builder and configuration editors.

Supported folder families are:

- `shared/SNAPRed/export/gsa/<group>`;
- `shared/autoreduce/export/gsa/<group>`;
- legacy `shared/autoreduce/gsas`, `shared/autoreduce/GSAS`,
  `shared/data/gsas`, `shared/gsas` and immediate `shared` products.

Folders are enumerated explicitly and never recursively merged. Canonical
individual `SNAP<run>_<group>[_variant].gsa` and historical `SNAP_<run>...gsa`
products are recognized. A bounded historical prefix such as `SimpleMask_` is
retained as a separate variant. Different masks, file tags and timestamped
re-reductions do not silently replace one another. Multiple candidates for the
same run within a selection fail rather than choosing the newest file.

Only complete individual SLOG/FXYE exports trigger analysis. Raw event files,
processed NeXus workspaces, XYE/CSV exports and arbitrary summed products do
not trigger a fit. A raw-only experiment needs an instrument-supported
reduction first. The adapter never runs a facility reduction script.

The standard grouping names are `all` (one focused histogram), `bank` (two)
and `column` (six). Named angular groupings such as `1deg`, `2deg` and `5deg`
and the documented `2_4` group are recognized when explicitly selected. Exported histogram labels are not
physical detector IDs. Select one source histogram per reviewed profile
mapping; independent watches can analyze other histograms. No averaging or
joint refinement is added.

## Profiles and calibration provenance

Supply a GSAS-II `.instprm`, enter its actual **Bank label**, and confirm the
state/group/histogram mapping. The source label, profile label and staged local
label are independent. A selected source histogram and profile block are staged
as local Bank 1 without changing numeric records. The complete source/profile
hashes, state ID, lite/native mode, exact reduction-record hash, calibration and
normalization versions, correction annotations and original bank mapping are
retained. Display TOF↔d↔Q uses that same checksum-checked profile and all four
coefficients (`difC`, `difA`, `difB`, `Zero`); nominal export DIFC is not a
replacement calibration.

SNAPWrap creates approximate profiles by default. Its documented exporter uses
nominal geometry and common starting peak-shape constants. Merely finding a
same-name `.instprm` is not scientific acceptance. Geometry agreement checks
are wrong-bank sanity checks, not an accuracy certificate. A nonfinite profile
or lookup table is rejected instead of being silently repaired.

The archive has a useful authoritative reduction hierarchy:

- `shared/Calibration/SNAPInstPrm/InstrumentParameterIndex.json` and versioned
  `SNAPInstPrm.json` define instrument configuration/state-ID schemas and run
  validity intervals. Despite the name, they are not GSAS-II peak-shape files.
- `shared/Calibration/Powder/<state>/<lite|native>/diffraction/` holds
  calibration indexes, detector-coefficient HDF5 files and exact records.
- The corresponding `normalization/` hierarchy records vanadium corrections.
- An individual export's `redRecord` points to its original reduction record,
  including the calibration and normalization actually used.

Current mutable indexes must not overwrite the historical product receipt.
Some real exports point to missing `/None/ReductionRecord.json` paths although
several timestamped reductions exist nearby. The adapter preserves the named
state and reports missing provenance; it never substitutes a latest/nearest
record. Missing provenance requires an explicit exploratory acknowledgement.
Changing state, grouping, lite/native mode, calibration/normalization version,
correction annotation or export variant requires a separately checked mapping.

A curated profile registry would be useful only with exact state, grouping,
histogram, reduction convention and version/run validity, source hashes and
independent standard acceptance. Downloading a generic file by beamline name
or deriving peak shapes from nominal geometry is insufficient. No unqualified
automatic profile entries are packaged by this change.

## Identities, logs and conditions

Sample identifiers are independently checked in bounded NeXus fields and
run-info XML where available. Public ONCat metadata can supply a bulk browsing
preview; it cannot authorize a calibration or replace selected-run source
verification. Names and titles from exact-run CSV rows remain useful display
evidence, and conflicts stay visible.

Unsigned unset IDs `4294967295` and `18446744073709551615` are withheld, with
their original observations retained. Real SNAP standards used these values
alongside a name of “No sample”; a silicon title does not create an acquisition
sample ID. All samples can still analyze such runs, but transition linking
requires a verified specimen identity. Recorded sample names can also be stale:
a real run titled as a Pb loading still retained another material's sample
name. Candidate chemistry must not be inferred blindly from either field.

SNAP facility logs include multiple active and inactive temperature channels.
The adapter exposes them as candidates requiring a reviewed specimen-device
binding, with original units and acquisition-window summaries. It never chooses
the largest/nonzero sensor. Pump/load pressures in bar are not automatically
the sample pressure in a DAC, and pressure-calibrant equation-of-state constants
are not a pressure measurement. Use the existing reviewed axis-mapping CSV for
verified sample pressure or other external coordinates. No pressure is guessed
from titles, filenames or a nominal load.

## Missing bins and signed observations

Modern SNAP exports commonly contain `nan` intensity with zero uncertainty.
Those are absent bins, not evidence of absent diffraction peaks. Their numeric
records and full coordinate grid are preserved; the standard GSAS-II import
gives them zero fit weights. Positive-weight nonfinite observations and
negative/nonfinite uncertainties are rejected. Truncated or changing files
remain pending rather than being fit as complete patterns.

For provenance-marked staged SNAP inputs only, the batch importer restores
finite signed observations with positive uncertainties before the normal
GSAS-II TOF/bin-width conversion. GSAS-II's standard FXYE reader otherwise masks
nonpositive intensities. Missing bins retain zero weights. Unmarked input and
existing POWGEN/NOMAD imports retain their established behavior.

The raw transition reader matches SNAPWrap's documented bin-integrated export
and GSAS-II's bin-centering/density conversion, retaining signed observations
and masks. Different state/calibration/normalization/correction scopes cannot
share a comparison group. Coordinate conversion reorders samples and masks
together; it does not interpolate missing bins or change detector evidence.

## Validation and limits

The local survey inspected 19 selected readable historical/recent IPTS folders,
not every SNAP proposal. Seventeen contained selected GSAS products; 19 small
products, matching small profiles/records and 15 distinct scalar/log-only
NeXus subsets were staged (8.69 MB of distinct files; about 9.1 MB transferred
including two repeated metadata reads). No event arrays or complete IPTS were copied.

All 19 products parsed, including historical multibank files, prefix variants,
modern auto/manual export folders and masked bins. Nineteen source histograms
across 11 proposals with supplied adjacent profiles were checked through
GSAS-II: 47,190 points, 533 zero-weight missing bins, exact raw-bank coordinate
and positive-observation/weight agreement. Six historical cases had no matching
GSAS-II profile and remain user choices. These checks validate import/staging,
not the scientific acceptance of the approximate profiles.

The current upstream high-level `add_powder_histogram(databank=...)` can reset
the selection during initialization. A real non-first-bank control exposed
bank-1 data with a bank-2 profile. Independent validation therefore sets the
native reader selection after initialization and checks the selected raw-bank
grid. One-bank staging avoids this ambiguity without modifying GSAS-II.

Separate basic Si diagnostics on runs 63335, 64413 and 73470 completed with
Rwp 25.34%, 16.35% and 17.09%; fixed approximate profiles gave a ≈5.418–5.420 Å
for a supplied 5.4307 Å reference. Those fits are not accepted absolute lattice
calibrations. A real offline Linux Full RADAR-PD/one-pass smoke on run 73470
also recovered Si, Rwp 17.0844%, a=5.417908 Å, and published a 4,113-point fitted
observation plus GPX, plots and durable `needs_review`. Its scientific pipeline
timer was 9.022 s, separate from image startup, library build and UI timing.

Tests exercise source confinement, partial→complete arrivals, later scans,
variant ambiguity, exact-record/mode/state conflicts, sample sentinels,
role/sample filtering before backfill, wrong profile-bank mapping, immutable
staging, signed values/masks, checkpoint recovery, changed future states,
reprocessed-input guards and Live configuration freezing. A Chrome form review
selected a real NIST silicon sample and reached the reviewed exploratory
preflight, with missing-record and condition-mapping notices still visible.
No browser-launched job or narrow-screen acceptance is claimed. Real evidence
and private facility products remain in the workspace audit, outside git.

Before NDIP deployment, merge the shared deployment fixes, reconcile HB-2A's
generic adapter/facility extensions, test the combined UI in a browser, verify
the NOVA pod's `/SNS/SNAP` mount/ACLs, and obtain a matching accepted profile
for a calibration standard and a representative multiphase sample. Publish
successful immutable NOVA and batch images and then update Galaxy pins using
the existing runbook. None of those release/deployment steps is performed here.

## Primary technical sources

- [ORNL SNAP reduction and grouping](https://powder.ornl.gov/bragg_diffraction/data_reduction/snap.html)
- [ORNL exportData profile caveat and folder layouts](https://powder.ornl.gov/bragg_diffraction/data_reduction/snap/utils/exportData.html)
- [SNAPWrap export implementation](https://github.com/neutrons/SNAPWrap/blob/next/src/snapwrap/io.py)
- [SNAPRed calibration/state concepts](https://snapred.readthedocs.io/en/latest/glossary.html)
- [GSAS-II source](https://github.com/AdvancedPhotonSource/GSAS-II)
