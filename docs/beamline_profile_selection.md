# HB-2A and SNAP Live setup

Both adapters remain previews. Reduced-data discovery, metadata qualification,
instrument-profile identity and scientific acceptance are separate checks.

## HB-2A

Choose the IPTS, reduced-product normalization and experiment-scoped sample
label. SPICE experiment + scan identify an acquisition; scan numbers alone do
not. The raw SPICE file supplies acquisition dates, measured m1/Ge setting and
date-qualified collimator position. A reviewed collimator override is available
only when that measurement is absent; it cannot replace a conflicting value.

The automatic profile option matches the entire acquisition interval against
an explicitly published cycle, Ge reflection and IN/OUT collimation. It verifies
the packaged file checksum, neutron CW type, wavelength and physical Gaussian
variance over the measured range. The published filename/source is shown even
before the experiment review is confirmed. The original profile, registry and
source hashes are retained in staged provenance and saved session records.

The packaged registry covers 52 published configurations across 11 dated cycles
in 2023–2025. It does not cover every intervening day or any 2026 cycle. Missing,
ambiguous or contradictory mappings require a supplied, reviewed `.instprm`.
Different experiments, configurations or profile mappings require separate Live
sessions. No nearest-cycle fallback is used. A restored automatic session uses
the same registry identity and validates the saved profile checksum.

Optional measured SPICE channels require an explicit quantity/channel/unit
review. Detector counts, diffraction motors and setpoints cannot become sample
conditions. A counting-time-weighted summary describes the sampled readbacks;
it is not proof of thermal equilibrium.

## SNAP

Choose one individual GSAS export family, source histogram and matching supplied
TOF profile/Bank label. State, grouping, lite/native mode, calibration,
normalization, mask/export variant and corrections remain separate contracts.
Automatic SNAP profile selection is disabled. An adjacent generated `.instprm`,
detector calibration HDF5 or nominal resolution constant does not establish an
accepted GSAS-II peak-shape calibration. A missing exact reduction record is
reported; a sibling/latest record is never substituted.

The optional **Specimen temperature** section lists named sensors and their
recorded units/validity after checking the experiment. No sensor is selected by
its magnitude or by the run title. Review the specimen-connected sensor and
units before using it. A reviewed unit can fill an absent unit but cannot
override a conflicting recorded unit. Zero Kelvin, inadequate time coverage and
malformed logs are not qualified axes. DAC sample pressure requires a reviewed
pressure mapping; pump/load pressure is not substituted.

The parent NeXus interval is not proof of an exported pattern's interval. Before
using these sensors, review that this export family covers the full acquisition.
Time-filtered exports need an explicit interval/axis mapping. This review, the
binding and compact acquisition context travel with the staged input. Published
GPX copies retain the source and review basis; eligible full-interval temperature
readbacks populate the GSAS-II sample temperature rather than a default value.

Sensor selection is frozen before uploads, included in watch/checkpoint and
submission identities, and retained in the analysis receipt. The binding is
scoped to the IPTS, specimen identity, environment selector and channel/unit
schema. Changing readback values within that scope is supported. A specimen,
environment or schema change requires a new review; discovery/staging checks
prevent a changed source from silently inheriting an earlier binding. An
unidentified specimen cannot acquire a saved sensor binding. Changing sensor,
units, IPTS, sample or export family clears the applicable review.

## Acceptance still required

Published profile identity, a successful native import and a completed fit are
different evidence. Independent standard fits and quantitative sample checks
remain necessary for scientific acceptance. On NDIP, also verify the read-only
HFIR/SNS mounts and proposal ACLs, a newly arriving scan, checkpoint restore and
the published GPX/export. A local test does not prove those deployment details.
