# POWGEN run and bank profile resolution

NOVA resolves each selected reduced run independently. It does
not select a profile for an entire IPTS from its latest run. An IPTS can span
calibration cycles, wavelengths, frequencies, detector groupings and guide modes.

`powgen_setup.py` reads the reduced GSA header and literal scientific arguments
from its adjacent Mantid reduction script. Scripts are parsed with Python AST;
they are never imported or executed. Supported historical keyword spellings
include `CalibrationFile`, `CalFileName` and `CalFilename`. A second reduction
that explicitly does not write GSAS output is excluded from GSA provenance.

The packaged catalog records the source, normalized-text SHA256, calibration
cycle, HR/HI mode, frequency, wavelength, source bank, profile bank and grouping.
Public profiles also require agreement with the published run interval. The
open-ended latest interval requires explicit matching reduction provenance.
No rule chooses the nearest wavelength or calibration year. Decimal tolerances
cover reporting precision only. A guide position can contradict recorded mode;
it does not establish mode on its own.

Sources:

- [Official POWGEN profile downloads and run intervals](https://neutrons-user-facilities.ornl.gov/instruments/powgen/)
- Explicit small parameter files under `/SNS/PG3/shared/CALIBRATION/`, recorded
  individually in `nova/src/radar_pd_nova/data/powgen_calibrations/catalog.json`.
- [Mantid SaveGSS format and intensity conventions](https://docs.mantidproject.org/nightly/algorithms/SaveGSS-v1.html)
- [GSAS-II instrument-file reader documentation](https://gsas-ii.readthedocs.io/en/latest/GSASIIutil.html)

## Separate analyses per bank

Each selected bank becomes an independent ordinary Analyze job. Multiple banks
of a run have identifiers such as `PG3_37875_BANK3` and `PG3_37875_BANK4`.
The initial scan limit applies before bank expansion, so the latest five runs
may produce more than five analyses. The existing active-job limit still applies.
Plots keep banks in separate traces/rows; bank results are not averaged.

The existing batch tool consumes one histogram. NOVA stages a single selected
FXYE bank and a corresponding single-bank instrument file in session temporary
storage, remapping only the local bank labels to 1. TOF, intensity and uncertainty
records are retained verbatim. Legacy GSAS parameter records, including PAB
lookup tables, are preserved rather than converted into a scalar-only profile.
The original source bank, reduction evidence, source/staged hashes and selected
profile checksum are retained in the Galaxy watch checkpoint. The IPTS is read-only.

GSAS-II local Bank labels are distinct from GSA source bank IDs. For example,
a wavelength-specific profile can have internal `Bank:1` while correctly
calibrating a GSA BANK 2. The catalog records this mapping explicitly.

Source files must be unchanged between inspection and staging. Duplicate banks,
truncated records, invalid uncertainties and nonmonotonic TOF fail before upload.
Selected banks absent from a particular run are skipped; preflight reports an
error if none of the selected banks occurs in the selected initial runs.

## Preflight and recovery

Preflight checks every selected initial run/bank after sample and scan-range
selection. New-only mode checks the latest matching setup as a preview; every
arriving run is inspected again before submission. A manually selected wavelength
must agree with every selected file. Automatic mode retains per-run detection.

One manual profile cannot be applied across mixed cycles, modes, wavelengths,
frequencies or banks. Multiple-bank files require an explicit validated mapping.
A later setup change stops reuse of a manual fallback.

Watch checkpoints include the resolver/catalog revision, bank selection and
source directory. Old checkpoints cannot silently restore jobs created with a
different profile-selection contract. A new watch may therefore require fresh
validated analyses rather than resume a pre-resolver checkpoint.

Historical named cycle folders (for example `2017-B` or `2019_B`) are discovered
as explicit folder choices. Only the selected folder is listed. Current and
archived reductions are not merged: reprocessed copies of the same run can have
different calibration provenance and must not be silently substituted.

## Limits and evidence still required

The current catalog is a conservative prototype, not a claim of universal
historical coverage. Missing provenance, conflicting guide/mode evidence,
custom or partial-detector groupings, ambiguous profile variants, missing
profiles and unpublished frequency/wavelength settings need a verified mapping.

## Run review and scientific outputs

Sample IDs are acquisition identifiers, not proof of sample presence. The run
review selector shows titles alongside run numbers. Explicit background/empty
and ramp/cooling title hints are excluded by default; users can include either
category or exclude additional individual runs. Unclassified titles remain
included, so users should review the selection. This policy applies to later
arriving runs too and is part of the saved watch contract.

Bootstrap screening now converts percent-valued lattice tolerances into fractions
exactly once: 0.6 percent becomes 0.006, not 0.6. Screening cell trials restore
their scale-only checkpoint when the correlation worsens, GSAS-II fails, or the
cell changes by more than 2 percent in length or 2 degrees in angle. Main-phase
recovery restores the checkpoint before a retry or fallback. Polish trials also
check cell movement relative to the accepted checkpoint.

Full TOF analyses finish with a transactional joint-cell refinement of the phases
selected by discovery. `full.final_joint_cell_cycles` controls its budget (default
12; zero disables it). Instrument coefficients and atomic coordinates are held.
This step does not prescribe phase identities or target fractions. Failed trials
retain the last accepted model. It is distinct from joint refinement of multiple
detector banks, which remains unsupported.

The normalized summary includes the final GPX's cell lengths, angles, volume,
phase ID, source GPX and covariance-propagated standard uncertainties. Missing
uncertainties are null, never fabricated zeroes. A lattice CSV, per-scan table
and phase/bank/sample-separated plot are available in NOVA. The experiment CSV
also records sample, conditions and final Rwp. Vanishing phases and invalid cells
raise `needs_review` even when Galaxy finishes normally. `checks_passed` describes
these numerical checks; it is not a certification of phase or fraction accuracy.

Fitting range and background remain explicit reusable-configuration choices.
NOVA does not silently replace them with settings tuned to a particular sample.
A detector-calibration HDF5 file cannot substitute for GSAS peak-shape parameters.

An inventory/metadata match is not a successful scientific analysis. In particular,
profile selection does not validate phase discovery, cell refinement, phase
fractions, sample-environment background, temperature units or experiment role.
A shared sample ID can also label an empty-furnace or normalization measurement.
Those roles must not be inferred from the sample ID alone.

The implementation has not been deployed. Before release, resolve the required
legacy mappings and the independent scientific model-acceptance regression,
then perform a fresh NDIP end-to-end smoke test.
