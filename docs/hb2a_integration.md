# HB-2A Live integration

Status: development preview in NOVA 0.4.23. It has not been deployed on NDIP. Import equivalence and a native
refinement smoke test do not establish quantitative scientific acceptance.

NOVA 0.4.23 adds exact, checksum-verified selection from 52 facility-published
profiles across 11 explicitly dated cycles. Missing dates/configurations still
require a supplied profile. See [profile selection and acceptance](beamline_profile_selection.md)
for current coverage, review requirements and SNAP condition handling.

The combined HB-2A/SNAP review preserved the current POWGEN/NOMAD discovery
and saved-profile recovery paths. Completed HB-2A watches restore their supplied
CW profile once; display controls verify its saved checksum and retain native
2theta if that profile is unavailable. The beamline selector explicitly labels
this support as a preview.

## Supported local path

Choose **Live → HB-2A**, an IPTS, one normalization family, and refresh samples.
The source is `/HFIR/HB2A/IPTS-<number>/shared/autoreduce`, read only. The
adapter accepts individual reduced XYE products named
`HB2A_exp<experiment>_scan<scan>_autoreduced_norm_time.dat` (collection-time
normalization), or `_autoreduced.dat` (monitor normalization). Legacy individual
`.dat` and `.xye` exports are an explicit, separately reviewed family. They are
not assigned a normalization from their intensity magnitude.

Automatically combined scan ranges, mutable `__HB2A_reduced*` products,
detector-resolved exports, GSS duplicates, raw SPICE tables, and fixed-angle
temperature/field scans do not trigger a powder analysis. An IPTS with only raw
files needs supported instrument reduction first; this adapter does not run
Mantid or reconstruct uncertainties from normalized counts.

HB-2A uses neutron constant-wavelength geometry. The reduced histogram is the
instrument-reduced combined detector bank, not detector 1 or 44 independent
physical-bank analyses. It is submitted as one ordinary Analyze histogram.
Live configuration uses `cw`; POWGEN and NOMAD retain `tof` and their existing
bank selection. The Single-pattern draft remains independent.

## Acquisition and sample identity

SPICE scan numbers restart in each experiment, including several experiments
within one IPTS. The immutable identifier is therefore
`HB2A_EXP<experiment>_SCAN<scan>`, retaining separate integer experiment and scan
fields. No synthetic packed number is presented as a facility run number.

Raw metadata is joined only to the exact
`exp<experiment>/Datafiles/HB2A_exp<experiment>_scan<scan>.dat`. Proposal,
experiment and scan headers must agree. The parser checks declared row widths,
finite metadata, point numbers, counting time, a 2theta scan, and a completion
footer. Stopped scans remain visible in sample discovery but require explicit
inclusion after reviewing measured angular coverage. Incomplete files retry.

Where no sample ID is recorded, dropdown entries use the SPICE `samplename`
scoped to its experiment. This is explicitly a **sample label**, not a verified
physical-specimen identifier. Reusing the same label for a different specimen
cannot be resolved automatically; review titles and exclude acquisitions or
use separate watches. Unquoted commas in archived summary CSV titles/names
cannot shift identity fields: CSV columns are not used as sample authority.

The watch snapshots only currently existing IDs at startup. New experiments
can start at scan 1, late reductions can arrive out of order, and subsequent
polls do not require a future list of scans. Initial-selection limits remain
separate from later arrivals. Checkpoints retain this cursor and the full
normalization, exclusion, condition-binding and profile-review contract.
Replaced reductions or raw metadata never silently relabel acknowledged jobs.

## Instrument setup and profile policy

Measured `m1` identifies a documented Ge reflection only near its actual
setting. It is not a refined wavelength. The date-qualified Mantid conventions
are preserved:

| Acquisition date | Measured setting | Interpretation |
| --- | --- | --- |
| All inspected periods | `m1≈9.45`, `m1≈0` | Ge113 (~2.41 Å), Ge115 (~1.54 Å) |
| From 2021-04-25 | `m1≈−4.375` | Ge117 (~1.12 Å) |
| Before 2021-02-23 | `colltrans≈0`, `≈±80` | IN, OUT |
| From 2021-02-23 | `colltrans≈0`, `≈±80` | OUT, IN |

Intermediate positions, changing settings, or missing monochromator evidence
do not select the nearest profile. Some inspected 2026 SPICE files omit
`colltrans`. Their sample labels still populate, but preflight requires an
explicit reviewed IN/OUT choice. A choice cannot override contradictory
measured values. Changes in experiment, Ge, collimation, normalization or
acquisition-channel schema stop reuse of one manually supplied profile.

An explicitly supplied profile must be a modern, single-bank neutron CW `.instprm`
with `Type:PNC`, `Bank:1`, finite wavelength/zero/profile terms, a wavelength
consistent with the measured Ge setting, and positive Gaussian variance over
the measured range. These are wrong-mode/setup sanity checks, **not calibration
acceptance**. The user reviews cycle, collimation, sample environment and
reduction corrections; resulting summaries retain `needs_review` notices.

The packaged registry now matches exact published date intervals, Ge reflections
and collimation against unchanged, checksum-pinned facility profiles. The new
official facility website downloads were accessible on 2026-10-27; the initial
0.4.22 investigation encountered HTTP 403 at the older links. Native imports
and actual-range variance checks passed for all 52 included profiles. This
establishes published calibration identity and software compatibility, not
independent standard acceptance for every sample. No asset is downloaded at
job runtime and no latest profile is extrapolated into older/later cycles. One legacy
archive file also had inconsistent Ge wording versus its wavelength, another
reason not to select by filename alone. Detector-efficiency `vcorr` files and
detector exclusions are reduction inputs, not Rietveld peak-shape profiles.

## Conditions and plots

SPICE records measured point channels alongside requests and inactive sensors,
without universally declaring units or the active sample sensor. Optional
temperature, magnetic-field and pressure selectors require explicit channel,
unit and review choices. Known count/motor channels and setpoints cannot be
misrepresented as sample-condition readbacks. Celsius converts explicitly to
kelvin and pressure to pascals. Exposure-weighted point means, ranges, channel,
recorded units and source hashes are retained. This is point-sample weighting,
not reconstruction of unrecorded continuous environment history.

The original Results and Transition tracking tabs are retained. The measured
curve path supports native 2theta; calibrated d and Q display uses the exact
selected, checksum-verified CW profile and its zero correction:
`d = Lam / (2 sin((2theta − Zero)/2))`, `Q = 2π/d`. Display conversion relabels
and reorders measured samples; it does not resample or feed converted curves
into inference. A CW profile cannot create a fictitious TOF axis. Missing
profiles leave native 2theta available. Region overlays, zoom and ribbon use
the same coordinate mapping.

XYE staging preserves numeric records and supplied uncertainties. Zero/zero
bins retain GSAS-II's zero-weight mask. Negative intensities are currently
blocked because the native XYE importer clips them; they must not disappear
silently. No square-root errors are invented for normalized data. Staging adds
bounded provenance and, when qualified, a temperature header for the GPX.

The statistical detector remains phase independent, but its sensitivity and
false-alert rate have not been calibrated for HB-2A. Magnetic-only peaks,
polarized measurements, diffuse scattering, environment/can scattering and
absorption can leave residuals without implying a missing nuclear phase.
Fixed atomic displacement parameters can bias temperature-dependent phase
fractions. Neither a low Rwp nor a pattern symbol certifies phase identity,
magnetic order or absolute fractions.

## Evidence and remaining acceptance

Read-only investigation inventoried 437 IPTS-named directories and inspected
20 selected proposal layouts spanning 2016–2026. Thirty selected reduced
products and their raw metadata from 15 IPTS were copied to a local mirror.
All 30 sample labels resolve. Thirteen pass strict acquisition/setup checks;
seven were stopped scans and 13 lack measured collimator metadata (categories
overlap). Seventeen original/staged pairs imported identically into the native
GSAS-II runtime in x, intensity and weights, including 770 zero-weight bins.

An independent smoke refinement of the existing repository Mn3GeN example
improved Rwp from 20.38% (scale/background) to 19.38% after cell refinement.
Instrument and atoms stayed fixed, the original GPX was preserved, and Rwp was
independently recomputed. Its remaining residuals are substantial: this is a
workflow check, not an accepted scientific model. The d conversion was also
checked against GSAS-II `Pos2dsp` for all 2,431 example coordinates.

Synthetic tests cover historical collimator reversal, Ge117 date guards,
partial and stopped acquisitions, exact identity, normalization separation,
profile mistakes, conditions/units, bank restrictions, new-only restart,
late arrivals, source revision, CW settings and coordinate conversion.
Real evidence and private patterns stay outside this repository in the local
`audit/2026-10-26/hb2a-integration` directory. Real IPTS-21073 legacy `.xye`
numerical validation was not performed: an additional private-data transfer
was rejected by automatic approval review. Legacy syntax has synthetic tests.

Before NDIP acceptance, obtain reviewed dated profiles/standard measurements,
verify their positions/widths and lattice constants through the exact batch
image, test a representative multiphase/magnetic sample, validate device and
physical-specimen mappings, and calibrate HB-2A transition sensitivity. Confirm
the interactive pod has the authorized read-only `/HFIR/HB2A` mount. Local DGX
readability does not establish NOVA/NDIP mount or proposal ACLs. Then repeat
real browser, monitor-receipt, GPX/GSAS-II handoff and export tests after the
normal immutable-image deployment process.

The 27 October combined review passed 912 NOVA tests and 43 adapter/import
checks. A full offline RADAR-PD run on the repository Mn3GeN example, using its
two candidate CIFs, completed in 14.4 seconds with Rwp 16.69% and retained
`needs_review`. This closes the software pipeline check for that example;
it does not replace the missing quantitative acceptance on the new IPTS cohort.

Primary sources:

- [ORNL HB-2A instrument and cycle calibration table](https://neutrons-user-facilities.ornl.gov/instruments/powder/)
- [ORNL HB-2A reduced-product guide](https://powder.ornl.gov/bragg_diffraction/data_reduction/hb2a.html)
- [Mantid HB2AReduce documentation](https://docs.mantidproject.org/nightly/algorithms/HB2AReduce-v1.html)
- [Inspected Mantid implementation, pinned revision](https://github.com/mantidproject/mantid/blob/6f4e0603c548ede18f513a336298fffe24d48b1c/Framework/PythonInterface/plugins/algorithms/HB2AReduce.py)
