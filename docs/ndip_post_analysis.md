# Selectable post-analysis views

NOVA 0.4.3 adds saved scientific views for single results and live POWGEN/NOMAD series. The Analyze image must also contain the new observation extractor to provide final fit-state and covariance information. Updating the NOVA image alone does not upgrade older results.

## Using the views

Before starting, choose **Post-analysis views** in the setup form. The defaults retain phase evolution and absolute cell trends. Save the reusable configuration to preserve this selection. Live-monitor selections and subsequently pinned references are also stored in the watch checkpoint; **Save view settings** saves changes immediately, including after monitoring stops.

The dashboard focuses on one selected view. Open **Axis, phases and display** to change its metric, axis, representation, visible phases, reference, acquisition branch, uncertainties or connection lines. **Duplicate view** preserves a separate set of choices, for example absolute cell values and fractional expansion. **Remove this view** removes only that instance; removing a recipe from the setup dropdown removes all its instances.

## Expansion and lattice thermometry

In **Thermal expansion / cell change**, choose **Lattice signals (optional)** to
plot explicit phase/direction pairs together, for example MgO a, corundum a and
corundum c. This avoids assuming every cell is cubic, or that equal numerical
cell edges represent independent parameters. Lengths and volumes can share a
normalized scale; their absolute values require separate views.

The representations include absolute values, differences, relative change in
percent, **fractional expansion in per mille** `1000 × (L/L₀ − 1)`, and the
**normalized lattice signal** `L/L₀`. Choose an explicit reference scan or retain
the existing first-eligible-acquisition pin. Points retain acquisition order and
separate compatible banks, samples, profiles and branches. The horizontal axis
can be switched between run number, temperature and other eligible metadata;
Kelvin/Celsius changes display coordinates only.

**Ratio of two lattice signals** uses an explicit numerator and denominator
from the same final model and histogram. A two-phase ratio can cancel a common
multiplicative lattice scale, but does not remove phase-specific strain,
composition or reaction effects. Values are available from older summaries;
ratio error bars require the joint cross-phase covariance. A missing phase is a
gap, not a zero or a silently substituted phase.

Open **Reference comparison / lattice thermometry (optional)** to import a CSV:

```csv
temperature_K,value,unit,source
300,4.0000,angstrom,Reviewed material-specific reference
400,4.0040,angstrom,Reviewed material-specific reference
500,4.0082,angstrom,Reviewed material-specific reference
```

The numbers above are illustrative, not a material calibration. Accepted units
are `angstrom`, `angstrom^3`, and `dimensionless` for a lattice ratio. Supply
3–2000 rows with strictly increasing `temperature_K` (or `temperature_C`),
positive values, one unit and explicit provenance. Curves use shape-preserving
cubic interpolation of log(value), with no extrapolation. The configuration
and export preserve the complete table and source-file SHA-256.

Choose one phase and cell metric, or an explicit two-signal ratio; an explicit
baseline scan is required. Review the curve's material/composition, pressure,
phase range and the assumption that the baseline logged temperature is correct.
No reference is assigned automatically from a formula or phase name. Changing
the phase, ratio, sample, bank, baseline, curve or imported axis mapping clears
the review acknowledgement.

Available comparisons are expansion with a reference overlay, excess expansion
above the reference, inferred sample temperature, and inferred minus logged
temperature. Inference solves `F(T)/F(T₀) = L/L₀`; ambiguous inversions,
unsupported ranges and locally insensitive inversions are withheld. A recorded
or explicitly reviewed mapped temperature supplies the logged condition. The
baseline's zero offset is an anchor identity, not a temperature validation.

Error bars show **conditional standard uncertainties (1σ)**. Normalized values
propagate current and baseline lattice variance; a two-signal ratio uses their
joint covariance, including cross-phase correlation. The baseline normalizes
to exactly zero (or one); its physical uncertainty is not zero. Points sharing
a baseline are correlated, so these bars are not independent observations.
Temperature inference propagates through the curve derivative; optional
baseline temperature standard uncertainty and explicitly supplied logged
temperature uncertainty are included. An axis-mapping CSV can add a
`standard_uncertainty` column in the same units as its value.

Unknown instrument/calibration, reference-model, composition and other
systematic uncertainties are excluded, rather than estimated from Rwp or
invented. Older outputs without joint covariance show ratio values without
invented error bars. **Show recorded temperature span on x** displays the
acquisition minimum–maximum separately; that range is not a standard
uncertainty, and is not borrowed from a different imported mapping or sensor.

Presentation changes recalculate existing observations. They do not submit new refinements. Clicking a live-series point selects its scan in the existing scan inspector. Plotly supports zoom, pan and figure download; numeric x zoom is linked between views with the same axis units and meanings. Visible ranges are retained for replay/export. Resetting the Plotly axes restores automatic ranges.

## Available calculations

| Selected family | Result and required inputs |
| --- | --- |
| Phase evolution | Absolute modeled-crystalline wt%, percentage-point difference, or explicit relative percent. Hiding phases never renormalizes the remaining values. |
| Thermal expansion / cell change | a, b, c, angles or volume, as absolute values, differences or relative changes. Optional branch-specific expansion fits require at least three distinct measured temperatures, an eligible refined length/volume, an acceptable within-scan temperature range and no unresolved covariance disagreement. Fits use raw cell values, not normalized points that share a reference. |
| Lattice shape / distortion | c/a, b/a or (a−b)/(a+b), with full relevant cell covariance when available. A ratio fixed by symmetry is a model constraint, not an independently measured distortion. |
| Reference lattice strain | Relative a, b, c, volume or d(hkl), optionally in microstrain units. Enter three nonzero-as-a-vector Miller indices for d(hkl). This is lattice change relative to a reference; stress requires another physical model. |
| Transition / hysteresis | Acquisition-path overlays of a selected observable. An optional threshold gives bracketing scan/axis intervals, without asserting a thermodynamic transition. Increasing/decreasing paths stay separate. Explicit metadata branch labels are retained when supplied. |
| Reaction progress | A defined observable normalized using fixed user-entered start/end values and a timezone-qualified time zero. Overshoot remains visible. This is a progress proxy; chemical conversion, rate-law fitting and stoichiometry are not inferred automatically. |
| Operando correlation | A structural observable versus recorded voltage, capacity, pressure, field or time, or a reviewed per-run mapping. Raw electrochemistry-log clock alignment and acquisition-window integration are not implemented here; imported values must already represent the appropriate scan window. |
| Sample / composition / preparation | Absolute structural metrics or phase fractions versus a reviewed sample-ID mapping. Nominal composition and preparation history remain labeled separately from measured conditions. |
| Coherent-domain size / microstrain | Discoverable with a prerequisite explanation. The additional instrument-qualified size/strain refinement recipe is not implemented; stored GSAS-II defaults are not plotted as measurements. |
| Atomic geometry / site order | Discoverable with a prerequisite explanation. Validated atom/site definitions and coordinate/occupancy refinement recipes are not implemented; fixed CIF quantities do not become measured site changes. |

The last two families cannot be enabled merely by acknowledging a warning. They require additional scientific implementation and validation. Reaction-rate estimation, kinetic model fitting, automatic transition detection, raw external-log ingestion and atomic/size refinement remain separate extensions.

## Reviewed axis mappings

Acquisition metadata supplies axes when available. Unknown temperature units, nonpositive temperature and conflicting metadata do not become valid temperatures by interpreting a run title. Celsius values with explicit units are converted to kelvin. Measured values and preparation/nominal values cannot be silently mixed in one series.

For composition, treatment categories, or an already aligned external channel, import a UTF-8 CSV under **Scientific definitions and additional inputs**:

```csv
axis,run_id,sample_id,value,unit,source,kind,verified
sample_state,,sample-A,0.25,nominal fraction,reviewed preparation sheet,nominal_composition,true
voltage,NOM_123_BANK5,,4.10,V,reviewed acquisition-window mean,measured,true
```

These are illustrative identifiers. Use exact scan IDs from the dashboard. `sample_state` keys use the sample ID; other mapped axes use the run ID. Allowed kinds are `measured`, `nominal_composition`, `preparation_history` and `user_supplied`. Review each value and set `verified=true`. Conflicting duplicate mappings reject the complete import. **Clear imported mappings** allows explicit replacement; metadata remains the fallback when a mapping is cleared.

Optional `branch` and `cycle` columns retain reviewed process labels, for example `charge` and `3`. They keep different cycles separate even when their voltage ranges overlap. Explicit acquisition metadata can also supply branch, cycle and process-stage labels. No physical cycle number is inferred solely from a voltage reversal.

## References, missing values and grouping

Automatic references use acquisition order, not job-completion order. Earlier pending scans prevent premature baseline selection. Once pinned, a reference does not change because an earlier result arrives, the plot is zoomed, or another phase is hidden. An unavailable or excluded reference leaves relative values unavailable until an explicit replacement is selected. **Choose new automatic reference** deliberately clears the active view's pins and selects again from eligible acquisitions.

Series are separated by instrument/IPTS, sample, phase model identity and basis, bank, initial instrument profile, reduction compatibility and recipe. Unknown sample IDs are isolated by run. Model identity uses the original phase name (which can include the candidate/CIF identity), symmetry setting and atom labels/types; it does not hash fitted cells/coordinates or merge solely by chemical formula. Reviewed equivalence across different models/settings is not automated.

A phase absent from one accepted model creates a gap. It is not automatically zero or proven absent. Repeated temperatures and acquisition order are preserved, and heating/cooling paths are not globally sorted into a single line. Identical retries deduplicate; conflicting active GPX revisions are withheld until one revision is selected.

An unknown/fixed parameter can be included only through the explicitly exploratory model-values option. It cannot qualify an expansion coefficient. Older results without fit-state metadata do not silently become independent cell measurements; use a current Analyze result or inspect those older model values explicitly. A read-only Galaxy job to upgrade historical GPX extraction has not been added.

## Online pattern indicators and boundary rules

The Transition tracking tab consumes each completed reduction as it arrives,
including before refinement finishes. No final list or future scan is needed.
The user selects indicators and a required count: one indicator can be sufficient
even if other indicators do not trigger. New/lost signal, resolved splitting,
relative peak width, shape and relative intensity remain separate measurements.
Unavailable selected indicators are not counted as negative evidence.

The current-group reference is fixed until a measured change meeting the selected
rule starts or returns to a group. This detects accumulated changes. Selecting the preceding
scan instead explicitly requests adjacent comparisons. Symbol reuse compares
with earlier group references; it does not infer a chemical phase identity.
Horizontal heatmap lines follow that rule and cached evidence. The indicator
history plots the triggered count with the selected required-count threshold;
it does not present pattern labels as a numerical score. Missing indicator
measurements show an availability bound. Filled condition intervals are omitted
because they can be mistaken for identified phase regions.

The heatmap and measured-pattern ribbon share a calibrated TOF/d/Q horizontal
axis and zoom. A vertical scan cursor updates the ribbon and scan inspector.
Rainbow (Turbo) is the default palette. Cursor/axis/palette/zoom changes only
affect display copies and never rerun refinement or change native inference.
Exact, checksum-verified physical/profile-bank mappings are required for TOF
conversion, including the nonlinear and zero terms. This follows GSAS-II's
[coordinate equation](https://github.com/AdvancedPhotonSource/GSAS-II/blob/main/GSASII/GSASIIlattice.py)
and [legacy ICONS interpretation](https://github.com/AdvancedPhotonSource/GSAS-II/blob/main/GSASII/GSASIIfiles.py).
Missing profiles leave calibrated axes unavailable instead of using nominal
export DIFC. Intensity retains its original measurement units.

Width changes assume fixed instrumental/sample broadening within the compatible
bank and setup. Native-grid local profile fits give conditional FWHM/position
measurements. Default width evidence requires 3% change, an error ratio of 3 and
two supporting isolated regions; thresholds are editable. Unresolved broadening
can trigger without two resolved daughter peaks. Crowding, failed fits or
insufficient sampling can withhold width evidence. Disabling the fixed-broadening
assumption makes the width indicator unavailable, not unchanged.

The selected indicators, comparison baseline, sensitivity, assumption and policy
hash are saved in the watch checkpoint and evidence JSON. User edits reclassify
received data immediately without repeating RADAR-PD/GSAS-II. Ordinary arrivals
append; late/corrected reductions cause an explicit replay in acquisition order.
The standalone API supports `append(scan)` and `reclassify(policy)` with no
completed-series input. Comparison and preparation caches are bounded.

Final Rwp is available as a separate, opt-in model-dependent indicator. Only an
explicit completed Full-analysis final-project Rwp is used. Matched project
receipts and equivalent configuration/library/recipe/profile/fit domains are
required; contributing fitted-point overlap must be at least 98%. The rule uses
both an absolute change in percentage points and a relative change. Pending or
incompatible fits are unavailable, not negative evidence. A late final result
causes received-data reclassification only if Rwp is selected. Rwp is global and
does not create a localized region.

The optional changed-region overlay uses localizable measured-peak evidence
from an explicitly selected previous, boundary-reference or first-scan pair.
Boxes span adjacent compared rows, or mark only the two endpoints of a
nonadjacent comparison. A local indicator filter, overlap merging, common
masked-gap exclusion and a 16-region display cap keep the view readable.
Native local fits also measure evolution of an existing doublet's separation:
asymmetric single and two-component descriptions compete without phase/hkl
knowledge. Recognized doublets below the change threshold optionally have dotted
boxes, while solid boxes passed a local rule. Their separation measurements and
conditional error ratios are inspectable. Neither box proves a phase transition.
The reference and shown/total count are displayed; all evidence remains in the
report. Overlay switches never change the boundary rule or raw intensity data.
The acquisition cursor previews cached native ribbons and pair highlights in
the browser, committing full scan details on release. Visible row indices stay
stable during dragging while newly received scans continue through analysis.

## Scientific observations and uncertainty

Analyze publishes `Scientific_observables.json` beside the lattice CSV and includes the same packet in the normalized result summary. It contains source GPX SHA-256, phase/basis identity, profile/recipe context, per-metric final refinement state and available cell covariance. NOVA consumes that packet; it does not unpickle arbitrary GPX uploads.

Final GSAS-II vary-list variables and symmetry determine cell states. A stored refine flag or the mere existence of a parameter is insufficient. General constraints outside the supported direct reciprocal-cell mapping remain unverified. This conservatively withholds some quantities rather than inventing an uncertainty.

Validation found disagreement between the GSAS-II public cell-ESD API and independent symmetry-Jacobian propagation for some non-cubic phases. Existing scalar API ESDs are retained, but the conflicting full covariance block and expansion coefficients are withheld for review. There is no dataset-specific profile override or suppression of the warning. Ratios/hkl do not receive fabricated independent errors when their covariance is unavailable.

Relative-point uncertainties assume independent scan fits and remain conditional on the model and calibration. A shared reference correlates normalized observations. Self-reference is exactly zero algebraically; that does not establish zero physical uncertainty. Expansion fits use raw cells with weighted least squares when complete positive scalar variances are available, otherwise residual-based OLS. Neither method includes temperature uncertainty or calibration systematics. NOMAD profile/scientific qualification remains an independent requirement.

## Saving and exporting

**Download views and observations** creates an offline ZIP containing the versioned settings/results, raw observations, per-view CSV and standalone Plotly HTML for available views. It includes references, phase colors, exclusions, recipe versions and visible plot ranges. Hidden points and excluded points retain their separate meanings. The original scientific refinement configuration is not rewritten when dashboard presentation settings change.

Configuration contracts and watch checkpoints are backward compatible. Unsupported recipe/schema versions and unknown settings are rejected rather than silently discarded. The source NOVA version is 0.4.3; its source XML image pin must be updated only after a successful release image exists. No deployment is implied by this implementation.

## Validation

Run the NOVA suite and focused batch contracts:

```bash
python -m pytest nova/tests \
  tests/test_ndip_adapter.py tests/test_ndip_catalog_bundle.py \
  tests/test_ndip_lattice.py tests/test_ndip_observables.py \
  tests/test_ndip_input_provenance.py tests/test_gsasii_gui_contract.py -q
```

Tests cover normalization and covariance, stable references and asynchronous recovery, fixed/stale/symmetry-constrained values, missing phases, mixed units and metadata meanings, distinct samples/banks/calibrations/recipes, branch preservation, ramp-qualified fitting, endpoint definitions, conflicting revisions, mappings, offline exports and presentation callbacks that must not launch jobs.

Separate local validation replays trusted previously produced POWGEN/NOMAD GPX files in native GSAS-II and a cached Linux batch runtime; checks the actual NOVA browser UI using synthetic data; and exercises a synthetic 1,000-scan, eight-phase series. Those checks establish software behavior and extraction compatibility, not independent scientific accuracy of those experiments. Private facility/proposal evidence stays outside the repositories.
