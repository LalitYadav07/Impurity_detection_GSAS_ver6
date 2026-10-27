# RADAR-PD NOVA

This package is the interactive NDIP/NOVA client for RADAR-PD. It does not run
the scientific pipeline in the interactive container. It uploads or resolves
inputs, submits the existing `neutrons_radar_pd_analyze_prototype` Galaxy tool,
recovers jobs from the user's Galaxy history, and renders completed artifacts.

The Full and Rapid paths share one setup form and one Galaxy submission
contract. Scientific work remains inside the versioned RADAR-PD batch image.

## Facility experiment integration

The Data Collection panel provides a single working-folder browser for mounted
SNS and HFIR experiment data. The user chooses the facility, instrument, and
IPTS, then navigates one directory level at a time. Directory listings are lazy:
RADAR-PD never scans an entire IPTS tree or recursively enumerates raw files.
After a working folder is selected, each input control shows only compatible
files from that folder.

Diffraction data, instrument profile, and optional main-phase CIF retain
independent sources. Each may come from the selected working folder, a laptop
upload, or Galaxy History. This supports common mixed cases such as using a
reduced pattern from `/SNS`, an instrument profile generated in the UI, and a
CIF uploaded from a laptop. For suitable X-ray data, the built-in Cu K-alpha
profile remains available.

Galaxy History remains the authoritative result store. The user may also ask
NDIP's authenticated Export Datasets job to copy the complete results archive
into the selected existing working folder. The archive receives a unique
run-and-job filename because the NDIP exporter copies one file and does not
create destination directories. Publication is confined to the selected IPTS.
A publication failure is reported separately and does not turn a successful
scientific job into a failed job.

Galaxy's authenticated remote-file sources remain a fallback when `/SNS` or
`/HFIR` is not mounted in the interactive pod. Import the desired files through
**Upload -> Choose remote files**, then select the resulting datasets from
**Galaxy History** in RADAR-PD.

### Live experiment monitor

Select **Live** and choose a beamline. POWGEN retains its cycle-profile and
bank-selection workflow. The NOMAD development adapter supports individual
Bragg reductions with an explicit source bank/focused histogram and user-supplied
profile; it remains exploratory pending calibration-standard acceptance.
See [NOMAD integration](../docs/nomad_integration.md) for supported layouts,
bank mapping, correction notices and release requirements.

Live setup is self-contained; preparing inputs in Single pattern is optional.
Its four sections are **Experiment**, **Candidate structures**, **Analysis
settings**, and **Review and start**. Select the built-in catalog, build a
mini/augmented library from CIFs or CIF ZIPs, or choose/upload a previously
built portable library. Builds show usable and rejected sources and select
their result only in the workflow that requested it. Source CIF ZIPs are
distinguished from finished portable library ZIPs.

Live has its own scientific draft, independent of Single pattern. CIF chemistry
provides an editable element suggestion; missing chemistry is reported, and
existing user entries are never overwritten automatically. Neutron TOF is set
for Live. Full/Rapid budgets, fit regions, background and expert controls use
the same editors and validation as Single pattern. A saved History configuration
can be reused, copied for editing, or imported from YAML/JSON.

**Check experiment and inputs** validates the draft and read-only source without
creating jobs or uploading a configuration. **Test latest scan** submits one
analysis for the latest matching scan and selected test bank, leaving continuous
monitoring stopped. **Start monitoring** saves the exact validated configuration
automatically; uploads become durable Galaxy inputs. An unchanged configuration
is reused within the session, including between a trial and monitoring. Changes
during preparation abort the launch, and running jobs retain their original
configuration/library IDs. One setup applies to all selected samples; use a
separate setup when their chemistry or refinement requirements differ.

This removes the need to create a configuration file before using Live. It does
not remove instrument/profile review requirements, including NOMAD physical-bank
versus profile-bank confirmation. Starting a fresh NOVA session with a new draft
creates a new configuration; select the earlier saved configuration to resume
its matching watch checkpoint.

#### POWGEN

The monitor handles large `shared/autoreduce` folders by considering only
canonical `PG3_<run>.gsa` files, without a 5,000-entry cutoff or recursive
searches. After entering an IPTS, leave the field or click **Refresh sample IDs**
to populate sample IDs, reported names, and completed-scan counts. Public ONCat
metadata and validated scan tables provide a bulk browsing index, cached for
five minutes. Catalog previews do not enter the verified identity cache. Runs
missing from the catalog fall back to bounded NeXus/run-info reads; detector
events are never loaded. Catalog access sends no user credential.

The selected runs are independently checked against their source files before
analysis. A finite initial scan limit stops after that many eligible scans;
unselected samples and older scans do not require deep metadata inspection.

Choose **All samples** or select one or more **Sample IDs and names**. The initial
scan limit (including **All existing scans**) applies after this selection;
subsequent matching scans are monitored with the same selection. Scans without
a readable sample ID are reported separately and remain available with
**All samples**. Stop monitoring before changing the sample selection. It is
saved in the Galaxy watch checkpoint so a restart cannot restore a different
sample selection. Completed identity reads are cached until their NeXus file
changes; missing metadata is retried when the sample list is refreshed.
Independent sources must agree on an ID. The expandable identity check shows
sources, unresolved runs and conflicts; malformed rows, placeholder IDs and
foreign-proposal log entries are not guessed into a sample. Table fields do
not authorize a profile. See the metadata resolver section in the NOMAD
integration document for the 0.4.2 development behavior and validation limits.

The **Live** workflow is a session-scoped orchestration layer
around the existing RADAR-PD Analyze tool. It does not duplicate or modify the
scientific pipeline. The user selects an IPTS, the experiment wavelength,
Full/Rapid settings, and an optional main-phase CIF. While the
NOVA session remains open, the monitor:

1. lists only `/SNS/PG3/<IPTS>/shared/autoreduce` without recursion;
2. waits for the canonical `PG3_<run>.gsa` reduction;
3. resolves an exact, checksum-verified packaged POWGEN `.instprm` profile, or
   uses an explicitly supplied cycle profile saved in Galaxy when the cycle is
   not packaged;
4. submits the unchanged `neutrons_radar_pd_analyze_prototype` tool; and
5. displays discovered, submitted, completed, and failed scans as they change.

The default initial selection is the latest five matching scans. Users can
choose a different initial limit, all existing scans, or new scans only.
Later polls submit newly appearing matching runs. Watch
checkpoints are stored as Galaxy datasets and restored after a NOVA restart, so
Galaxy-acknowledged jobs are not submitted twice. The IPTS is always read-only;
results and provenance remain in Galaxy History.

This panel is appropriate for a supervised beamline session. A permanent,
unattended trigger must use **NDIP Ingress -> saved Galaxy workflow -> RADAR-PD
Analyze**. That production trigger can reuse the same profile registry and
Analyze contract without depending on a browser or NOVA pod lifetime.

The setup rail can save a portable configuration plus a `radar-pd-watch/v1`
recipe for a continuously arriving reduced-data folder. The recipe is consumed
by `scripts/ndip_ipts_watch.py`, which must run as an NDIP-managed worker; it is
not tied to the lifetime of the NOVA browser session.

## Selectable post-analysis

Select scientific views before running and adjust their axes, phases, references and representation in the dashboard. Configurations and live checkpoints preserve these choices; presentation changes do not launch refinements. The matching Analyze extractor supplies final fit-state and covariance information. See [post-analysis usage and scientific limits](../docs/ndip_post_analysis.md) for the implemented calculations, reviewed axis mappings and prerequisites for advanced recipes.

## Local checks

```bash
python -m pytest nova/tests -q
```

## Container

```bash
docker build -f nova/dockerfiles/Dockerfile -t radar-pd-nova:local nova
docker run --rm -p 8081:8081 \
  -v /SNS:/SNS:ro \
  -v /HFIR:/HFIR:ro \
  -e RADAR_PD_SNS_ROOT=/SNS \
  -e RADAR_PD_HFIR_ROOT=/HFIR \
  -e GALAXY_URL=https://ndip-test.ornl.gov \
  -e GALAXY_API_KEY=... \
  -e HISTORY_ID=... \
  -e EP_PATH=/ \
  radar-pd-nova:local
```

The image listens on port `8081`. Nginx preserves NOVA's injected entry-point
path and proxies HTTP and WebSocket traffic to the Trame server on port `8080`.

## NDIP deployment

1. Push the `birthright-container` branch and wait for the
   **Publish NOVA image to GitHub Container Registry** workflow to pass. It
   publishes both `nova-<version>` and immutable `nova-<git-sha>` tags.
2. Pin `nova/galaxy/radar_pd_nova.xml` to the immutable Git SHA tag from that
   successful workflow.
3. Copy the XML into the `prototype` branch under
   `tools/neutrons/powder_diffraction/radar_pd_nova.xml`.
4. Push `prototype` and launch **RADAR-PD Interactive** on `ndip-test`.

The interactive image is intentionally lightweight. It does not contain the
scientific catalog or run GSAS-II itself. It submits
`neutrons_radar_pd_analyze_prototype`, whose versioned batch image remains the
scientific execution environment and Galaxy remains the source of record for
inputs, jobs, outputs, and provenance.

Galaxy-authorized browsing does not require facility mounts inside the NOVA pod;
NDIP's Galaxy file-source configuration owns that access. Direct working-folder
browsing requires read-only `/SNS` and `/HFIR` mounts. Direct result publication
and directory watching are separate operational features and require a narrowly
scoped writer identity with access only to approved experiment folders. Raw data
trees must remain read-only.

## Transition tracking (0.4.11)

Live experiments open on **Results**, with the familiar phase-fraction,
refinement-quality and composition plots, completed-scan selector and full-result
link immediately available. The original individual-run Results, Interactive
Plots, Run Monitor and File Browser pages remain. **Transition tracking** is the
only additional experiment tab. Its linked pattern-symbol timeline,
diffraction intensity map and selected-scan inspector share the existing live
watch and Galaxy results; display actions do not submit or repeat refinements.

The separate Compare scans and Experiment trends tabs, nested evidence tabs and
uncalibrated shape-threshold control are retired from the live interface. The
observed-pattern tracker is the sole live pattern-comparison engine. Archived comparison
exports can still be read/replayed; existing configuration contracts are retained.
Qualified condition/device mappings are available in the tracker's Conditions
and provenance details and remain scoped to an instrument, IPTS and specimen.

Selectable derived plots remain under **Additional plots and derived quantities**
in Results. Supported cell, strain, distortion, hysteresis, normalized-progress
and correlation calculations retain their references, filters, uncertainties,
observations and downloads. Unimplemented broadening/site-order choices are not
offered. An individual fit does not need a duplicate experiment-trend panel or
one-point trend setup. Opening its page preserves the live experiment history.

The measured-peak detector sees measured coordinates, intensities, errors, masks
and explicit specimen/bank/reduction provenance. It allows independent local
peak movement and never receives CIF identities, hkl, fitted intensities,
Rwp, phase fractions or refined lattice parameters. Recurring symbols denote
compatible *observed patterns*, not identified chemical or magnetic phases.
The live boundary policy processes single arrivals. It compares with the current
group's fixed reference (default), or the preceding eligible scan. Earlier
assignments do not use later acquisitions. A measured change meeting the selected
rule creates a boundary candidate; a unique match to an earlier reference reuses its symbol.
Late/revised products and explicit policy edits replay received history; ordinary
arrivals append results. The original all-member comparison remains available
through the legacy engine API for reproducing older audit exports.

**Boundary indicators** lets the user select new/lost signal, resolved splitting,
relative peak width, local shape and relative intensity, and require any chosen
count. Defaults select appearance, splitting and width with one required.
Shape and intensity are opt-in because ordinary thermal effects can affect them.
The count is a rule, not a probability or independent statistical voting.
Changing it updates the indicator history, recurring markers and heatmap boundary
lines without submitting refinements. The indicator history plots the actual
selected-indicator count against a dashed required-count line, with an explicit
no-comparison row and unavailable-indicator bounds. It is not a continuous
similarity probability. Pattern symbols retain their categorical identity in the
timeline. The ambiguous filled boundary intervals have been removed.

The diffraction map defaults to a high-contrast Turbo rainbow palette (Cividis
remains selectable). A vertical scan cursor selects a measured-pattern ribbon
above the heatmap. Both share the horizontal zoom; that range survives cursor
movement and new scans. The ribbon uses the same background subtraction and
linear/asinh/log transformation as the map, while hover values retain the
measured intensity. Masks remain gaps.
White dashed lines mark boundary candidates. A green left-margin arrow and
highlighted run/condition tick identify the selected scan without drawing a
second horizontal line through the map.

TOF, d-spacing and Q are selectable when the exact physical/profile bank mapping
is available. Checksum-verified modern GSAS-II and legacy GSAS TOF profiles supply
the full coordinate equation including Zero, difA and difB where present; Q is
2π/d. The increasing positive physical branch is checked over the complete data
range. NOMAD nominal export geometry is never substituted for a calibrated
profile. Unsupported, missing, revised or mismatched profiles withhold the
unavailable axes. Coordinate conversion only relabels/reorders display samples;
intensity units and native statistical analysis remain unchanged.

Width evidence assumes fixed instrument/sample broadening within the qualified
setup. Isolated native-grid pseudo-Voigt fits measure FWHM/position with
conditional errors, residual-scatter inflation and a 0.5% relative error floor.
Defaults require a 3% width difference, error ratio 3 and two supporting regions.
This catches some unresolved broadening without declaring a resolved split.
Crowded, masked, undersampled or unsuccessful width estimates are unavailable.
The assumption, selections, sensitivity and policy hash persist in the watch
checkpoint and evidence export. They are not physical phase identification.

**Final Rwp change** is an optional sixth boundary indicator implemented in a
separate refinement adapter. It uses the explicit completed Full-analysis final
Rwp, never an arbitrary best candidate or a pending metric. Each result's project
and curve receipts must match. Between results, configuration, candidate library,
recipe, profile, coordinate/grid and fitted domain must agree; contributing
fitted points must overlap by at least 98%.
Small invalid-point variations are reported. Defaults require both 1 percentage
point and 10% relative change, in either direction; sensitivity and direction
are editable. It is whole-pattern, model-dependent evidence with no localized
peak interval. Defaults leave Rwp unselected. Late final results update received
assignments only when the Rwp indicator is selected; no refinement is repeated.
Qualified final-fit receipts are cached; ordinary arrivals parse only new or
revised receipts instead of reopening every older pattern.

**Changed-region overlay** offers Off (default), Ribbon only, and Ribbon +
compared scan pair. Choose the previous scan, boundary reference, or first scan
for localization. Boxes mark both endpoints of that comparison; adjacent pairs
share a box, while nonadjacent references mark only the two endpoint rows.
A local-indicator filter controls the evidence shown. Overlapping intervals
merge, common masked gaps are excluded, and up to 16 regions are shown with the
displayed total. Boxes never mark Rwp as a peak-local change or alter grouping.
The explicit Zoom to highlighted regions action links the ribbon and map to
their displayed intervals, making narrow evidence inspectable. Zooming adds
finer native display bins within the visible domain while retaining coarse
coverage across the full reference domain. Expanding the browser's axis before
a server refresh therefore cannot create artificial blank areas. The combined
grid is bounded to 1,025 columns. **Reset zoom** and double-clicking the map
restore both linked horizontal axes, full-domain data, and a fresh view revision;
color settings, selected scan and the boundary rule are preserved. Locked color
limits and the detector data remain unchanged.

**Splitting / doublet evolution** supplements resolved-component counting with
native, weighted local component fits. An asymmetric single pseudo-Voigt
competes with a common-width two-component model; dense, masked, poorly sampled,
unsuccessful and inadequately identified fits are withheld. A persistent doublet
can therefore show a measured separation change without gaining a maximum.
Defaults require both 0.10 component FWHM and a conditional error ratio of 3.
Errors include residual-scatter inflation and a 1.5% component-width floor.
These descriptive components do not identify phases or hkl: independently moving
neighboring reflections can also change separation. Recognized doublets below
the change rule optionally remain visible as **dotted** boxes; **solid** boxes
passed a local rule. The component measurements table explains the distinction.

The acquisition cursor previews native ribbons, row pointers and pair overlays
locally in the browser; full fit/evidence details load on release. Cached previews
have a 500,000-native-point budget, retaining the nearest scans without
resampling. A scan outside that cache loads on release. New arrivals continue to
be analyzed during dragging, with the visible index held stable until release.
The browser owns the thumb through dragging and until its latest request is
acknowledged. Client identity, experiment scope and gesture sequence prevent
older replies, duplicate release events and earlier tabs from moving it back.
Native pointer capture retains the drag outside the narrow control. Plot draws
run one at a time and discard intermediate waiting frames, so rapid direction
changes do not replay an outdated rendering backlog. The cursor widget's JavaScript
tests can be run with `node nova/tests/client_cursor.test.js` from the repository root.
Indicator and overlay multi-select controls keep the browser's current choices
while older server acknowledgements complete, so rapid clicks are not undone.

When direct facility watching is available, complete selected FXYE banks are
read without writing to the source and can appear before a refinement finishes.
The reader is bounded, checks declared record counts and a stable read, keeps
negative intensities and masks, and converts explicitly declared SLOG bin
integrals to density. Unsupported products and missing sample/geometry identity
remain visible as unavailable comparison evidence. Published, checksummed
`Pattern_observation.json` receipts provide a separate recovered-results route.
Their imported ordinate basis is retained rather than silently mixed with
reduced-source density. NOVA never unpickles projects for these views.

Selecting a timeline symbol or map row updates the same native-coordinate
inspector. Compare with the previous scan, first matching pattern or an explicit
pinned reference; switch to the existing saved refinement to inspect its
observed/calculated/difference curves. Candidate identities, reported fractions,
cell values and conditional uncertainties are shown separately. Missing fits do
not erase observed-pattern history. Inspecting an older scan pauses follow-latest.
The inspector can open that completed scan's full RADAR-PD result, including scans
hidden by a separate results display filter. A first scan is not compared with
itself as if an earlier observation existed.

The map defaults to asinh contrast and estimated local-background removal for
display; open **Map display** to change these options. Linear
and positive-only log views are also available. Log masks non-positive values
explicitly. Each display basis has one shared color scale, frozen from the
initial received window; arrivals do not silently recolor earlier rows. The
explicit rescale action uses the visible window. Peak-preserving display bins
and a maximum 100-row window bound browser work; inference uses native arrays,
and older observations remain accessible through scan selection. References,
marker identities and color limits are saved in the normal Galaxy watch
checkpoint without curve arrays or credentials.
The tracking report download exports the evidence index, settings and receipt
hashes; its recorded source products retain the measured curve arrays.

Pattern-state recurrence is an advisory extension of the calibrated pair
detector. Real-data replay and controlled cases test its mechanics; these
markers do not independently confirm thermodynamic transitions, magnetic origin,
propagation vectors or space-group changes. Texture, normalization, reduction
and sample-environment changes remain alternative explanations. Broad diffuse
changes and movements outside the configured search bound require review.

## Archived scan interpretation (0.4.4)

Version 0.4.4 introduced **Compare scans**, comparing each acquisition with the
previous compatible scan, a pinned reference, matching recorded conditions,
and a bounded search of similar recent history. Separate specimen, bank,
profile, reduction-family and setup identities are retained. Job completion
order does not define acquisition order. Earlier pending jobs keep reference
and persistence conclusions provisional; multiple fits of one acquisition need
an explicit active revision.

The measured overlay and difference are visible on the main page. Selecting a
pattern observation focuses both axes on its region; refinement changes use a
separate table with readable candidate labels, and conditions have their own
tab. Each visible observation belongs to the selected pair, rather than mixing
adjacent-scan and pinned-reference findings. **Experiment trends** is a separate
view in that version. The live interface now uses the consolidated 0.4.5 layout
above. Curve zoom, the calculated-fit toggle and these view choices did not rerun
or change the analysis. The expanded inspector retains the same evidence.

The updated Analyze collector publishes `tables/Pattern_observation.json` from
its selected full-mode GPX. It preserves native histogram coordinates, measured
intensities, reported errors, calculated intensities, masks and fit domains.
NOVA fetches this checksum-addressed artifact independently of the large result
archive. Older outputs and Rapid results without a selected project show missing
curve evidence explicitly; updating NOVA alone cannot supply that evidence.

Named NeXus condition logs carry units, readback/request roles, time origins,
coverage, exposure-weighted summaries and trajectories. Parent-run conditions
are labeled when a reduced product's narrower time interval is unconfirmed.
Pressure/gas devices and ambiguous field readbacks require a verified,
specimen-scoped device mapping. Unqualified units and incomplete time coverage
cannot be repaired by choosing a device. Raw receipts remain unchanged.

Findings are deterministic candidates with alternatives and inspectable curve
regions. Practical effect thresholds and conservative reported-error gates
are engineering defaults, not calibrated event probabilities or phase detection
limits. Missing candidates are not zero fractions. Model comparisons require a
compatible candidate basis, recipe and fitted domain. Scientific input warnings
remain applicable. This feature issues no experiment-completion or beamline
control command.

Settings, pinned reference hashes, active revisions and a compact findings index
are saved with the watch checkpoint. **Export comparison** includes exact
curve packets, the pair's metadata, settings and evidence. After extracting it:

```bash
python -m radar_pd_nova.scan_interpretation manifest.json \
  --settings settings.json --output replay.json
```

Replaying a pair does not reproduce full-history persistence conclusions without
the original history receipts. UI phase filters, zoom, feature-region selection
and fit visibility do not alter the inference inputs. Historical replay runs in
a coalescing background worker so the interface remains responsive.
