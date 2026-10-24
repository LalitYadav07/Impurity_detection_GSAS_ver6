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

#### POWGEN

The monitor handles large `shared/autoreduce` folders by considering only
canonical `PG3_<run>.gsa` files, without a 5,000-entry cutoff or recursive
searches. After entering an IPTS, leave the field or click **Refresh sample IDs**
to populate sample IDs, all reported names, and completed-scan counts from
the corresponding NeXus files, per-run run-info XML and validated scan tables.
Only bounded sample identity fields are read for this
list; detector events are never loaded.

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

The **POWGEN Live Experiment** panel is a session-scoped orchestration layer
around the existing RADAR-PD Analyze tool. It does not duplicate or modify the
scientific pipeline. The user selects an IPTS, the experiment wavelength, a
reusable Full/Rapid configuration, and an optional main-phase CIF. While the
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
