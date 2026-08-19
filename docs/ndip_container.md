# RADAR-PD NDIP Batch Image

`Dockerfile.ndip` builds the batch image used by the local Galaxy wrappers. It runs the existing Full or Rapid pipeline through `scripts/ndip_runner.py`; it does not run the Streamlit interface.

## Local build

```bash
docker build -f Dockerfile.ndip -t radar-pd-ndip:local .
docker run --rm radar-pd-ndip:local --help
```

The crystallographic catalogs are deliberately not downloaded during this image build. For local testing, mount a catalog root at `/opt/radar-pd/data` or pass `--db-pack` to `analyze`. The eventual NDIP image should use a versioned catalog layer or a read-only mounted catalog and record its version in `RADAR_PD_DATABASE_VERSION`.

## Internal image with built-in catalogs

For NDIP deployment, generate a versioned bundle from the controlled ORNL
catalog tree and add it to the tested application image. The generated archive
is ignored by Git and is published only as part of an internal Harbor image.

```bash
python scripts/ndip_catalog_bundle.py build \
  --source /path/to/radar-pd-local-source-data \
  --output .ndip_catalog/radar-pd-catalog.tar.gz \
  --version catalog-YYYYMMDD

docker build \
  --file Dockerfile.ndip.with-catalog \
  --build-arg RADAR_PD_APP_IMAGE=savannah.ornl.gov/radar-pd/radar-pd:TAG \
  --tag savannah.ornl.gov/radar-pd/radar-pd:TAG-catalog-YYYYMMDD \
  .
```

The bundle contains only immutable `database_neutron` and `database_xray`
assets. It never includes user libraries, runs, or workspaces. The identical
metadata JSON used by both databases is stored once and linked at image-build
time. `Dockerfile.ndip.with-catalog` verifies every required runtime file and
fails the build if the catalog is incomplete.

## Batch entry points

```text
configure       create a portable path-independent YAML configuration
analyze         run Full or Rapid RADAR-PD and normalize outputs
resolve-ipts    resolve one reduced run from the mounted SNS IPTS tree
build-library   build a mini or augmented CIF catalog
compare-series  combine normalized summaries from mapped jobs
collect         normalize an existing run directory
```

The image writes Galaxy-ready collections under the selected output directory: `plots/`, `tables/`, `phases/`, `gpx/`, and `diagnostics/`. It also writes `summary.json`, `state.json`, `gpx_index.json`, `report.html`, and `results.zip`.

## SNS IPTS folder watcher

The same image contains a restart-safe worker for continuously arriving reduced
patterns. The NOVA application writes a `radar-pd-watch/v1` recipe and its
portable RADAR-PD configuration into an explicitly selected IPTS `shared/`
folder. A managed NDIP worker consumes that recipe:

```bash
python /opt/radar-pd/scripts/ndip_ipts_watch.py \
  --recipe /SNS/HB2A/IPTS-12345/shared/radar/watch/radar-pd-watch.json \
  --facility-root /SNS
```

Use `--once` for a scheduler-driven sweep; omit it for a persistent worker. The
worker waits for a file's size and modification time to remain stable, records
fingerprints in an atomic state file, uses a recoverable heartbeat lease,
retries bounded failures, and publishes
each completed result directory by atomic rename. Failed attempts retain their
logs in numbered evidence directories and never overwrite a previous result.

The worker must be deployed by NDIP operations as a managed service or scheduled
job. Saving a recipe in NOVA does not start a background process inside the
interactive browser session.

## Facility mount contract

Ordinary one-off browsing should use Galaxy-authenticated remote-file sources;
the selected object is imported into History before Analyze runs, and the NOVA
pod does not need a native `/SNS` path. Mount the SNS tree read-only only when
direct mounted browsing or run resolution is explicitly enabled. Publishing
results and directory watching require a separate, narrowly scoped read-write
mount that permits the worker identity to write only to the selected
experiment's `shared/` tree. RADAR-PD rejects paths outside
`/SNS/<instrument>/IPTS-*/shared`, path traversal, and symlink escapes. Raw
NeXus directories are never writable through this integration.
