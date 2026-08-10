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
