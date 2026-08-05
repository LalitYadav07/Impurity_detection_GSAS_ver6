# RADAR-PD NDIP Batch Image

`Dockerfile.ndip` builds the batch image used by the local Galaxy wrappers. It runs the existing Full or Rapid pipeline through `scripts/ndip_runner.py`; it does not run the Streamlit interface.

## Local build

```bash
docker build -f Dockerfile.ndip -t radar-pd-ndip:local .
docker run --rm radar-pd-ndip:local --help
```

The crystallographic catalogs are deliberately not downloaded during this image build. For local testing, mount a catalog root at `/opt/radar-pd/data` or pass `--db-pack` to `analyze`. The eventual NDIP image should use a versioned catalog layer or a read-only mounted catalog and record its version in `RADAR_PD_DATABASE_VERSION`.

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
