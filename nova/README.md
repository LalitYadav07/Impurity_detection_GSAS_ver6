# RADAR-PD NOVA

This package is the interactive NDIP/NOVA client for RADAR-PD. It does not run
the scientific pipeline in the interactive container. It uploads or resolves
inputs, submits the existing `neutrons_radar_pd_analyze_prototype` Galaxy tool,
recovers jobs from the user's Galaxy history, and renders completed artifacts.

The Full and Rapid paths share one setup form and one Galaxy submission
contract. Scientific work remains inside the versioned RADAR-PD batch image.

## Local checks

```bash
python -m pytest nova/tests -q
```

## Container

```bash
docker build -f nova/dockerfiles/Dockerfile -t radar-pd-nova:local nova
docker run --rm -p 8081:8081 \
  -e GALAXY_URL=https://ndip-test.ornl.gov \
  -e GALAXY_API_KEY=... \
  -e HISTORY_ID=... \
  -e EP_PATH=/ \
  radar-pd-nova:local
```

The image listens on port `8081`. Nginx preserves NOVA's injected entry-point
path and proxies HTTP and WebSocket traffic to the Trame server on port `8080`.

## NDIP deployment

1. Build and publish an immutable image, for example
   `savannah.ornl.gov/radar-pd/radar-pd-nova:<git-sha>`.
2. Replace `REPLACE_WITH_IMMUTABLE_TAG` in
   `nova/galaxy/radar_pd_nova.xml` with that tag.
3. Copy the XML into the `prototype` branch under
   `tools/neutrons/powder_diffraction/radar_pd_nova.xml`.
4. Push `prototype` and launch **RADAR-PD Interactive** on `ndip-test`.

The interactive image is intentionally lightweight. It does not contain the
scientific catalog or run GSAS-II itself. It submits
`neutrons_radar_pd_analyze_prototype`, whose versioned batch image remains the
scientific execution environment and Galaxy remains the source of record for
inputs, jobs, outputs, and provenance.
