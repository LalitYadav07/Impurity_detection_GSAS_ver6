# Verified NDIP-test runtime from the 0.4.0 release. Application source below
# replaces the prior package; builds do not require Savannah availability.
FROM ghcr.io/lalityadav07/impurity_detection_gsas_ver6:nova-78f2dcedcd5eed1db1c0885a47dc47cac18d63e0@sha256:9bd0abb274eec3bfe4a55a01b4beccecf55660afa920be7d4be6e1ef46aa0a5e

USER root

# Hide NOVA developer-resource links in the deployed scientific workspace.
ENV PIXI_ENVIRONMENT_NAME=production \
    PYTHONPATH=/src/src

WORKDIR /src
COPY pyproject.toml README.md /src/
RUN apt-get update \
    && apt-get install -y --no-install-recommends hdf5-tools \
    && rm -rf /var/lib/apt/lists/*
COPY src /src/src
# The pinned runtime uses Python 3.10 and does not contain SciPy. Keep the
# measured-pattern engine reproducible without upgrading the NOVA stack.
RUN python -m pip install --no-cache-dir 'scipy==1.15.3' \
    && python -m pip install --no-cache-dir --no-deps /src \
    && python -m compileall -q /src/src/radar_pd_nova
