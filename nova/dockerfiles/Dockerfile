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
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends hdf5-tools tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY src /src/src
# Serve the versioned guide through the same path-prefixed NDIP endpoint as NOVA.
COPY help /app/www-content/user-guide/
# The pinned runtime uses Python 3.10. Install the numerical/metadata readers
# explicitly: the application install intentionally skips dependency resolution.
# The h5py wheel reuses the existing NumPy; it must not upgrade the NOVA stack.
RUN python -m pip install --no-cache-dir 'scipy==1.15.3' \
    && python -m pip install --no-cache-dir --no-deps --only-binary=:all: 'h5py==3.13.0' \
    && python -m pip install --no-cache-dir --no-deps 'RapidFuzz==3.14.3' \
    && python -m pip install --no-cache-dir --no-deps /src \
    && python -m compileall -q /src/src/radar_pd_nova
