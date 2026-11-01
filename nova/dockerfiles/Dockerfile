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
# The pinned runtime uses Python 3.10. Install the numerical/metadata readers
# explicitly: the application install intentionally skips dependency resolution.
# The h5py wheel reuses the existing NumPy; it must not upgrade the NOVA stack.
# The reviewed PyTorch CPU wheel excludes CUDA packages. Its content hash pins
# the download even where ORNL network certificate interception requires a
# host-specific trust exception.
RUN python -m pip install --no-cache-dir 'scipy==1.15.3' \
    && python -m pip install --no-cache-dir --no-deps --only-binary=:all: 'h5py==3.13.0' \
    && python -m pip install --no-cache-dir --no-deps 'RapidFuzz==3.14.3' \
    && python -m pip install --no-cache-dir --no-deps 'galvani==0.5.0' \
    && python -m pip install --no-cache-dir --trusted-host download-r2.pytorch.org \
       'https://download-r2.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=7f91a2200e352745d70e22396bd501448e28350fbdbd8d8b1c83037e25451150'

# Keep the expensive fixed runtime cached across source and guide edits.
COPY src /src/src
RUN python -m pip install --no-cache-dir --no-deps /src \
    && python -c 'from radar_pd_nova.unet_core.production import FocusTransitionPredictor; FocusTransitionPredictor()._load()' \
    && python -m compileall -q /src/src/radar_pd_nova /src/src/gpx_recipe

# Documentation-only updates must not reinstall the scientific/ML runtime.
COPY help /app/www-content/user-guide/
