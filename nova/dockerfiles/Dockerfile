FROM savannah.ornl.gov/radar-pd/radar-pd-nova:nova-0.3.2@sha256:657057335458b5484853494ed929470bd0817bf83a4afcb3e0320a026889d7d7

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
RUN python -m pip install --no-cache-dir --no-deps /src \
    && python -m compileall -q /src/src/radar_pd_nova
