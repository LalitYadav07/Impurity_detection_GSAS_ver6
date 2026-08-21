FROM savannah.ornl.gov/radar-pd/radar-pd-nova:nova-0.3.2

USER root

# Hide NOVA developer-resource links in the deployed scientific workspace.
ENV PIXI_ENVIRONMENT_NAME=production \
    PYTHONPATH=/src/src

WORKDIR /src
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python -m compileall -q /src/src/radar_pd_nova
