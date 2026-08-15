FROM savannah.ornl.gov/radar-pd/radar-pd-nova:nova-0.2.0

USER root

# Hide NOVA developer-resource links in the deployed scientific workspace.
ENV PIXI_ENVIRONMENT_NAME=production

WORKDIR /src
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall /src
