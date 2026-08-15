FROM savannah.ornl.gov/radar-pd/radar-pd-nova:nova-0.1.39

USER root

WORKDIR /src
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall /src
