FROM regproxy.ornl.gov/hub_proxy/kitware/trame:py3.10-2025-05

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext-base nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir /src

COPY dockerfiles/nginx.conf.template /etc/nginx/templates/radar-pd.conf.template
COPY scripts/run_container.sh scripts/run_nginx.sh scripts/run_trame.sh /usr/local/bin/
RUN chmod 0755 \
        /usr/local/bin/run_container.sh \
        /usr/local/bin/run_nginx.sh \
        /usr/local/bin/run_trame.sh \
    && touch /var/log/nginx/error.log \
    && chmod 0666 /var/log/nginx/error.log \
    && rm -f /etc/nginx/sites-enabled/default

ENV PYTHONUNBUFFERED=1 \
    TRAME_CLIENT_TYPE=vue3 \
    EP_PATH=/

EXPOSE 8081
ENTRYPOINT []
USER 1000:1000
CMD ["/usr/local/bin/run_container.sh"]
