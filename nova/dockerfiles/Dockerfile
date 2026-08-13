FROM regproxy.ornl.gov/hub_proxy/kitware/trame:py3.10-2025-05

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext-base nginx wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir supervisor /src

RUN python -m trame.tools.www --client-type vue3 --output /app/www-content \
    && mkdir -p /app/www-content/assets/js \
    && NOVA_TRAME_LOCATION="$(python -c 'import os; import nova.trame; print(os.path.dirname(nova.trame.__file__))')" \
    && cp "${NOVA_TRAME_LOCATION}"/view/theme/assets/js/*.js /app/www-content/assets/js/

COPY dockerfiles/nginx.conf.template /etc/nginx/nginx.conf.template
COPY dockerfiles/supervisord.conf /etc/supervisord.conf
COPY scripts/run_nginx.sh /run_nginx.sh
COPY scripts/run_trame.sh /run_trame.sh
RUN chmod 0755 \
        /run_nginx.sh \
        /run_trame.sh \
    && rm -f /etc/nginx/sites-enabled/default \
    && chmod og+rwX -R /var/lib/nginx /var/log/nginx /etc/nginx /root /src

ENV PYTHONUNBUFFERED=1 \
    TRAME_CLIENT_TYPE=vue3 \
    EP_PATH=/

EXPOSE 8081
ENTRYPOINT []
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
