FROM mambaorg/micromamba:2.9.0-ubuntu22.04

ARG GSASII_REF=dcd09eb2a4392b94d22cda3c69021137e6b14620
ARG GSASII_UID=1000
ARG GSASII_GID=1000
ARG RADAR_PD_GSASII_GUI_VERSION=0.1.10

LABEL org.opencontainers.image.title="RADAR-PD GSAS-II Interactive" \
      org.opencontainers.image.description="Browser-hosted native GSAS-II desktop for RADAR-PD GPX handoff" \
      org.opencontainers.image.source="https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6" \
      org.opencontainers.image.gsasii.revision="${GSASII_REF}" \
      org.opencontainers.image.version="${RADAR_PD_GSASII_GUI_VERSION}"

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    PATH=/opt/conda/bin:${PATH} \
    DISPLAY=:1 \
    HOME=/home/gsasii \
    USER=gsasii \
    LOGNAME=gsasii \
    XDG_RUNTIME_DIR=/tmp/runtime-gsasii \
    GSASII_REF=${GSASII_REF} \
    GSASII_GUI_VERSION=${RADAR_PD_GSASII_GUI_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        adwaita-icon-theme \
        build-essential \
        ca-certificates \
        dbus-x11 \
        fonts-dejavu-core \
        fonts-liberation \
        gfortran \
        gettext-base \
        git \
        libgl1-mesa-dri \
        libglu1-mesa \
        libgtk-3-0 \
        libnss-wrapper \
        libxkbcommon-x11-0 \
        nginx \
        novnc \
        openbox \
        supervisor \
        websockify \
        x11-utils \
        x11-xserver-utils \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER gsasii_gui/environment.yml /tmp/gsasii-environment.yml
USER $MAMBA_USER
RUN micromamba install --yes --name base --file /tmp/gsasii-environment.yml \
    && micromamba clean --all --yes

USER root
RUN git clone --filter=blob:none https://github.com/AdvancedPhotonSource/GSAS-II.git /opt/GSAS-II \
    && cd /opt/GSAS-II \
    && git checkout --detach "${GSASII_REF}"

RUN /opt/conda/bin/python -m pip install \
        --no-build-isolation \
        --no-cache-dir \
        --no-deps \
        /opt/GSAS-II \
    && /opt/conda/bin/python -c "import GSASII; import wx; print('GSAS-II and wxPython imports succeeded')"

RUN groupadd --gid "${GSASII_GID}" gsasii \
    && useradd --uid "${GSASII_UID}" --gid gsasii --create-home --shell /bin/bash gsasii \
    && mkdir -p /workspace /tmp/runtime-gsasii /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix \
    && chown -R gsasii:gsasii /workspace /tmp/runtime-gsasii /home/gsasii

COPY gsasii_gui/nginx.conf.template /etc/nginx/gsasii.conf.template
COPY gsasii_gui/supervisord.conf /etc/supervisor/conf.d/gsasii.conf
COPY gsasii_gui/scripts/ /opt/gsasii-gui/
RUN chmod 0755 /opt/gsasii-gui/*.sh

EXPOSE 8080
WORKDIR /workspace
USER gsasii

ENTRYPOINT []
CMD ["/opt/gsasii-gui/start.sh"]
