FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libgfortran5 \
    libgomp1 \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create ML_components and download models from GitHub
# .pt files are stored as regular git objects (not LFS) so raw URL works
RUN mkdir -p ML_components && \
    curl -L https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6/raw/main/ML_components/residual_training.pt -o ML_components/residual_training.pt && \
    curl -L https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6/raw/main/ML_components/two_phase_training.pt -o ML_components/two_phase_training.pt

# Copy just the extraction helper early (allows caching of the download layer)
COPY --chown=user scripts/extract_xray_db.py scripts/extract_xray_db.py

# Best-effort catalog bake. Google Drive can intermittently block headless
# builders with quota/public-link checks, so catalog download must not prevent
# the Space image from building. If this step cannot fetch a catalog, the app
# still starts and shows the database install/upload controls in the UI.
RUN pip install --quiet gdown && \
    mkdir -p data/database_neutron data/database_xray && \
    ( \
      gdown 1BxPXjdbn7oYTXKfDeLct5-2PMkhcLVSH -O /tmp/database_neutron.zip && \
      python3 scripts/extract_xray_db.py /tmp/database_neutron.zip data/database_neutron && \
      rm -f /tmp/database_neutron.zip \
    ) || ( \
      echo "WARNING: neutron database download failed during Docker build; install it from the app UI." && \
      rm -f /tmp/database_neutron.zip \
    ) && \
    ( \
      gdown 12H19jI3mGcYBpJrQRtY-5_WaMjFyIMah -O /tmp/database_xray.zip && \
      python3 scripts/extract_xray_db.py /tmp/database_xray.zip data/database_xray && \
      rm -f /tmp/database_xray.zip \
    ) || ( \
      echo "WARNING: X-ray database download failed during Docker build; install it from the app UI." && \
      rm -f /tmp/database_xray.zip \
    )

# Copy the rest of the application
COPY --chown=user . .

# Expose Streamlit port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
