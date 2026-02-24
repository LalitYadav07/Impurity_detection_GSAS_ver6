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

# Download both databases from Google Drive at build time so they are baked into the image.
# Uses a dedicated Python script to handle Windows-style backslash paths in the archive.
RUN pip install --quiet gdown && \
    gdown 1BxPXjdbn7oYTXKfDeLct5-2PMkhcLVSH -O /tmp/database_neutron.zip && \
    python3 scripts/extract_xray_db.py /tmp/database_neutron.zip data/database_neutron && \
    rm -f /tmp/database_neutron.zip && \
    gdown 12H19jI3mGcYBpJrQRtY-5_WaMjFyIMah -O /tmp/database_xray.zip && \
    python3 scripts/extract_xray_db.py /tmp/database_xray.zip data/database_xray && \
    rm -f /tmp/database_xray.zip

# Copy the rest of the application
COPY --chown=user . .

# Expose Streamlit port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
