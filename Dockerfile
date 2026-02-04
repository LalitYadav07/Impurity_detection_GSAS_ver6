FROM python:3.11-slim

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
    libgl1-mesa-glx \
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

# Create ML_components and download models from GitHub (since HF rejects binary pushes)
RUN mkdir -p ML_components && \
    curl -L https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6/raw/master/ML_components/residual_training.pt -o ML_components/residual_training.pt && \
    curl -L https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6/raw/master/ML_components/two_phase_training.pt -o ML_components/two_phase_training.pt

# Copy the rest of the application
COPY --chown=user . .

# Expose Streamlit port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
