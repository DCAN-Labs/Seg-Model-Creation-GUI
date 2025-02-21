# Use Python base image
FROM python:3.7-slim

# Install system dependencies for PyQt5
RUN apt-get update && apt-get install -y \
    python3-pyqt5 \
    qtbase5-dev \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    git \
    build-essential \
    python3-dev \
    python3-distutils \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone repositories
RUN git clone https://github.com/DCAN-Labs/Seg-Model-Creation-GUI.git && \
    git clone https://github.com/DCAN-Labs/SynthSeg.git && \
    git clone https://github.com/DCAN-Labs/dcan-nn-unet.git

# Install Python dependencies
# Assuming your GUI repo has a requirements.txt
COPY requirements.txt .
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --no-cache-dir -r requirements.txt

# Set environment variables for repository paths
ENV SYNTHSEG_PATH=/app/SynthSeg
ENV NNUNET_PATH=/app/dcan-nn-unet

# Set display environment variable for GUI
ENV DISPLAY=:0

# Command to run the GUI
# Replace with your actual startup command
CMD ["python", "/app/Seg-Model-Creation-GUI/pyqt_test.py"]
