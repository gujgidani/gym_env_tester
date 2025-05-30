# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and add the SUMO stable PPA to get the latest SUMO version
RUN apt-get update && apt-get install -y \
    software-properties-common \  # Enables adding PPAs
    gnupg \                       # Required for verifying package sources
    ca-certificates && \         # For secure HTTPS connections
    add-apt-repository ppa:sumo/stable && \  # Add SUMO official repository
    apt-get update               # Refresh package lists after adding new repository

# Install required system packages and SUMO tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip ca-certificates \        # Build tools and utilities
    libxerces-c-dev libfox-1.6-dev libgdal-dev \                   # SUMO dependencies
    libgl1-mesa-dev libglu1-mesa-dev libpng-dev libproj-dev \     # Graphics and GIS libraries
    libxml2-dev libtool python3 python3-dev python3-pip python3-setuptools \  # Python environment
    libffi-dev libssl-dev curl libglib2.0-0 libgl1 \               # Networking and runtime dependencies
    sumo sumo-tools sumo-doc \                                     # SUMO simulator, tools, and docs
 && rm -rf /var/lib/apt/lists/*                                   # Clean up to reduce image size

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set working directory inside the container
WORKDIR /app

# Set the SUMO_HOME environment variable for Python-based SUMO scripts
ENV SUMO_HOME=/usr/share/sumo

# Copy Python dependencies file into the container
COPY requirements.txt .

# Install all required Python packages
RUN pip install -r requirements.txt

# Copy the entire current project directory into the container
COPY . .

# Start a Bash shell when the container runs (can be overridden by other commands)
CMD ["bash"]
