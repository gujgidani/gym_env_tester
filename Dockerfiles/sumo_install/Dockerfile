# Base image
FROM ubuntu:22.04

# Avoid interactive prompts during builds
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies required for SUMO and add SUMO PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    ca-certificates && \
    add-apt-repository ppa:sumo/stable && \
    apt-get update

# Install SUMO and its tools only
RUN apt-get install -y --no-install-recommends \
    sumo sumo-tools sumo-doc \
 && rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME environment variable
ENV SUMO_HOME=/usr/share/sumo

# Set working directory
WORKDIR /app

# Files can be copied here
# COPY ..

# Default command
CMD ["bash"]
