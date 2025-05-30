FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    ca-certificates && \
    add-apt-repository ppa:sumo/stable && \
    apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip ca-certificates \
    libxerces-c-dev libfox-1.6-dev libgdal-dev \
    libgl1-mesa-dev libglu1-mesa-dev libpng-dev libproj-dev \
    libxml2-dev libtool python3 python3-dev python3-pip python3-setuptools \
    libffi-dev libssl-dev curl libglib2.0-0 libgl1 \
    sumo sumo-tools sumo-doc \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

ENV SUMO_HOME=/usr/share/sumo

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]