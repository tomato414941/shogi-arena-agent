ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        openssh-client \
        p7zip-full \
        python3-venv \
        rclone \
        rsync \
        unzip \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "cshogi>=0.9.7" \
        "numpy<2" \
        "python-shogi>=1.1.1" \
        "tokenizers>=0.23.1" \
        "zstandard>=0.23"
