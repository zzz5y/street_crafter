FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv gcc-11 g++-11 cmake

# conda
ENV PATH /opt/conda/bin:$PATH 
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf /tmp/*

RUN sudo apt update
RUN sudo apt install rustc cargo
RUN sudo apt install curl


# Create the user
RUN addgroup --gid $GROUP_ID user
RUN useradd --create-home -s /bin/bash --uid $USER_ID --gid $GROUP_ID docker
RUN adduser docker sudo
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER docker

# Setup street_crafter
RUN /opt/conda/bin/python -m ensurepip
RUN /opt/conda/bin/python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118


RUN sudo apt-get update && sudo apt-get install -y ca-certificates



WORKDIR /host

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN sudo chmod +x /entrypoint.sh

COPY requirements.txt /host/requirements_docker.txt
RUN /opt/conda/bin/python -m pip cache purge
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade pip setuptools
RUN /opt/conda/bin/python -m pip install --no-cache-dir ujson
# RUN /opt/conda/bin/python -m pip install -v --no-cache-dir -r requirements_docker.txt