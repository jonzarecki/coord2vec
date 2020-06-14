FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install useful commands
RUN apt-get update && apt-get install -y \
      software-properties-common \
      cmake \
      git \
      curl wget \
      ca-certificates \
      nano vim \
      openssh-server

# Install geoml environment
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

WORKDIR /home/root
COPY environment.yaml /opt/geoml_environment.yml
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda env create -f /opt/geoml_environment.yml && conda clean -ya


# Init coord2vec environment
ENV CONDA_DEFAULT_ENV=geoml
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /bin/bash -c "source activate $CONDA_DEFAULT_ENV"

# Start running
USER root
WORKDIR /home/root

ENTRYPOINT ["/bin/bash"]
CMD []