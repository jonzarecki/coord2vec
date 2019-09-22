FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

# run with nvidia-docer (command installed via apt)

# Based on
# https://switch2osm.org/manually-building-a-tile-server-18-04-lts/

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install useful commands
RUN apt-get update
RUN apt-get install software-properties-common -y
# Install java (for h20)
RUN add-apt-repository ppa:openjdk-r/ppa -y
RUN apt-get update
RUN apt-get install openjdk-9-jdk -y

# Install coord2vec environment
WORKDIR /home/root
RUN curl https://gist.githubusercontent.com/jonzarecki/bef39acdd631f6cbba6ce9b04b539138/raw/ef4e5542409259d871189978495bee271b75ad53/geo_envffile.yml > environment.yml
RUN conda env create -f environment.yml
RUN conda init bash

# Init coord2vec environment
ENV PATH /opt/conda/envs/coord2vec/bin:$PATH
RUN /bin/bash -c "source activate coord2vec"

# Start running
USER root
WORKDIR /home/root
# Init coord2vec environment
#RUN echo "conda activate coord2vec" >> ~/.basrhc
#ENV PATH /opt/conda/envs/coord2vec/bin:$PATH
#RUN /bin/bash -c "source activate coord2vec"
#RUN conda init bash
#RUN conda activate coord2vec
ENTRYPOINT ["/bin/bash"]
CMD []