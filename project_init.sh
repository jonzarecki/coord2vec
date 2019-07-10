#!/usr/bin/env bash

conda init bash
conda --version  # make sure anaconda is installed
#conda env create -f environment.yml
conda activate coord2vec

sudo chmod 777 /var/run/docker.sock
bash ./coord2vec/feature_extraction/osm/initialize_osm_postgres.sh
bash ./coord2vec/image_extraction/init_tile_servers.sh

sudo apt install openjdk-11-jdk -y
java -version
sudo update-alternatives --config java

bash jupyter notebook --ip=0.0.0.0 --port=8200