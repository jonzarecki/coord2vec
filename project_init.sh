#!/usr/bin/env bash


conda --version  # make sure anaconda is installed
conda env create -f environment.yml
conda activate coord2vec


bash ./coord2vec/feature_extraction/osm/initialize_osm_postgres.sh
bash ./coord2vec/image_extraction/init_tile_servers.sh


sudo apt install openjdk-11-jdk
java -version
sudo update-alternatives --config java

