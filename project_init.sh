#!/usr/bin/env bash


conda --version  # make sure anaconda is installed
conda env create -f environment.yml
conda activate coord2vec


bash ./feature_extraction/osm/initialize_osm_postgres.sh