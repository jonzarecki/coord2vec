#!/bin/bash

#conda init bash


if conda --version | grep -q "conda" ; then # make sure anaconda is installed
#    eval "$(conda shell.bash hook)"
#    conda env update -f environment.yml
#    conda activate coord2vec
    echo "Virtual env updated, activate using 'conda activate coord2vec'"
else
    exit 1
fi

#if [[ -n `which java` ]] ; then
#    echo "java is installed"
#else
#    sudo apt install openjdk-11-jdk -y
#    sudo update-alternatives --config java
#fi

# start osm servers
sudo chmod 777 /var/run/docker.sock
bash ./coord2vec/feature_extraction/osm/initialize_osm_postgres.sh
bash ./coord2vec/feature_extraction/ors/initialize_ors.sh
bash ./coord2vec/image_extraction/init_tile_servers.sh

echo `dig +short myip.opendns.com @resolver1.opendns.com`
conda activate coord2vec
cd tmp/coord2vec_zarecki
jupyter notebook --ip=0.0.0.0 --port=8190
# run seperatly
conda activate coord2vec
tensorboard --logdir /media/yonatanz/yz/tensorboard_runs/ --port=8192

