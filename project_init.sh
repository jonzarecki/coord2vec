#!/usr/bin/env bash


conda --version  # make sure anaconda is installed
conda env create -f environment.yml
conda activate coord2vec


bash ./coord2vec/feature_extraction/osm/initialize_osm_postgres.sh
bash ./coord2vec/image_extraction/init_tile_servers.sh

sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer


sudo apt-get update && sudo apt-get upgrade
wget -c --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/11.0.1+13/90cf5d8f270a4347a95050320eef3fb7/jdk-11.0.1_linux-x64_bin.tar.gz
mkdir /opt/java
tar -zxf jdk-11.0.1_linux-x64_bin.tar.gz -C /opt/java

sudo apt install openjdk-11-jdk
java -version
sudo update-alternatives --config java

