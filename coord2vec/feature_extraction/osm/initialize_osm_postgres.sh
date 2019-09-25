#!/usr/bin/env bash

p=`pwd`
cd "${0%/*}/"
echo `pwd`

docker stop $(docker ps -aq --filter ancestor=osm-postgis-server) && docker rm $(docker ps -aq --filter ancestor=osm-postgis-server)
docker volume rm osm-postgis-data
docker volume create osm-postgis-data
sudo chmod 777 osm-postgis-server/run.sh

docker build osm-postgis-server/ -t osm-postgis-server:0.1

# Download Israel as sample if no data is provided
if [[ ! -f ./data.osm.pbf ]]; then
    echo "WARNING: No import file at /data.osm.pbf, so importing Israel as example..."
    wget https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf -O ./data.osm.pbf
fi

cdir=`pwd`
docker run --rm -e THREADS=24 -p 127.0.0.1:15432:5432 -v $cdir/data.osm.pbf:/data.osm.pbf --name osm-postgis-server \
                -d -v osm-postgis-data:/var/lib/postgresql/10/main osm-postgis-server:0.1 import
cd $p


# restart exited docker
#    docker start  `docker ps -q -l` # restart it in the background
#    docker attach `docker ps -q -l` # reattach the terminal & stdin


############ now done in the dockerfile
# Change docker to
#    nano /etc/postgresql/10/main/postgresql.conf
#    # change  listen_address='*'
#
#    nano /etc/postgresql/10/main/pg_hba.conf
#    # add  `host    all             all             0.0.0.0/0            trust`
#
#    now you can write `psql -h 127.0.0.1 -p 15432 -d gis -U renderer`
#    and it will move to the correct postgres cmdline