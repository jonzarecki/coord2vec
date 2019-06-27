#!/usr/bin/env bash

p=`pwd`
cd "${0%/*}"

docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
docker volume rm openstreetmap-data
docker volume create openstreetmap-data
docker build osm-postgis-server/ -t osm-postgis-server

# Download Israel as sample if no data is provided
if [[ ! -f ./data.osm.pbf ]]; then
    echo "WARNING: No import file at /data.osm.pbf, so importing Israel as example..."
    wget https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf -O ./data.osm.pbf
fi

cdir=`pwd`
docker run -e THREADS=24 -p 5432:5432 -v $cdir/data.osm.pbf:/data.osm.pbf -v openstreetmap-data:/var/lib/postgresql/10/main osm-postgis-server import
cd $p


# restart exited docker
#docker start  `docker ps -q -l` # restart it in the background
#docker attach `docker ps -q -l` # reattach the terminal & stdin

nano /etc/postgresql/10/main/postgresql.conf
# change  listen_address='*'

nano /etc/postgresql/10/main/pg_hba.conf
# add  host    all             all             0.0.0.0/0            trust

now you can write `psql -h localhost -U postgres -d gis -U postgres`
and it will move to the correct postgres cmdline
sudo /etc/init.d/postgresql restart
