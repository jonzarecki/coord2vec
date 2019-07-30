#!/usr/bin/env bash

p=`pwd`
cd "${0%/*}/"
echo `pwd`

docker stop $(docker ps -aq --filter ancestor=osm-tile-server) && docker rm $(docker ps -aq --filter ancestor=osm-tile-server)
sudo chmod 777 openstreetmap-tile-server/run.sh
docker build openstreetmap-tile-server/ -t osm-tile-server

# Download Israel as sample if no data is provided
if [[ ! -f ./data.osm.pbf ]]; then
    echo "WARNING: No import file at /data.osm.pbf, so importing Israel as example..."
    wget https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf -O ./data.osm.pbf
    #    wget https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf -O ./israel-and-palestine.osm.pbf
    wget http://download.geofabrik.de/asia/china-latest.osm.pbf -O ./china.osm.pbf
fi

cdir=`pwd`
#docker run -e THREADS=24 -p 127.0.0.1:8080:80 -v $cdir/data.osm.pbf:/data.osm.pbf \
#            -v openstreetmap-data-tile:/var/lib/postgresql/10/main -td osm-tile-server import

if docker volume ls | grep -qw osm-data-tile; then
    echo "Postgres already rendered, if not, remove volume osm-data-tile"
else
#    echo blas
    docker volume rm osm-data-tile
    docker volume create osm-data-tile

    docker run -v $cdir/data.osm.pbf:/data.osm.pbf -v osm-data-tile:/var/lib/postgresql/10/main \
                    osm-tile-server import &

    wait

fi


docker run -e THREADS=24 -p 8101:80 -v osm-data-tile:/var/lib/postgresql/10/main \
                    -d osm-tile-server run project_building_only.mml
echo "starting building tile server"

sleep 15

docker run -e THREADS=24 -p 8102:80 -v osm-data-tile:/var/lib/postgresql/10/main \
                    -d osm-tile-server run project_road_only.mml
echo "starting road tile server"

sleep 15

docker run -e THREADS=24 -p 8103:80 -v osm-data-tile:/var/lib/postgresql/10/main \
                    -d osm-tile-server run project_landcover_only.mml
echo "starting landcover tile server"

sleep 15

wait

echo "done building tile servers"
#docker run -e THREADS=24 -p 8081:80 -v osm-data-tile-road:/var/lib/postgresql/10/main \
#                    -d osm-tile-server run project_landcover_only.mml

cd $p