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
fi

cdir=`pwd`
#docker run -e THREADS=24 -p 127.0.0.1:8080:80 -v $cdir/data.osm.pbf:/data.osm.pbf \
#            -v openstreetmap-data-tile:/var/lib/postgresql/10/main -td osm-tile-server import

docker volume rm osm-data-tile-building
docker volume create osm-data-tile-building

docker volume rm osm-data-tile-road
docker volume create osm-data-tile-road

docker run -v $cdir/data.osm.pbf:/data.osm.pbf -v osm-data-tile-building:/var/lib/postgresql/10/main \
                osm-tile-server import &

docker run -v $cdir/data.osm.pbf:/data.osm.pbf -v osm-data-tile-road:/var/lib/postgresql/10/main \
                osm-tile-server import &

wait

docker run -e THREADS=24 -p 8080:80 -v osm-data-tile-building:/var/lib/postgresql/10/main \
                    -d osm-tile-server run project_building_only.mml &


docker run -e THREADS=24 -p 8081:80 -v osm-data-tile-road:/var/lib/postgresql/10/main \
                    -d osm-tile-server run project_road_only.mml &

wait


#docker run -e THREADS=24 -p 8081:80 -v osm-data-tile-road:/var/lib/postgresql/10/main \
#                    -d osm-tile-server run project_landcover_only.mml

cd $p