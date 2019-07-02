#!/bin/bash
#docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
#docker volume rm openstreetmap-data
#docker volume create openstreetmap-data
sudo chmod 777 openstreetmap-tile-server/run.sh
docker build openstreetmap-tile-server/ -t osm-tile-server
docker run -v /data/home/zarecki/osm_data/israel-and-palestine-latest.osm.pbf:/data.osm.pbf -v openstreetmap-data:/var/lib/postgresql/10/main osm-tile-server import
docker run -e THREADS=24 -p 8080:80 -v openstreetmap-data:/var/lib/postgresql/10/main -d osm-tile-server run project_building_only.mml

#docker run -e THREADS=24 -p 8080:80 -v openstreetmap-data:/var/lib/postgresql/10/main -d osm-tile-server run project_road_only.mml
docker ps
