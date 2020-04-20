#!/bin/bash
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
docker volume rm openstreetmap-data
docker volume create openstreetmap-data
docker run -e THREADS=24 -v /media/yonatanz/yz/Research/coord2vec/osm/israel-and-palestine-latest.osm.pbf:/data.osm.pbf -v openstreetmap-data:/var/lib/postgresql/10/main overv/openstreetmap-tile-server import
docker run -e THREADS=24 -p 8080:80 -v openstreetmap-data:/var/lib/postgresql/10/main -d osm-tile-server run project_building_only.mml
#docker run -e THREADS=24 -p 8080:80 -v openstreetmap-data:/var/lib/postgresql/10/main -d osm-tile-server run project_road_only.mml
docker ps