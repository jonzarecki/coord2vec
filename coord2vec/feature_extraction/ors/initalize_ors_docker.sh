#!/usr/bin/env bash

docker run --rm -d -p 8100:8080 --name ors-service \
        -v $ORS-FILE:/ors/code/data/osm_file.pbf giscience/openrouteservice:latest