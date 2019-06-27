#!/usr/bin/env bash

# assumes there exists a osm.pdb named data.osm.pdb in this folder

# TODO: can move to docker
# Install PostgreSQL and osm2pgsql
sudo apt-get install -y postgresql postgresql-contrib postgis postgresql-10-postgis-2.4
sudo apt-get install -y osm2pgsql


# Initialize PostgreSQL
service postgresql start
sudo createuser renderer
sudo createdb -E UTF8 -O renderer gis
sudo psql -d gis -c "CREATE EXTENSION postgis;"
sudo psql -d gis -c "CREATE EXTENSION hstore;"
#sudo psql -d gis -c "ALTER TABLE geometry_columns OWNER TO renderer;"
#sudo psql -d gis -c "ALTER TABLE spatial_ref_sys OWNER TO renderer;"

# Download Israel as sample if no data is provided
if [ ! -f /data.osm.pbf ]; then
    echo "WARNING: No import file at /data.osm.pbf, so importing Israel as example..."
    wget -nv https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf -O ./data.osm.pbf
fi

# Import data
sudo osm2pgsql -d gis --create --slim -G --hstore --multi-geometry --tag-transform-script /home/renderer/src/openstreetmap-carto/openstreetmap-carto.lua -C 2048 --number-processes ${THREADS:-4} -S /home/renderer/src/openstreetmap-carto/openstreetmap-carto.style /data.osm.pbf
