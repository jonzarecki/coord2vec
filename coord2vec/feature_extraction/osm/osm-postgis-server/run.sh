#!/bin/sh
sudo rm /done

echo 1
if [ "$1" = "import" ]; then
    # Initialize PostgreSQL
    service postgresql start
    sudo -u postgres createuser renderer
    sudo -u postgres createdb -E UTF8 -O renderer gis
    sudo -u postgres psql -d gis -c "CREATE EXTENSION postgis;"
    sudo -u postgres psql -d gis -c "CREATE EXTENSION hstore;"
    sudo -u postgres psql -d gis -c "CREATE EXTENSION pgRouting;"
    sudo -u postgres psql -d gis -c "ALTER TABLE geometry_columns OWNER TO renderer;"
    sudo -u postgres psql -d gis -c "ALTER TABLE spatial_ref_sys OWNER TO renderer;"

    # If no data was provided
    if [[ ! -f /data.osm.pbf ]]; then
        echo "ERROR: No import file at /data.osm.pbf"
        exit 1
    fi

    # Import data
    sudo -u renderer osm2pgsql -d gis -l --create -C 3072 -G --hstore --multi-geometry --tag-transform-script /home/renderer/src/openstreetmap-carto/openstreetmap-carto.lua --number-processes 16 -S /home/renderer/src/openstreetmap-carto/openstreetmap-carto.style /data.osm.pbf
    service postgresql start

    sudo sleep 10 && touch /done &
    echo 'done'
    tail -f /dev/null  # keeps the docker running after it's finished
fi
