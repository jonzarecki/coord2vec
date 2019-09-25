#!/bin/sh
sudo rm /done

if [ "$1" = "import" ]; then
    # Initialize PostgreSQL
    service postgresql start
    sudo -u postgres createuser renderer --no-password
    sudo -u postgres createdb -E UTF8 -O renderer gis
    sudo -u postgres psql -d gis -c "CREATE EXTENSION postgis;"
    sudo -u postgres psql -d gis -c "CREATE EXTENSION hstore;"
    sudo -u postgres psql -d gis -c "ALTER TABLE geometry_columns OWNER TO renderer;"
    sudo -u postgres psql -d gis -c "ALTER TABLE spatial_ref_sys OWNER TO renderer;"

    # Download Luxembourg as sample if no data is provided
    if [[ ! -f /data.osm.pbf ]]; then
        echo "WARNING: No import file at /data.osm.pbf, so importing Luxembourg as example..."
        wget -nv http://download.geofabrik.de/europe/luxembourg-latest.osm.pbf -O /data.osm.pbf
    fi

    # Import data
    sudo -u renderer osm2pgsql -d gis -l --create -C 3072 -G --hstore --multi-geometry --tag-transform-script /home/renderer/src/openstreetmap-carto/openstreetmap-carto.lua --number-processes 16 -S /home/renderer/src/openstreetmap-carto/openstreetmap-carto.style /data.osm.pbf
    service postgresql start

    sudo sleep 15 && touch /done &
    echo 'done'
    tail -f /dev/null  # keeps the docker running after it's finished
fi

