#!/bin/bash

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
    if [ ! -f /data.osm.pbf ]; then
        echo "WARNING: No import file at /data.osm.pbf, so importing Luxembourg as example..."
        wget -nv http://download.geofabrik.de/europe/luxembourg-latest.osm.pbf -O /data.osm.pbf
    fi

    # Import data
    sudo -u renderer osm2pgsql -d gis --create --slim -G --hstore --multi-geometry --tag-transform-script /home/renderer/src/openstreetmap-carto/openstreetmap-carto.lua -C 2048 --number-processes ${THREADS:-4} -S /home/renderer/src/openstreetmap-carto/openstreetmap-carto.style /data.osm.pbf

    exit 0
fi

if [ "$1" = "run" ]; then
    # Initialize mapnik style
    echo "$2"
    p=`pwd`
    cd /home/renderer/src/

    sudo -u renderer carto /home/renderer/src/openstreetmap-carto/"$2" > /home/renderer/src/openstreetmap-carto/mapnik.xml
    cd $p

    # Initialize PostgreSQL and Apache
    service postgresql start
    service apache2 restart

    # Configure renderd threads
    cd /home/renderer/
    mkdir /home/renderer/"$2"/
    cp /usr/local/etc/renderd.conf /home/renderer/"$2"/
#    sudo -u renderer carto /home/renderer/src/openstreetmap-carto/"$2" > /home/renderer/"$2"/mapnik.xml

    sed -i -E "s/num_threads=[0-9]+/num_threads=${THREADS:-4}/g" /home/renderer/"$2"/renderd.conf

    # Run
    sudo -u renderer renderd -f -c /home/renderer/"$2"/renderd.conf

    exit 0
fi

echo "invalid command"
exit 1