FROM ubuntu:19.04

# Based on
# https://switch2osm.org/manually-building-a-tile-server-18-04-lts/

# Install dependencies
RUN apt-get update
RUN apt-get install -y libboost-all-dev git-core tar unzip wget bzip2 build-essential autoconf libtool libxml2-dev libgeos-dev libgeos++-dev libpq-dev libbz2-dev libproj-dev munin-node munin libprotobuf-c-dev protobuf-c-compiler libfreetype6-dev libtiff5-dev libicu-dev libgdal-dev libcairo-dev libcairomm-1.0-dev libagg-dev liblua5.2-dev ttf-unifont lua5.1 liblua5.1-dev libgeotiff-epsg

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN adduser --disabled-password --gecos "" renderer
USER renderer

# Install latest osm2pgsql
RUN mkdir /home/renderer/src
WORKDIR /home/renderer/src
RUN git clone https://github.com/openstreetmap/osm2pgsql.git
WORKDIR /home/renderer/src/osm2pgsql
USER root
RUN apt-get install -y make cmake g++ libboost-dev libboost-system-dev libboost-filesystem-dev libexpat1-dev zlib1g-dev libbz2-dev libpq-dev libgeos-dev libgeos++-dev libproj-dev lua5.2 liblua5.2-dev
USER renderer
RUN mkdir build
WORKDIR /home/renderer/src/osm2pgsql/build
RUN cmake ..
RUN make
USER root
RUN make install
USER renderer

# Configure stylesheet (used in tag transformation)
WORKDIR /home/renderer/src
RUN git clone https://github.com/jonzarecki/openstreetmap-carto.git

# Install PostgreSQL
USER root
RUN apt-get install -y postgresql postgresql-contrib postgis postgresql-11-postgis-2.5
RUN apt-get install -y postgresql-11-pgrouting osm2pgrouting
USER renderer

# Change conf files to make postgres accessible from the outside and fast
USER root
RUN echo 'host    all             all             0.0.0.0/0            trust' >> /etc/postgresql/11/main/pg_hba.conf
RUN sed -i -e 's/localhost/*/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#listen_addresses/listen_addresses/g' /etc/postgresql/11/main/postgresql.conf

RUN sed -i -e 's/shared_buffers = 128MB/shared_buffers = 2GB/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#work_mem = 4MB/work_mem = 10GB/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#maintenance_work_mem = 64MB/maintenance_work_mem = 1GB/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#max_worker_processes = 8/max_worker_processes = 100/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#max_parallel_workers_per_gather = 2/max_parallel_workers_per_gather = 16/g' /etc/postgresql/11/main/postgresql.conf
RUN sed -i -e 's/#max_parallel_workers = 8/max_parallel_workers = 50/g' /etc/postgresql/11/main/postgresql.conf

# Start running
USER root
RUN apt-get install -y sudo systemd curl
COPY run.sh /
HEALTHCHECK --interval=10s --timeout=1s --start-period=70s --retries=10 CMD pg_isready && cat /done || exit 1

ENTRYPOINT ["/run.sh"]
CMD []