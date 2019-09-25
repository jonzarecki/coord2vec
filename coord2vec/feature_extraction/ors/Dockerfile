FROM openrouteservice/openrouteservice:latest

ARG OSM_FILE=data.osm.pbf

# Copy osm data file
COPY data.osm.pbf /ors-core/data/osm_file.pbf

HEALTHCHECK --interval=10s --timeout=5s --start-period=70s --retries=120 CMD curl -f http://localhost:8080 || exit 1

# Start the container
EXPOSE 8080
CMD /usr/local/tomcat/bin/catalina.sh run