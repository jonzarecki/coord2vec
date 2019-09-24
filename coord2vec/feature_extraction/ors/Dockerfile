FROM openrouteservice/openrouteservice:latest

ARG OSM_FILE=data.osm.pbf

# Copy osm data file
COPY data.osm.pbf /ors-core/data/osm_file.pbf

# Start the container
EXPOSE 8080
CMD /usr/local/tomcat/bin/catalina.sh run