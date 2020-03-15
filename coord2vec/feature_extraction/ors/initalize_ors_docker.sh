#sed -i -e 's/visible="false"//g' data_orig.osm
#sed -i -e 's/visible="true"//g' data_orig.osm
#python3 ./convert_xml_ids.py data_orig.osm data.osm
#./osmconvert64 data.osm --drop-author --drop-version --out-pbf -o=osm_file.pbf

docker run --rm -d -p 8100:8080 --name ors-service
    -v ~/repositories/coord2vec_in/osm_file.pbf:/ors-core/data/osm_file.pbf giscience/openrouteservice:latest


# docker run --rm -d -p EXTERNAL_PORT:8080 --name ors-service \
#             -v OSM_FILE:/ors-core/data/osm_file.pbf giscience/openrouteservice:latest