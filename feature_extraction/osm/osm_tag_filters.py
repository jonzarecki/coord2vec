
# Built from http://download.geofabrik.de/osm-data-in-gis-formats-free.pdf

SCHOOL = "amenity='school'"
PLACE_OF_WORSHIP = "religion='christian' OR religion='jewish' OR religion='muslim'" \
                    " OR religion='buddhist' OR religion='hindu' OR religion='taoist'" \
                    " OR religion='shintoist' OR religion='sikh'"




# ROADS

# major roads
MOTORWAY = [('highway', 'motorway')]
TRUNK = [('highway', 'trunk')]
PRIMARY = [('highway', 'primary')]



# BUILDING
BUILDING = "building IS NOT NULL"
