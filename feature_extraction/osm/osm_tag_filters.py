
# Built from http://download.geofabrik.de/osm-data-in-gis-formats-free.pdf

# Points




# Lines

# major roads
MOTORWAY = "highway='motorway'"
TRUNK = "highway='trunk'"
PRIMARY = "highway='primary'"

# smaller roads
RESIDENTIAL_ROAD = "highway='residential'"

# Polygons

SCHOOL = "amenity='school'"
PLACE_OF_WORSHIP = "religion='christian' OR religion='jewish' OR religion='muslim'" \
                    " OR religion='buddhist' OR religion='hindu' OR religion='taoist'" \
                    " OR religion='shintoist' OR religion='sikh'"
HOSPITAL = "amenity='hospital'"  # 'building=hostpital' is weird and should not be included
BUILDING = "building IS NOT NULL"
