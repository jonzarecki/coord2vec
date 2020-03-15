# Built from http://download.geofabrik.de/osm-data-in-gis-formats-free.pdf

OSM_POINT_TABLE = "planet_osm_point"
OSM_LINE_TABLE = "planet_osm_line"
OSM_POLYGON_TABLE = "planet_osm_polygon"
JUNCTIONS_TABLE = "ways_vertices_pgr"

# Points
JUNCTIONS = "1=1"  # JUNCTIONS_TABLE

# Lines
BOUNDARY = "boundary IS NOT NULL"

ROAD = 'highway IS NOT NULL'
# major roads
PRIMARY_ROAD = "highway='primary'"
SECONDARY_ROAD = "highway='secondary'"
TERTIARY_ROAD = "highway='tertiary'"
MAJOR_ROAD = f'{PRIMARY_ROAD} or {SECONDARY_ROAD} or {TERTIARY_ROAD}'
# minor roads
RESIDENTIAL_ROAD = "highway='residential'"
SERVICE_ROAD = "highway='service'"
TRACK_ROAD = "highway='track'"
PATH_ROAD = "highway='path'"
MINOR_ROAD = f'{RESIDENTIAL_ROAD} or {SERVICE_ROAD} or {TRACK_ROAD} or {PATH_ROAD}'

# Polygons
# SCHOOL = "amenity='school'"
# HOSPITAL = "amenity='hospital'"  # 'building=hostpital' is weird and should not be included
BUILDING = "building IS NOT NULL"
SHOP = "shop IS NOT NULL"
PARK = "leisure IS NOT NULL"
AMENITY = "amenity IS NOT NULL"

# multi-geometry
ALL = "1=1"

BUILDING_NAME = 'building'
MAJOR_ROAD_NAME = 'major_road'
MINOR_ROAD_NAME = 'minor_road'

