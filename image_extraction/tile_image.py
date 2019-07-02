import staticmap
from PIL import Image
from staticmap import StaticMap, Polygon, CircleMarker
from staticmap.staticmap import _lon_to_x, _lat_to_y




def render_single_tile(m: StaticMap, ext: list):
    ex_poly = [[ext[0], ext[1]],
               [ext[0], ext[3]],
               [ext[2], ext[1]],
               [ext[2], ext[3]]]
    polygon = Polygon(ex_poly, 'white', 'white', True)
    m.add_polygon(polygon)
    m.zoom = m._calculate_zoom()

    # get extent of all lines
    extent = ext

    # m.determine_extent(zoom=m.zoom)

    # calculate center point of map
    lon_center, lat_center = (extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2
    m.x_center = _lon_to_x(lon_center, m.zoom)
    m.y_center = _lat_to_y(lat_center, m.zoom)

    image = Image.new('RGB', (m.width, m.height), m.background_color)

    m._draw_base_layer(image)
    m.polygons.remove(polygon)
    m._draw_features(image)
    return image

m = StaticMap(500, 500, url_template='http://52.232.47.43:8080/tile/{z}/{x}/{y}.png')
center = [34.7855, 32.1070]

ext = [34.7855, 32.1070, 34.7855+0.001, 32.1070+0.001]
# ex_poly = [[ext[0], ext[1]],
#                 [ext[0], ext[3]],
#                 [ext[2], ext[1]],
#                 [ext[2], ext[3]]]
# polygon = Polygon(ex_poly, 'white', 'white', True)
# m.add_polygon(polygon)
# image = m.render()


image = render_single_tile(m, ext)
# marker_outline = CircleMarker((34.7855, 32.1070), 'white', 18)
# marker = CircleMarker((34.7855, 32.1070), '#0036FF', 12)
#
# m.add_marker(marker_outline)
# m.add_marker(marker)

# image = m.render(zoom=15)
image.save('marker2.png')
# image.save('bla3.png')

image.show()