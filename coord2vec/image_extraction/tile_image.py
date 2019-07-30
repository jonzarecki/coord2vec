from typing import List
import numpy as np
from PIL import Image
from staticmap import StaticMap, Polygon
from staticmap.staticmap import _lon_to_x, _lat_to_y

from coord2vec import config
from coord2vec.config import IMG_WIDTH, IMG_HEIGHT


def render_single_tile(m: StaticMap, ext: list) -> Image:
    """
    Revised version of StaticMap.render() allows explicit extent + empty maps
    Args:
        m: Static map object for rendering
        ext: the map extent in  (min_lon, min_lat, max_lon, max_lat)

    Returns:
        The returned RGB image as an PIL.Image
    """
    ex_poly = [[ext[0], ext[1]],
               [ext[0], ext[3]],
               [ext[2], ext[1]],
               [ext[2], ext[3]]]
    polygon = Polygon(ex_poly, 'white', 'white', True)
    m.add_polygon(polygon)
    m.zoom = m._calculate_zoom()

    # get extent of all lines
    extent = ext

    # calculate center point of map
    lat_center, lon_center = (extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2
    m.x_center = _lon_to_x(lon_center, m.zoom)
    m.y_center = _lat_to_y(lat_center, m.zoom)

    image = Image.new('RGB', (m.width, m.height), m.background_color)

    m._draw_base_layer(image)
    m.polygons.remove(polygon)
    m._draw_features(image)
    return image


def generate_static_maps(url_port_template: str, ports: List[int]) -> List[StaticMap]:
    """
    Small utility function for generating multiple StaticMaps
    Args:
        url_port_template: The url_template but with an additional parameter {p} for the port
        ports: List of port numbers for the StaticMap objects

    Returns:
        List of StaticMaps object initialized with the correct url_templates
    """
    return [StaticMap(IMG_WIDTH, IMG_HEIGHT, url_template=url_port_template.replace('{p}', str(p)),
                      delay_between_retries=15, tile_request_timeout=5)
            for p in ports]


def render_multi_channel(static_maps: List[StaticMap], ext: list) -> np.array:
    """
    Renders a multi-channel image with the list of StaticMaps, each of them is converted into grayscale and
    concatenated into the array
    Args:
        static_maps: List of StaticMap objects
        ext: The extent of the image

    Returns:
        numpy array where each channel is a greyscale
    """
    multi_arr = np.zeros((len(static_maps), IMG_HEIGHT, IMG_WIDTH,))

    for i, m in enumerate(static_maps):
        im_arr = np.array(render_single_tile(m, ext).convert('L'))
        multi_arr[i, :, :] = im_arr

    return multi_arr


if __name__ == '__main__':
    m = StaticMap(IMG_WIDTH, IMG_HEIGHT, url_template=config.tile_server_dns_noport.replace('{p}', '8080'))
    center = [34.7805, 32.1170]
    s = generate_static_maps(config.tile_server_dns_noport, config.tile_server_ports)
    ext = [34.7855, 32.1070, 34.7855 + 0.001, 32.1070 + 0.001]

    image = render_single_tile(m, ext)
    image_multi = render_multi_channel(s, ext)

    image.save('marker2.png')
    Image.fromarray(image_multi[1], mode='L').save('aa.png')
    image.show()
