from coord2vec.image_extraction.tile_utils import build_tile_extent

from coord2vec.image_extraction.tile_image import render_single_tile, StaticMap



if __name__ == '__main__':
    m = StaticMap(500, 500, url_template='http://52.232.47.43:8080/tile/{z}/{x}/{y}.png')
    center = [34.7855, 32.1070]

    image = render_single_tile(m, build_tile_extent(center))

    image.show("Rendered tile")