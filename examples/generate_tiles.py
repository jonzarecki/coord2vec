from coord2vec.image_extraction.tile_image import render_single_tile, StaticMap


def build_extent(center_coord):
    return [center_coord[0], center_coord[1], center_coord[0] + 0.001, center_coord[1] + 0.001]


if __name__ == '__main__':
    m = StaticMap(500, 500, url_template='http://52.232.47.43:8080/tile/{z}/{x}/{y}.png')
    center = [34.7855, 32.1070]

    image = render_single_tile(m, build_extent(center))

    image.show("Rendered tile")