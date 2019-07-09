from coord2vec.image_extraction.tile_utils import sample_coordinate_in_range

israel_range = [29.593, 34.085, 32.857, 34.958]

if __name__ == '__main__':
    SAMPLE_NUM = 50_000
    sampled_coords = {}
    for _ in range(SAMPLE_NUM):
        sample_coordinate_in_range()