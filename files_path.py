import os

COORD2VEC_ABS_PATH = os.path.dirname(__file__)


def join_abs_path(rel_path):
    """
    concat COORD2VEC_ABS_PATH to rel_path
    Args:
        rel_path: path to a file inside the project that starts with coord2vec (ex: coord2vec/evaluation/...)

    Returns:
        abs_path
    """

    abs_path = os.path.join(COORD2VEC_ABS_PATH, rel_path)
    return abs_path


MANHATTAN_PKL_PATH = join_abs_path("coord2vec/Noam_Adir/manhattan/manhattan_house_prices.pkl")
