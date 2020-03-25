from unittest import TestCase

import geopandas as gpd
import timeit

from coord2vec.Noam_Adir.models.geom_graph_builder import GeomGraphBuilder
from shapely.geometry import Point

from coord2vec.Noam_Adir.pipeline.base_pipeline import extract_geographical_features
from coord2vec.Noam_Adir.pipeline.preprocess import load_data_from_pickel, generic_clean_col, \
    ALL_MANHATTAN_FILTER_FUNCS_LIST
from coord2vec.Noam_Adir.pipeline.utils import get_non_repeating_coords


class TestGeomGraphBuilder(TestCase):

    def setUp(self) -> None:
        # pickle_folder = "/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir/pipline/"
        pickle_folder = ""
        pickle_file_name = "manhattan_house_prices.pkl"
        manhattan_df = load_data_from_pickel(f"{pickle_folder}{pickle_file_name}", "lon", "lat")
        manhattan_df = generic_clean_col(manhattan_df, ALL_MANHATTAN_FILTER_FUNCS_LIST)
        cleaned_features = manhattan_df[['sold', 'priceSqft', 'numBedrooms', 'numBathrooms', 'sqft', 'coord']]
        all_features = extract_geographical_features(cleaned_features)
        self.all_features = get_non_repeating_coords(all_features)
        geoms = gpd.GeoSeries([Point(coord[0], coord[1]) for coord in self.all_features["coord"]])
        self.gb = GeomGraphBuilder(geometries=geoms)

    def test_manhattan_graph(self):
        print("start testing graph building")
        n_nodes = self.gb.graph.number_of_nodes()
        self.assertEqual(len(self.all_features), n_nodes)
        print("number of nodes/buildings is:", len(self.all_features))
        t1 = timeit.default_timer()

        self.gb.set_method("DT")
        self.gb.construct_vertices()
        n_dt_edges = self.gb.graph.number_of_edges()
        t2 = timeit.default_timer()
        print("manhattan dataset, number of edges with DT graph building:", n_dt_edges)
        print("DT construction time:", t2-t1)

        self.gb.set_method("RNG")
        self.gb.construct_vertices()
        n_rng_edges = self.gb.graph.number_of_edges()
        t3 = timeit.default_timer()
        print("manhattan dataset, number of edges with RNG graph building:", n_rng_edges)
        print("RNG construction time:", t3-t2)

        # theoretical boundries
        self.assertTrue((n_nodes - 1) <= n_rng_edges <= (3 * n_nodes - 6))
        self.assertGreaterEqual(n_dt_edges, n_rng_edges)

        # # very long method
        # dist = "10"
        # self.gb.set_method(f"fix_distance_{dist}")
        # self.gb.construct_vertices()
        # n_fix10_edges = self.gb.graph.number_of_edges()
        # t4 = timeit.default_timer()
        # print("manhattan dataset, number of edges with fix_distance_10+ graph building:", n_fix10_edges)
        # print(f"fix{dist} construction time:", t4 - t3)
