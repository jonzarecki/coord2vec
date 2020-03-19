from unittest import TestCase

from coord2vec.Noam_Adir.models.geom_graph_builder import *
from shapely.geometry import Point


class TestGeomGraphBuilder(TestCase):

    def setUp(self) -> None:
        rect_points = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
        self.rect_gs = gpd.GeoSeries(rect_points)
        self.tri_gs = [Point(0, 0), Point(1, 0), Point(0, 1)]

    def test_add_geometries(self):
        gb_1 = GeomGraphBuilder(self.rect_gs)
        gb_2 = GeomGraphBuilder()
        gb_2.add_geometries(self.rect_gs)
        self.assertEqual(gb_1.graph.number_of_nodes(), 4)
        self.assertEqual(gb_2.graph.number_of_nodes(), 4)
        self.assertEqual(gb_1.graph.number_of_edges(), 0)
        self.assertEqual(gb_2.graph.number_of_edges(), 0)

    def test_construct_vertices(self):
        methods_name = ["RNG", "DT", "fix_distance_1m", "fix_distance_1.5m", "fix_distance_1.5", "fix_distanc_1.5"]
        expected_numbers_of_edges = [4, 5, 4, 6, 6, 4]
        for method_name, expected_number_of_edges in zip(methods_name, expected_numbers_of_edges):
            gb = GeomGraphBuilder(self.rect_gs, method=method_name)
            gb.construct_vertices()
            self.assertEqual(gb.graph.number_of_edges(), expected_number_of_edges)


    def test_construct_vertices_rng(self):
        gb_rect = GeomGraphBuilder(self.rect_gs)
        gb_rect.construct_vertices_RNG()
        self.assertEqual(gb_rect.graph.number_of_nodes(), 4)
        self.assertEqual(gb_rect.graph.number_of_edges(), 4)

        gb_tri = GeomGraphBuilder(self.tri_gs)
        gb_tri.construct_vertices_RNG()
        self.assertEqual(gb_tri.graph.number_of_nodes(), 3)
        self.assertEqual(gb_tri.graph.number_of_edges(), 2)


    def test_construct_vertices_dt(self):
        gb = GeomGraphBuilder(self.rect_gs)
        gb.construct_vertices_DT()
        self.assertEqual(gb.graph.number_of_nodes(), 4)
        self.assertEqual(gb.graph.number_of_edges(), 5)

        gb_tri = GeomGraphBuilder(self.tri_gs)
        gb_tri.construct_vertices_DT()
        self.assertEqual(gb_tri.graph.number_of_nodes(), 3)
        self.assertEqual(gb_tri.graph.number_of_edges(), 3)

    def test_construct_vertices_fix_distance(self):
        gb = GeomGraphBuilder(self.rect_gs)
        gb.construct_vertices_fix_distance(1)
        self.assertEqual(gb.graph.number_of_edges(), 4)

        gb2 = GeomGraphBuilder(self.rect_gs)
        gb2.construct_vertices_fix_distance(1.5)
        self.assertEqual(gb2.graph.number_of_edges(), 6)

        gb_tri = GeomGraphBuilder(self.tri_gs)
        gb_tri.construct_vertices_fix_distance(1)
        self.assertEqual(gb_tri.graph.number_of_nodes(), 3)
        self.assertEqual(gb_tri.graph.number_of_edges(), 2)
