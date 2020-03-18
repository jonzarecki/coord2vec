import networkx as nx
import geopandas as gpd
import re


class GeomGraphBuilder:

    defult_method = "RNG"

    def __init__(self, geometries: gpd.GeoSeries, method=defult_method):
        self.method = method
        self.graph = nx.Graph(geometries.centroid)

    def add_node(self, node):
        pass

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def construct_vertices(self):
        fix_match = re.match(r"fix_distance_(\d)+m?", self.method)
        if self.method == "RNG":
            # relative nighbohood graph
            self.construct_vertices_RNG()
        elif self.method == "DT":
            # Delaunay triangulation
            self.construct_vertices_DT()
        elif fix_match is not None:
            self.construct_vertices_fix_distance()
        else:
            # TODO change to log
            print("method mast be: RNG, DT, fix_distance")
            print(f"defult method is {GeomGraphBuilder.defult_method}")

    def construct_vertices_RNG(self):
        pass

    def construct_vertices_DT(self):
        pass

    def construct_vertices_fix_distance(self):
        pass
