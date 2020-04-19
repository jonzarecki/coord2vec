from typing import List, Tuple

import networkx as nx
import geopandas as gpd
import re
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import Delaunay
from tqdm import tqdm


class GeomGraphBuilder:
    """
    A class that allow building of graph around GeoSeries
    """
    default_method = "DT"

    def __init__(self, geometries=gpd.GeoSeries(), method=default_method):
        """
        A constructor that allows empty construction with no geometries
        Args:
            geometries: an optional GeoSeries to add as nodes
            method: the name of the method in the form of ether:
                    1. "RNG" - for relative neighborhood graph
                    2. "DT" - for Delaunay triangulation
                    3. "fix_distance_(number)" - to build edge if distance is less then (number)
                    default value is DT
        """
        self.method = method
        self.graph = nx.Graph()
        self.geom_ids_dict = {}

        self.add_geometries(geometries)

    def add_geometry(self, geom: BaseGeometry):
        """
        A method to add a geometry as a node in the graph
        Args:
            geom: the geometry to add

        """
        cur_id = self.graph.number_of_nodes()
        self.geom_ids_dict[cur_id] = geom
        self.graph.add_node(cur_id, geometry=geom)

    def add_geometries(self, geoms: gpd.GeoSeries):
        """
        A method to add series of geometries as nodes to the graph
        Args:
            geoms: GeoSeries of geometries to add as nodes to the graph

        """
        for geom in geoms:
            self.add_geometry(geom)

    def set_method(self, method: str):
        """
        seter method for method of building the graph
        Args:
            method: the name of the method in the form of ethier:
                    1. "RNG" - for relative neighborhood graph
                    2. "DT" - for Delaunay triangulation
                    3. "fix_distance_(number)" - to build edge if distance is less then (number)

        """
        self.method = method

    def construct_vertices(self):
        """
        construct verteces acoording to self.method and the current nodes added
        """
        fix_match = re.match(r"fix_distance_(\d+\.?\d*)m?", self.method)
        if self.method == "RNG":
            # relative nighbohood graph
            self.construct_vertices_RNG()
        elif self.method == "DT":
            # Delaunay triangulation
            self.construct_vertices_DT()
        elif fix_match is not None:
            self.construct_vertices_fix_distance(float(fix_match.group(1)))
        else:
            # TODO change to log
            print("method mast be: RNG, DT, fix_distance")
            print(f"defult method is {GeomGraphBuilder.default_method}")
            self.method = GeomGraphBuilder.default_method
            self.construct_vertices()

    def construct_vertices_RNG(self):
        """
        compute Delaunay triangulation and remove unnecessary edges
        """
        self.construct_vertices_DT()
        for u, v in tqdm(self.graph.edges):
            for n in self.graph.nodes:
                u_geom, v_geom, n_geom = tuple([self.graph.nodes[x]["geometry"] for x in [u, v, n]])
                if n != u and n != v and \
                        u_geom.distance(v_geom) > n_geom.distance(u_geom) and \
                        u_geom.distance(v_geom) > n_geom.distance(v_geom):
                    self.graph.remove_edge(u, v)
                    break

    def construct_vertices_DT(self):
        """
        uses Delaunay triangulation from scipy to cumpute edges for the nodes
        """
        self.graph = nx.create_empty_copy(self.graph)
        cur_nodes = [n for n, geom in self.graph.nodes(data="geometry")]
        cur_geoms = [geom for n, geom in self.graph.nodes(data="geometry")]
        points = self.transform_geometries_to_palanar_points(cur_geoms)
        delaunay_tri = Delaunay(points)
        indices, indptr = delaunay_tri.vertex_neighbor_vertices
        for i, n in tqdm(enumerate(cur_nodes)):
            neighbor_vertices_inds = indptr[indices[i]:indices[i + 1]]
            for point_ind in neighbor_vertices_inds:
                self.graph.add_edge(n, cur_nodes[point_ind])

    def construct_vertices_fix_distance(self, radius: float):
        """
        construct all edges for couple of nodes closer then a given radius
        Args:
            radius: the max distance between to nodes geometries for building an edge
        """
        self.graph = nx.create_empty_copy(self.graph)
        for n_i, geom_i in tqdm(list(self.graph.nodes(data="geometry"))):
            for n_j, geom_j in list(self.graph.nodes(data="geometry")):
                if n_i > n_j:
                    distance = geom_i.distance(geom_j)
                    if distance <= radius:
                        # self.graph.add_edge(n_i, n_j, distance=distance)
                        self.graph.add_edge(n_i, n_j)

    def get_adj_as_scipy_sparse_matrix(self):
        return nx.to_scipy_sparse_matrix(self.graph)

    @staticmethod
    def transform_geometries_to_palanar_points(geometries: List[BaseGeometry]) -> List[Tuple[float, float]]:
        """
        transforms a list of BaseEstimator of types Point, LineString or Polygon
                to list of tuple of coordinates of their centroids
        Args:
            geometries: list of geometries to transform

        Returns: list of coordinates of the geometries centroids (geom.centroid.x, geom.centroid.y)

        """
        points2d = []
        for geom in geometries:
            assert isinstance(geom, (Point, LineString, Polygon)), "geometries types supported:" \
                                                                   " Point, LineString, Polygon"
            x, y = geom.centroid.x, geom.centroid.y
            points2d.append((x, y))
        return points2d
