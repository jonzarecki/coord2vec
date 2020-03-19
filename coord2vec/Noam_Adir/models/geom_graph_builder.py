from typing import List, Tuple

import networkx as nx
import geopandas as gpd
import re
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import Delaunay


class GeomGraphBuilder:
    default_method = "RNG"

    def __init__(self, geometries=gpd.GeoSeries(), method=default_method):
        self.method = method
        self.graph = nx.Graph()
        self.geom_ids_dict = {}

        self.add_geometries(geometries)

    def add_geometry(self, geom: BaseGeometry):
        cur_id = self.graph.number_of_nodes()
        self.geom_ids_dict[cur_id] = geom
        self.graph.add_node(cur_id, geometry=geom)

    def add_geometries(self, geoms: gpd.GeoSeries):
        for geom in geoms:
            self.add_geometry(geom)

    def set_method(self, method: str):
        self.method = method

    def construct_vertices(self):
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
        self.construct_vertices_DT()
        for u, v in self.graph.edges:
            for n in self.graph.nodes:
                u_geom, v_geom, n_geom = tuple([self.graph.nodes[x]["geometry"] for x in [u, v, n]])
                if n != u and n != v and \
                        u_geom.distance(v_geom) > min(n_geom.distance(u_geom), n_geom.distance(v_geom)):
                    self.graph.remove_edge(u, v)
                    break

    def construct_vertices_DT(self):
        cur_nodes = [n for n, geom in self.graph.nodes(data="geometry")]
        cur_geoms = [geom for n, geom in self.graph.nodes(data="geometry")]
        points = self.transform_geometries_to_palanar_points(cur_geoms)
        delaunay_tri = Delaunay(points)
        indices, indptr = delaunay_tri.vertex_neighbor_vertices
        for i, n in enumerate(cur_nodes):
            neighbor_vertices_inds = indptr[indices[i]:indices[i + 1]]
            for point_ind in neighbor_vertices_inds:
                self.graph.add_edge(n, cur_nodes[point_ind])

    def construct_vertices_fix_distance(self, radius: float):
        for n_i, geom_i in list(self.graph.nodes(data="geometry")):
            for n_j, geom_j in list(self.graph.nodes(data="geometry")):
                if n_i > n_j:
                    distance = geom_i.distance(geom_j)
                    if distance <= radius:
                        self.graph.add_edge(n_i, n_j, distance=distance)

    def transform_geometries_to_palanar_points(self, geometries: List[BaseGeometry]) -> List[Tuple[float, float]]:
        points2d = []
        for geom in geometries:
            assert isinstance(geom, (Point, LineString, Polygon)), "geometries types supported:" \
                                                                   " Point, LineString, Polygon"
            x, y = geom.centroid.x, geom.centroid.y
            points2d.append((x, y))
        return points2d
