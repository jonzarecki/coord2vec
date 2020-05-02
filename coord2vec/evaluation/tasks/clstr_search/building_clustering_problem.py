import random
from abc import abstractmethod, ABC
from typing import List, Union

from cachetools import cached, LRUCache
from geopandas import GeoSeries
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
from simpleai.search import SearchProblem


from coord2vec.feature_extraction.osm.osm_utils import get_buildings_from_polygon, get_buildings_in_radius


class BuildingCluster:
    buildings: GeoSeries
    hull: Polygon

    def __init__(self, start_buildings: Union[GeoSeries, Polygon, MultiPolygon] = None):
        # start_buildings can be either a building or a geo-series of buildings
        if isinstance(start_buildings, Polygon) or isinstance(start_buildings, MultiPolygon):
            self.hull = start_buildings.convex_hull
            self.buildings = GeoSeries([start_buildings])
        elif isinstance(start_buildings, GeoSeries):
            self.hull = cascaded_union(start_buildings).convex_hull
            self.buildings = start_buildings
        else:
            raise AssertionError(f"start_building type {type(start_buildings)} is illegal")

        self.wkt = self.hull.wkt  # can be called multiple-times, expensive

    def __eq__(self, other):
        if isinstance(other, BuildingCluster):
            return self.wkt == other.wkt
        return False

    def __hash__(self):
        return hash(self.wkt)

    def __str__(self):
        return self.wkt


def add_building_copy(building_cluster: BuildingCluster, building_to_add: Polygon) -> BuildingCluster:
    new_buildings_gs = building_cluster.buildings.append(GeoSeries([building_to_add]), ignore_index=True)
    new_buildings_gs = GeoSeries([wkt.loads(w) for w in new_buildings_gs.apply(lambda p: p.wkt).unique()])
    return BuildingCluster(start_buildings=new_buildings_gs)


def remove_building_copy(building_cluster: BuildingCluster, building_to_remove: Polygon) -> BuildingCluster:
    old_wkt = building_to_remove.wkt
    old_buildings_gs = GeoSeries([wkt.loads(w) for w in building_cluster.buildings.apply(lambda p: p.wkt) if w != old_wkt])
    return BuildingCluster(start_buildings=old_buildings_gs)


class BuildingClusteringProblem(ABC, SearchProblem):
    def __init__(self, initial_state: BuildingCluster = None, action_radius_m: float = 50., *args, **kwargs):
        super(BuildingClusteringProblem, self).__init__(initial_state)
        self.action_radius_m = action_radius_m

    @cached(cache=LRUCache(maxsize=256), key=lambda s, bc: hash(bc))
    def actions(self, state: BuildingCluster) -> List[BuildingCluster]:
        """
           Returns the actions available to perform from `state`.
           The returned value is an iterable over actions.
           Actions are problem-specific and no assumption should be made about
           them.
        """
        new_states = [add_building_copy(state, new_b) for new_b in get_buildings_in_radius(state.hull, self.action_radius_m, state.hull)]
        if len(state.buildings) > 1:
            new_states = new_states + [remove_building_copy(state, old_b) for old_b in state.buildings]

        new_states = list(set(new_states))  # only unique
        random.shuffle(new_states)

        return new_states

    def result(self, state: BuildingCluster, action: BuildingCluster) -> BuildingCluster:
        """Returns the resulting state of applying `action` to `state`.
            In this case as action is a BuildingCluster, it just returns it (the action was already applied)
        """
        return action

    @abstractmethod
    def value(self, state: BuildingCluster) -> float:
        """Returns the value of `state` as it is needed by optimization
           problems.
           Value is a number (integer or floating point)."""
        pass

    # generate_random_state ? how do we want to search ? just one from every building ?
