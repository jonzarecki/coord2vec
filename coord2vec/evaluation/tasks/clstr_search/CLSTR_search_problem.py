from functools import partial
from typing import Callable, List, Tuple, Union
from cachetools import cached, LRUCache

from geopandas import GeoSeries
from shapely.geometry import Polygon, MultiPolygon
from simpleai.search import SearchProblem
from simpleai.search.local import beam_best_first
from simpleai.search.models import SearchNode
from sklearn.base import BaseEstimator

from coord2vec.evaluation.tasks.clstr_search.building_clustering_problem import BuildingClusteringProblem, BuildingCluster
from coord2vec.feature_extraction.features_builders import FeaturesBuilder


class CLSTRSearchProblem(BuildingClusteringProblem):
    def __init__(self, poly_builder: FeaturesBuilder, heuristic: BaseEstimator, initial_state: BuildingCluster,
                 action_radius_m: float = 10., *args, **kwargs):
        super().__init__(action_radius_m=action_radius_m, initial_state=initial_state, *args, **kwargs)
        self.heuristic = heuristic
        self.poly_builder = poly_builder

    @staticmethod
    @cached(cache=LRUCache(maxsize=256), key=lambda c, b: str((frozenset(bc for bc in c), b)))
    def add_features_to_clusters(clusters: List[BuildingCluster], builder: FeaturesBuilder) -> List[BuildingCluster]:
        hulls_feat_df = builder.transform(GeoSeries([s.hull for s in clusters]))  # cache can slow us down

        feats_array = hulls_feat_df#.to_numpy()
        for i, bc in enumerate(clusters):
            bc._feature = feats_array.iloc[i:i+1, :]  # '_feature' attribute only used in CLSTRSearchProblem

        return clusters

    def actions(self, state: BuildingCluster) -> List[BuildingCluster]:
        """
           Returns the actions available to perform from `state`.
           The returned value is an iterable over actions.
           Actions are problem-specific and no assumption should be made about
           them.
        """
        actions = super(CLSTRSearchProblem, self).actions(state)

        actions = self.add_features_to_clusters(actions, self.poly_builder)

        return actions

    def value(self, state) -> float:
        # TODO: use one-class model / anomaly detection here
        if self.heuristic is None:
            pass
            # don't use a learned heuristic
        # remember not to allow CLSTRs that I know exists
        if '_feature' not in state.__dict__:
            state = self.add_features_to_clusters([self.initial_state], self.poly_builder)[0]
        return self.heuristic.score_samples(state._feature)[0]  # lower is more anomylus


def building_to_CLSTR(start_building: Union[Polygon, MultiPolygon], poly_builder: FeaturesBuilder, heuristic: BaseEstimator = None,
                   search_alg: Union[Callable[[SearchProblem], SearchNode], partial] = beam_best_first
                   ) -> Tuple[BuildingCluster, float]:
    """
    Runs a search algorithm starting from $start_building,
        The search problem heuristic is defined using $poly_builder and $heuristic
    Args:
        start_building: The building which is the initial state for the search problem
        poly_builder: A feature builder on which $heuristic is trained
        heuristic: A model used to learn how the CLSTR looks
        search_alg: A simpleAI local-search algorithm

    Returns:
        The resulting buildingCluster and it's heuristic score.
    """
    search_prob = CLSTRSearchProblem(poly_builder, heuristic, initial_state=BuildingCluster(start_building))
    # TODO: also possible to return neighbors ?

    init_state = BuildingCluster(start_building)
    # rand_states = [init_state] + [BuildingCluster(b) for b in
    #                               get_k_nearest_buildings(start_building, 10, start_building)]
    # rand_states = CLSTRSearchProblem.add_features_to_clusters(rand_states, poly_builder)
    search_prob.generate_random_state = lambda *a: init_state  # rand_states[random.randint(0, len(rand_states) - 1)]
    res = search_alg(search_prob)
    return res.state, res.value
