from functools import partial
from unittest import TestCase
from unittest.mock import Mock

from geopandas import GeoSeries
from simpleai.search.local import hill_climbing, beam
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.svm import OneClassSVM

from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.config import POLYGON
from coord2vec.evaluation.tasks.clstr_search.building_clustering_problem import BuildingCluster
from coord2vec.feature_extraction.features.other_features.area_of_self import AreaOfSelf
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.feature_extraction.osm.osm_utils import get_buildings_from_polygon, get_buildings_in_radius



class _ClosestToSpecificArea(BaseEstimator):
    def score_samples(self, feats: pd.DataFrame):
        return [1. / abs(float(f[0]) - 823.1914) for f in feats.values]  # closer to area is larger


class _LargestArea(BaseEstimator):
    def score_samples(self, feats: pd.DataFrame):
        return [f[0] for f in feats.values]  # larger is better


class TestCLSTRAreaProblem(TestCase):

    def get_setup_variables(self):
        init_buildings = get_buildings_from_polygon(POLYGON, is_intersects=True)[:100]
        radius = 10
        poly_feat_builder = FeaturesBuilder([
            AreaOfSelf(),
        ])

        neighb_init_states = [BuildingCluster(init_buildings[1])] + [BuildingCluster(b) for b in
                                                                               get_buildings_in_radius(
                                                                                   init_buildings[1],
                                                                                   radius, init_buildings[1])]
        return neighb_init_states, poly_feat_builder

    def test_building_to_CLSTR_returns_larger(self):
        neighb_init_states, poly_feat_builder = self.get_setup_variables()
        res_bc, res_val = building_to_CLSTR(neighb_init_states[0].hull, poly_feat_builder, _LargestArea(),
                                         partial(hill_climbing, iterations_limit=3))
        self.assertIsInstance(res_bc, BuildingCluster)
        self.assertIsInstance(res_val, float)
        print(res_val)
        self.assertGreater(len(res_bc.buildings), len(neighb_init_states[0].buildings))

    def test_building_to_CLSTR_multiple_pipe_runs_and_returns_same(self):
        # TODO: weird, if this runs second weird errors will arise. "pyproj database disk image is malformed"
        # also when runs in parallel with pytest sometimes fail
        neighb_init_states, poly_feat_builder = self.get_setup_variables()

        res = parmap(lambda b: building_to_CLSTR(b.hull, poly_feat_builder, _ClosestToSpecificArea(),
                                              # partial(hill_climbing, iterations_limit=2)),
                                              partial(beam, beam_size=50, iterations_limit=3)),
                     neighb_init_states)  # , nprocs=1
        print(*[r[1] for r in res])

        self.assertSetEqual({r[1] for r in res}, {res[0][1]})
        for init_s, (bc, v) in zip(neighb_init_states, res):
            self.assertIsInstance(bc, BuildingCluster)
            self.assertIsInstance(v, float)

    def see_if_model_runs(self, model, neighb_init_states, poly_feat_builder):
        model.fit(poly_feat_builder.transform(GeoSeries([s.hull for s in neighb_init_states]),
                                                   use_cache=False))

        res_bc, res_val = building_to_CLSTR(neighb_init_states[0].hull, poly_feat_builder, model,
                                         partial(hill_climbing, iterations_limit=2))
        self.assertIsInstance(res_bc, BuildingCluster)
        self.assertIsInstance(res_val, float)

    def test_building_to_CLSTR_runs_with_one_class_model(self):
        neighb_init_states, poly_feat_builder = self.get_setup_variables()

        oneclass_model = OneClassSVM()
        self.see_if_model_runs(oneclass_model, neighb_init_states, poly_feat_builder)

    def test_building_to_CLSTR_runs_with_baseline_model(self):
        neighb_init_states, _ = self.get_setup_variables()

        poly_feat_builder = Mock()
        import numpy as np
        poly_feat_builder.transform = lambda polys, *args, **kwargs: pd.DataFrame(
            columns=['number_of_building_0m', 'area_of_self_0m', 'building_scores_avg_0m', 'building_scores_max_0m'],
            data=np.random.random((len(polys), 4)))

        oneclass_model = BaselineModel()
        self.see_if_model_runs(oneclass_model, neighb_init_states, poly_feat_builder)
