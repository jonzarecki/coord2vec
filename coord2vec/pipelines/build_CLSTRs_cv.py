import logging
import random
import time
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from lagoon.dags import DAG
from lagoon.executors.local_executor import LocalExecutor
from simpleai.search.local import hill_climbing_weighted_stochastic, beam
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.config import CLSTR_RESULTS_DIR, SCORES_TABLE, BUILDING_EXPERIMENT_NAME

from coord2vec.evaluation.tasks.one_class_baseline import BaselineModel
from coord2vec.evaluation.tasks.tasks_utils import hash_geoseries
from coord2vec.feature_extraction.feature_bundles import create_CLSTR_features
from coord2vec.feature_extraction.features.osm_features.building_scores import BuildingScores
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.pipelines.lagoon_utils.auto_stage import AutoStage
from coord2vec.pipelines.lagoon_utils.expr_saver_task import ExprSaverTask
from coord2vec.pipelines.lagoon_utils.for_wrapper import ForInputTask, ForCalcTask, define_for_dependencies
from coord2vec.pipelines.lagoon_utils.lambda_task import LambdaTask


def _create_specific_task():
    features = create_CLSTR_features()
    poly_feature_builder = FeaturesBuilder(features, cache_table="LOC_CLSTR_features")

    # TODO: do we want one-class ? our problem is not one-class
    models = {'One Class SVM': OneClassSVM(),
              'isolation forest': IsolationForest(),
              'Gaussians': EllipticEnvelope(),
              'baseline': BaselineModel()
              }
    specific_task = HandlerBuildCLSTRs(poly_feature_builder, models=models)  # normal
    return specific_task


# def build_model

def run_experiment_lagoon():
    np.random.seed(42)
    random.seed(42)
    S_program = AutoStage("program")

    get_task = LambdaTask(_create_specific_task, ["task"])
    S_program.add_auto(get_task)
    get_dataset = LambdaTask(lambda task: task.get_dataset(), ["geos", "y"])
    S_program.add_auto(get_dataset)

    # transform the CLSTRs to features
    # transform = LambdaTask(lambda task, geos, y: task.transform(geos[y]), ["X_true_CLSTR_df"])
    # S_program.add_auto(transform)

    # convert to buildings
    extract_buildings = LambdaTask(lambda task, geos, y:
                                   task.extract_buildings_from_polygons(geos, y, return_source=True),
                                   ["building_gs", "buildings_y", "source_indices"])
    S_program.add_auto(extract_buildings)

    def train_predict_on_split(task, building_gs, buildings_y, source_indices, geos, y,
                               source_train_indices, source_test_indices):
        building_train_indices = np.isin(source_indices, source_train_indices)
        building_test_indices = np.isin(source_indices, source_test_indices)

        # fetch train-set and fit
        buildings_train_gs = building_gs.iloc[building_train_indices].reset_index(drop=True)
        y_train_buildings = buildings_y[building_train_indices]

        buildings_test_gs = building_gs.iloc[building_test_indices].reset_index(drop=True)
        y_test_buildings = buildings_y[building_test_indices]

        train_true_geos = geos[np.isin(range(len(geos)), source_train_indices) & y]  # train-test in CLSTRs
        test_true_geos = geos[np.isin(range(len(geos)), source_test_indices) & y]  # train-test in CLSTRs

        fpb = task.embedder  # feature extractor for polygons
        # add the building scores feature
        train_hash = hash_geoseries(geos[source_train_indices])
        fpb.features += [BuildingScores(SCORES_TABLE, BUILDING_EXPERIMENT_NAME, 'BalancedRF1000',
                                        # TODO: doesn't match current MetaModel naming
                                        train_geom_hash=train_hash, radius=radius) for radius in [0, 25]]

        heuristic_guiding_model = BaselineModel()
        heuristic_guiding_model.fit(task.transform(train_true_geos))
        # for i in trange(5, desc="Training CLSTR heuristic"):
        #     potential_CLSTRs_test = parmap(lambda b: building_to_CLSTR(b, fpb, heuristic_guiding_model),
        #                                 random.sample(buildings_train_gs[y_train]), use_tqdm=True, desc="Calculating potential CLSTRs")
        #
        #     heuristic_guiding_model = OneClassSVM()
        #     heuristic_guiding_model.fit(task.transform(train_true_geos))

        # TODO: do smarter choice of what buildings to start from ?
        score_extractor = FeaturesBuilder(
            [BuildingScores(SCORES_TABLE, BUILDING_EXPERIMENT_NAME, 'BalancedRF1000', radius=0,
                            train_geom_hash=train_hash)])
        building_scores_sorted = score_extractor.transform(buildings_test_gs)['building_scores_avg_0m'].sort_values(
            ascending=False)

        building_scores = pd.Series(index=buildings_test_gs.iloc[building_scores_sorted.index],
                                    data=building_scores_sorted.values)

        # building_scores = gpd.GeoDataFrame(
        #     zip(buildings_test_gs, np.random.random(len(buildings_test_gs))),
        #     columns=['geometry', 'score'], geometry='geometry').set_index('geometry')

        # TODO: do smarter choice of what buildings to start from. now top scoring 250
        best_test_buildings_with_scores = building_scores.iloc[random.sample(range(1000), 500)]
        potential_CLSTRs_test = parmap(lambda b: building_to_CLSTR(b, fpb, heuristic_guiding_model,
                                                             partial(beam, beam_size=15, iterations_limit=15)),
                                    best_test_buildings_with_scores.index, use_tqdm=True,
                                    desc="Calculating potential CLSTRs", keep_child_tqdm=True, nprocs=16)

        # TODO: postprocessing, which CLSTRs to give. Related to how the fit together.
        print([p[1] for p in potential_CLSTRs_test])
        print([len(p[0].buildings) for p in potential_CLSTRs_test])
        sorted_potential_CLSTRs_test = list(sorted(potential_CLSTRs_test, key=lambda p: p[1], reverse=True))
        # TODO: choose with intel, depending on pluga, etc.
        best_potential_CLSTRs_test = pd.Series(index=[p[0].hull for p in sorted_potential_CLSTRs_test],
                                            data=MinMaxScaler().fit_transform(
                                                [[p[1]] for p in sorted_potential_CLSTRs_test])[:,
                                                 0])  # normalize scores, IMPORTANT
        print(best_potential_CLSTRs_test)

        return building_scores, geos.iloc[source_train_indices], y_train_buildings, geos.iloc[
            source_test_indices], test_true_geos, y_test_buildings, best_potential_CLSTRs_test

    for_input_task = ForInputTask(lambda task, geos, y: (task.kfold_split(geos, y, n_splits=4),),
                                  ["source_train_indices", "source_test_indices"], 4)
    S_program.add_dependency([get_task, get_dataset], for_input_task)

    for_params = ["building_scores", "train_geos", "y_train_buildings", "test_geos", "test_true_geos",
                  "y_test_buildings", "best_potential_CLSTRs_test"]
    for_train_predict_on_split = ForCalcTask(train_predict_on_split,
                                             for_params, [get_task, get_dataset, extract_buildings])

    def merge_predict_results(building_scores, train_geos, y_train_buildings, test_geos, test_true_geos,
                              y_test_buildings, best_potential_CLSTRs_test):
        return [{'building_scores': building_scores[i], 'train_geos': train_geos[i],
                 'y_train_buildings': y_train_buildings[i],
                 'test_geos': test_geos[i], 'test_true_geos': test_true_geos[i],
                 'y_test_buildings': y_test_buildings[i], 'best_potential_CLSTRs_test': best_potential_CLSTRs_test[i]}
                for i in range(len(building_scores))]

    save_params = ["model_results"]

    for_train_predict_on_split_merge = LambdaTask(merge_predict_results, save_params)

    define_for_dependencies(S_program, for_train_predict_on_split, for_input_task, for_train_predict_on_split_merge)

    expr_path = f"{CLSTR_RESULTS_DIR}/{datetime.now().isoformat(' ', 'seconds')}"
    saver = ExprSaverTask(expr_path, save_params)
    S_program.add_dependency(for_train_predict_on_split_merge, saver)

    st = time.time()
    main_dag = DAG("polygon_main")
    main_dag.add(S_program)
    main_dag.visualize()
    a = LocalExecutor(num_workers=4, logging_level=logging.INFO).execute(main_dag)  # , cache_dir="lagoon_cache"
    print(f"total runtime: {(time.time() - st) / 60.} m")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    run_experiment_lagoon()
