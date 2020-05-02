import itertools
import hashlib
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from functools import partial
from typing import List

import numpy as np
from lagoon.dags import DAG
from lagoon.executors.local_executor import LocalExecutor
import geopandas as  gpd
import pandas as pd
from coord2vec.config import BUILDINGS_FEATURES_TABLE, BUILDING_RESULTS_DIR, DISTANCE_CACHE_DIR, TRUE_POSITIVE_RADIUS
from coord2vec.experiments.experiment_loader import load_separate_model_results, get_all_existing_model_names
from coord2vec.feature_extraction.feature_bundles import create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.pipelines.lagoon_utils.auto_stage import AutoStage
from coord2vec.pipelines.lagoon_utils.expr_saver_task import ExprSaverTask
from coord2vec.pipelines.lagoon_utils.for_wrapper import ForInputTask, ForCalcTask, define_for_dependencies
from coord2vec.pipelines.lagoon_utils.lambda_task import LambdaTask


def _create_task():
    features = create_building_features(elements=None)
    feature_builder = FeaturesBuilder(features, cache_table=BUILDINGS_FEATURES_TABLE)

    # hyper params for model are defined in MetaModel
    task = TaskHandler(feature_builder, models=None)
    return task


def run_experiment_lagoon():
    np.random.seed(42)
    random.seed(42)
    curr_datetime = datetime.now()
    # curr_datetime = datetime(2020, 3, 8, 12, 56, 8)

    expr_path = f"{BUILDING_RESULTS_DIR}/{curr_datetime.isoformat(' ', 'seconds')}"
    if not os.path.exists(os.path.dirname(expr_path)):
        os.makedirs(expr_path, exist_ok=True)

    S_program = AutoStage("program")

    get_task = LambdaTask(_create_task, ["task"])

    get_dataset = LambdaTask(lambda task: task.get_dataset(), ["geos", "y"])
    S_program.add_dependency(get_task, get_dataset)

    cached_models_task = LambdaTask(lambda: get_all_existing_model_names(exp_datetime=curr_datetime),
                                    'cached_model_names')

    # convert to buildings
    # FIXME I'm ugly code
    positive_sampling_opts = [1]
    noise_sampling_to_positive_opts = [0.05]
    print(f"num of models: {len(_create_task().models_dict) * len(positive_sampling_opts) * len(noise_sampling_to_positive_opts)}")
    for _positive_sampling, _noise_sampling_to_positive in itertools.product(positive_sampling_opts, noise_sampling_to_positive_opts):
        pass_vars = {'positive_sampling': _positive_sampling, 'noise_sampling_to_positive': _noise_sampling_to_positive}
        extract_buildings = LambdaTask(
            lambda task, geos, y, positive_sampling, noise_sampling_to_positive: task.extract_buildings_from_polygons(geos, y, return_source=True,
                                                                       positive_sampling=positive_sampling,
                                                                       noise_sampling_to_positive=noise_sampling_to_positive),
            ["building_gs", "buildings_y", "source_indices"], passed_vars=pass_vars)
        S_program.add_dependency([get_task, get_dataset], extract_buildings)

        # transform the buildings to features
        transform = LambdaTask(lambda task, building_gs: task.transform(building_gs), ["X_df"])
        S_program.add_dependency([get_task, extract_buildings], transform)


        # create kfold split and y_soft #TODO split this into two tasks
        def create_kfold_and_y_soft(task, geos, y, source_indices, building_gs):
            # TODO: ugly, assumes passed bounding_geom is a goem list
            geos_kfold_split = task.kfold_split(geos, y, n_splits=task.n_splits)
            building_cv_idxs = [(np.isin(source_indices, source_train_indices).nonzero()[0],
                                 np.isin(source_indices, source_test_indices).nonzero()[0])
                                for (source_train_indices, source_test_indices) in geos_kfold_split]

            # TODO: one cache for all (doesn't depend on geos)
            y_true_soft = task.get_soft_labels(building_gs, radius=TRUE_POSITIVE_RADIUS, cache_dir=DISTANCE_CACHE_DIR)
            return building_cv_idxs, y_true_soft, geos_kfold_split

        create_kfold_y_soft_task = LambdaTask(create_kfold_and_y_soft, ['cv_idxs', 'y_true_soft', 'geos_kfold_split'])
        S_program.add_dependency([get_task, get_dataset, extract_buildings], create_kfold_y_soft_task)

        for_input_task = ForInputTask(lambda task: list(task.models_dict.values()), ['model'],
                                      len(_create_task().models_dict))  # TODO ugly due to no dynamic DAGs
        S_program.add_dependency([get_task], for_input_task)

        def train_predict_model(task, X_df, buildings_y, y_true_soft, cv_idxs, model, geos, geos_kfold_split, cached_model_names,
                                positive_sampling, noise_sampling_to_positive):
            try:
                model.model_name = f"{model.model_name},positive_sampling={positive_sampling}" \
                                            f",noise_sampling_to_positive={noise_sampling_to_positive}"
                print(noise_sampling_to_positive, positive_sampling, model.model_name)
                if model.model_name not in cached_model_names:
                    trained_model = model.fit(X_df, buildings_y, cv=cv_idxs, y_soft=y_true_soft, task=task)
                    results = trained_model.results

                    # add geos
                    results['geos_kfold_split'] = geos_kfold_split
                    results['geos'] = geos
                    # add task hp params
                    results['hp_dict']['positive_sampling'] = positive_sampling
                    results['hp_dict']['noise_sampling_to_positive'] = noise_sampling_to_positive

                # Save each model seperatly!
                multi_model_dir = os.path.join(expr_path, hashlib.md5(model.model_name.encode()).hexdigest())  # build consistent hash with md5


                if model.model_name not in cached_model_names:
                    os.makedirs(multi_model_dir, exist_ok=True)
                    with open(os.path.join(multi_model_dir, "results.pickle"), "wb+") as f:
                        pickle.dump({'kfold_results': results}, f)
                else:
                    logging.info(f"Loaded model {model.model_name} from cache")
                    with open(os.path.join(multi_model_dir, "results.pickle"), "rb") as f:
                        results = pickle.load(f)

                return results
            except:
                print(f"failed: {model.model_name},positive_sampling={positive_sampling}" \
                                            f",noise_sampling_to_positive={noise_sampling_to_positive}")
                from traceback import print_exc
                print_exc(file=sys.stdout)
                return None



        for_train_predict_on_model = ForCalcTask(train_predict_model, ['model_results'],
                                                 [get_task, get_dataset, extract_buildings, transform,
                                                  create_kfold_y_soft_task, cached_models_task], passed_vars=pass_vars)

        # merge_results_task = LambdaTask(merge_results, 'model_results')
        merge_results_task = LambdaTask(lambda model_results: model_results, 'model_results',
                                        override_input_names=['model_results'])
        define_for_dependencies(S_program, for_train_predict_on_model, for_input_task, merge_results_task)


    main_dag = DAG("main")
    main_dag.add(S_program)
    main_dag.visualize()
    a = LocalExecutor(num_workers=16, logging_level=logging.INFO).execute(main_dag)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    run_experiment_lagoon()
