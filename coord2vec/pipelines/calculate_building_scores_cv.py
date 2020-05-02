import os
import random
from datetime import datetime
from itertools import product

import logging
import numpy as np
import pandas as pd
from lagoon.dags import DAG, Stage
from lagoon.executors.local_executor import LocalExecutor
from shapely import wkt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from coord2vec.common.itertools import flatten
from coord2vec.config import BUILDINGS_FEATURES_TABLE, BUILDING_RESULTS_DIR, NEG_RATIO, SCORES_TABLE, \
    BUILDING_EXPERIMENT_NAME, TRUE_POSITIVE_RADIUS
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.evaluation.tasks.tasks_utils import save_scores_to_db, hash_geoseries
from coord2vec.feature_extraction.feature_bundles import create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.pipelines.lagoon_utils.expr_saver_task import ExprSaverTask
from coord2vec.pipelines.lagoon_utils.for_wrapper import define_for_dependencies, ForCalcTask, ForInputTask
from coord2vec.pipelines.lagoon_utils.lambda_task import LambdaTask
import geopandas as gpd
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier


def _create_task():
    features = create_building_features()
    feature_builder = FeaturesBuilder(features, cache_table=BUILDINGS_FEATURES_TABLE)

    models = {
        f'CatBoost_depth{depth}_lr{lr}_l2reg{l2reg}': CatBoostClassifier(depth=depth, learning_rate=lr,
                                                                         l2_leaf_reg=l2reg, iterations=300,
                                                                         verbose=False, thread_count=8, od_pval=1e-5)
        for depth, lr, l2reg in product([4, 7, 10], [0.03, 0.1, 0.15], [1, 4, 9])
        # 'SVM C=0.1': SVC(C=0.1, probability=True, gamma='auto'),
        # 'SVM C=0.01': SVC(C=0.01, probability=True, gamma='auto'),
        # 'SVM C=1': SVC(C=1, probability=True, gamma='auto'),
        # 'CatBoost': CatBoostClassifier(loss_function = 'CrossEntropy', iterations=300, depth=3, learning_rate=0.15, l2_leaf_reg=4, verbose=False),
        # 'Logistic Regression': LogisticRegression(),
        # 'BalancedRF1000': BalancedRandomForestClassifier(n_estimators=1000),
        # 'BalancedRF1000_depth4': BalancedRandomForestClassifier(n_estimators=1000, max_depth=4),
        # 'BalancedRF100_depth3': BalancedRandomForestClassifier(n_estimators=100),
        # 'BalancedRF100_depth5': BalancedRandomForestClassifier(n_estimators=100, max_depth=5),
        # 'XGBoost': XGBClassifier(n_estimators=50, early_stopping_round=10)
    }

    task = TaskHandler(feature_builder, models=models)
    return task


def run_experiment_lagoon():
    np.random.seed(42)
    random.seed(42)

    S_program = Stage("program")
    # task = _create_task()
    get_task = LambdaTask(_create_task, ["task"])

    get_dataset = LambdaTask(lambda task: task.get_dataset(), ["geos", "y"])
    # geos, y = task.get_dataset()
    S_program.add_dependency(get_task, get_dataset)

    # convert to buildings
    extract_buildings = LambdaTask(lambda task, geos, y:
                                   task.extract_buildings_from_polygons(geos, y, return_source=True),
                                   ["building_gs", "buildings_y", "source_indices"])
    # building_gs, buildings_y, source_indices = task.extract_buildings_from_polygons(geos, y, neg_ratio=2,
    #                                                                                 return_source=True)
    S_program.add_dependency([get_task, get_dataset], extract_buildings)

    # transform the buildings to features
    transform = LambdaTask(lambda task, building_gs: task.transform(building_gs), ["X_df"])
    # X_df = task.transform(building_gs)
    S_program.add_dependency([get_task, extract_buildings], transform)

    def train_predict_on_split(task, X_df, buildings_y, source_indices, source_train_indices, source_test_indices,
                               geos):
        train_indices = np.isin(source_indices, source_train_indices)
        test_indices = np.isin(source_indices, source_test_indices)

        # fetch train-set and fit
        X_train_df = X_df.iloc[train_indices]
        y_train = buildings_y[train_indices]

        # sample neg_ratio false samples
        num_neg_samples = int(NEG_RATIO * y_train.sum())
        X_train_neg_df = X_train_df[y_train == 0]
        random_indices = np.random.choice(range(len(X_train_neg_df)), num_neg_samples, replace=False)
        X_train_df = pd.concat([X_train_df[y_train == 1], X_train_neg_df.iloc[random_indices]])
        y_train = np.concatenate([y_train[y_train == 1], y_train[y_train == 0][random_indices]]).astype(int)

        # try larger labels - it didnt work..
        y_train_soft = task.get_soft_labels(gpd.GeoSeries(data=X_train_df.index.values), radius=TRUE_POSITIVE_RADIUS)
        # y_train = (y_train > 0.5).astype(int)


        X_test_df = X_df.iloc[test_indices]
        soft_labels_cache = os.path.join(os.path.curdir, "soft_cache", hash_geoseries(gpd.GeoSeries(X_test_df.index)))
        y_test_soft = task.get_soft_labels(gpd.GeoSeries(data=X_test_df.index.values), radius=TRUE_POSITIVE_RADIUS,
                                           cache_dir=soft_labels_cache)
        # y_test = buildings_y[test_indices].astype(int)  # for Catboost

        task.fit_all_models(X_train_df, y_train_soft)
        # task.fit_all_models(X_train_df, y_train_soft, X_test_df, y_test_soft)  # for Catboost
        task.save_all_models()
        train_probas_df = task.predict_all_models(X_train_df)
        test_probas_df = task.predict_all_models(X_test_df)

        # score models
        model2scores = {}
        model2score = task.score_all_models(X_test_df, y_test_soft)
        for model, score in model2score.items():
            model2scores.setdefault(model, []).append(score)

        print(f"Insererting building results to {SCORES_TABLE}")
        probas_df = task.predict_all_models(X_df)
        train_hash = hash_geoseries(geos[source_train_indices])
        save_scores_to_db(probas_df, SCORES_TABLE, BUILDING_EXPERIMENT_NAME, train_hash)

        return X_train_df, y_train, X_test_df, y_test_soft, train_probas_df, test_probas_df, task.models_dict, model2scores

    def merge_predict_results(X_train_df, X_test_df, y_train, y_test, source_train_indices,
                              source_test_indices, train_probas_df, test_probas_df, models_dict, model2scores):

        # merge_dicts
        kfold_results = [(X_train_df[i], X_test_df[i], y_train[i], y_test[i], source_train_indices[i],
                          source_test_indices[i], train_probas_df[i], test_probas_df[i], models_dict[i],
                          model2scores[i]) for i in
                         range(len(X_train_df))]

        return (kfold_results,)

    for_input_task = ForInputTask(lambda task, geos, y: (task.kfold_split(geos, y, n_splits=4),),
                                  ["source_train_indices", "source_test_indices"], 4)
    S_program.add_dependency([get_task, get_dataset], for_input_task)

    for_params = ["X_train_df", "y_train", "X_test_df", "y_test", "source_train_indices",
                  "source_test_indices", "train_probas_df", "test_probas_df", "models_dict",
                  "model2scores", "geos"]
    for_train_predict_on_split = ForCalcTask(train_predict_on_split,
                                             for_params, [get_task, get_dataset, extract_buildings, transform])

    for_train_predict_on_split_merge = LambdaTask(merge_predict_results, 'model_results')

    define_for_dependencies(S_program, for_train_predict_on_split, for_input_task, for_train_predict_on_split_merge)

    def print_model_scores(kfold_results):
        kfold_scores = [res[-1] for res in kfold_results]
        all_models = set(flatten([list(kfold.keys()) for kfold in kfold_scores]))

        for model in all_models:
            model_mean = np.mean(flatten([kfold[model] for kfold in kfold_scores]))
            print(f"{model} AUC: \t {model_mean}")

    print_model2scores = LambdaTask(print_model_scores, [])
    S_program.add_dependency(for_train_predict_on_split_merge, print_model2scores)

    def save_results(geos, kfold_results):
        return ([(geos.iloc[source_train_indices], X_train_df, y_train, train_probas_df,
                  geos.iloc[source_test_indices], X_test_df, y_test, test_probas_df,
                  models_dict, model2scores) for
                 X_train_df, X_test_df, y_train, y_test, source_train_indices,
                 source_test_indices, train_probas_df, test_probas_df, models_dict, model2scores in kfold_results],)
        # return X_train_df, y_train, X_test_df, y_test, geos.iloc[source_train_indices], \
        #        geos.iloc[source_test_indices], train_probas_df, test_probas_df, model2scores

    # change for results to objects to be saved
    save_params = ["model_results"]
    results2save = LambdaTask(save_results, save_params,
                              override_input_names=save_params)
    S_program.add_dependency([get_dataset, for_train_predict_on_split_merge], results2save)

    expr_path = f"{BUILDING_RESULTS_DIR}/{datetime.now().isoformat(' ', 'seconds')}"
    saver = ExprSaverTask(expr_path, save_params)

    S_program.add_dependency(results2save, saver)

    main_dag = DAG("main")
    main_dag.add(S_program)
    main_dag.visualize()
    a = LocalExecutor(num_workers=4, log_to=["elastic_prod"]).execute(main_dag)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    run_experiment_lagoon()
