import pandas as pd

from sklearn import svm
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from coord2vec.Noam_Adir.manhattan.pipeline import init_pipeline
from coord2vec.Noam_Adir.pipeline.base_pipeline import my_z_score_norm

if __name__ == "__main__":
    n_catboost_iter = 150
    catboost_lr = 1
    catboost_depth = 3

    models = [svm.SVR(),
              LinearRegression(),
              CatBoostRegressor(iterations=n_catboost_iter, learning_rate=catboost_lr, depth=catboost_depth)]
    metrics = {"r2": r2_score,
               "mse": mean_squared_error,
               "mae": mean_absolute_error}

    pipeline_dict = init_pipeline(models)
    task_handler = pipeline_dict["task_handler"]
    coords = pipeline_dict["coords"]
    unique_coords = pipeline_dict["unique_coords"]
    unique_coords_idx = pipeline_dict["unique_coords_idx"]
    price = pipeline_dict["price"]
    features_without_geo = pipeline_dict["features_without_geo"]
    only_geo_features_unique_coords = pipeline_dict["only_geo_features_unique_coords"]
    all_features = pipeline_dict["all_features"]
    all_features_unique_coords = pipeline_dict["all_features_unique_coords"]

    # results on all the building in manhattan
    train_features, test_features, train_price, test_price = train_test_split(all_features, price)
    train_features, test_features = my_z_score_norm(train_features.values, test=test_features.values)
    train_features, test_features = pd.DataFrame(train_features), pd.DataFrame(test_features)
    task_handler.fit_all_models(train_features, train_price)
    scores_1 = task_handler.score_all_model_multi_metrics(test_features, test_price, measure_funcs=metrics)
    print(scores_1)
    all_building_scores = pd.DataFrame(scores_1)

    # results on all the building in manhattan
    train_features, test_features, train_price, test_price = train_test_split(all_features_unique_coords,
                                                                              price[unique_coords_idx])
    task_handler.fit_all_models(train_features, train_price)
    scores_2 = task_handler.score_all_model_multi_metrics(test_features, test_price, measure_funcs=metrics)
    unique_building_scores = pd.DataFrame(scores_2)
    print(scores_2)
    print("results on all buildings\n", all_building_scores)
    print("results on non-repeating coords\n", unique_building_scores)

