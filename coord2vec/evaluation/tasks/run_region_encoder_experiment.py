import sys
import os
from typing import List

import random, numpy
random.seed(1554); numpy.random.seed(42)  # set random seeds

import numpy as np
import pandas as pd
from attr import dataclass
from tqdm import tqdm

from coord2vec.common.multiproc_util import parmap
from coord2vec.evaluation.tasks.region_encoder.grid.create_grid import RegionGrid
# from coord2vec.feature_extraction.features_builders import house_price_builder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from coord2vec.evaluation.tasks.region_encoder.config import get_config
from sklearn.model_selection import KFold
from coord2vec.evaluation.tasks.region_encoder.experiments.prediction import HousePriceModel, TrafficVolumeModel, \
    CheckinModel, PredictionModel

task = "house_price"
# estimator = sys.argv[2]  # based on the one that worked best in the paper for each task


# if len(sys.argv) == 1:
#     raise Exception("User must input task, estimator")
# else:
#     task = sys.argv[1]
#     estimator = sys.argv[2]
#     try:
#        n_epochs = int(sys.argv[3])
#     except IndexError:
#         n_epochs = 25


# assert(estimator in ['xgb', 'lasso', 'rf', 'mlp', 'ridge'])

n_folds = 5
n_epochs = 25

c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()

tmp = pd.DataFrame(region_grid.feature_matrix, index = region_grid.idx_coor_map.values())

class Model:
    model: PredictionModel
    name: str
    err: np.ndarray

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.err = np.zeros((n_folds, 2))

models: List[Model] = []

if task == 'house_price':
    estimator = "rf"
    input_data = region_grid.load_housing_data(c['housing_data_file'])
    input_data_m = region_grid.load_housing_data('/media/yonatanz/yz/data/region-encoder-data/nyc/zillow_house_price_manhattan.csv')
    # Initialize Models
    naive_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs)
    models.append(Model(naive_mod, 'Naive'))

    naive_raw_feature_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                            region_grid.weighted_mtx)
    models.append(Model(naive_raw_feature_mod, 'raw features'))
    naive_raw_feature_img_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                                region_grid.weighted_mtx, c['kmeans_file'])
    models.append(Model(naive_raw_feature_img_mod, 'raw features+kmeans'))
    deepwalk_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    models.append(Model(deepwalk_mod, 'DeepWalk Embedding'))

    node2vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['node2vec_file'])
    models.append(Model(node2vec_mod, 'Node2Vec Embedding'))

    nmf_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    models.append(Model(nmf_mod, 'Matrix Factorization'))
    pca_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    models.append(Model(pca_mod, 'PCA'))
    autoencoder_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    models.append(Model(autoencoder_mod, 'AutoEncoder'))
    joint_ae_dw = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
                                  c['autoencoder_embedding_file'])
    models.append(Model(joint_ae_dw, 'AutoEncoder + DeepWalk'))
    tile2vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['tile2vec_file'])
    models.append(Model(tile2vec_mod, 'Tile2Vec'))
    hdge_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['hdge_file'])
    models.append(Model(hdge_mod, 'HDGE'))
    msne_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'])
    models.append(Model(msne_mod, 'MSNE'))
    msne_tile2_vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'], c['tile2vec_file'])
    models.append(Model(msne_tile2_vec_mod, 'MSNE + Tile2Vec'))

    re_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    models.append(Model(re_mod, 'RegionEncoder'))

    # our models
    house_price_builder_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs,
                                              extract_coords_features=lambda c: house_price_builder.extract_coordinates(c))
    models.append(Model(house_price_builder_mod, 'House price builder'))


    # node2vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['node2vec_file'])
    # re_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    # joint_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    # nmf_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    # pca_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    # autoencoder_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    # joint_ae_dw = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
    #                               c['autoencoder_embedding_file'])
    # tile2vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['tile2vec_file'])
    # msne_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'])
    # msne_tile2_vec_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'], c['tile2vec_file'])
    # hdge_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['hdge_file'])

elif task == 'traffic':
    input_data = region_grid.load_traffic_data(c['traffic_data_file'], city=c['city_name'])
    # Initialize Models
    naive_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs)
    naive_raw_feature_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                               region_grid.weighted_mtx)
    naive_raw_feature_img_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                                region_grid.weighted_mtx, c['kmeans_file'])
    deepwalk_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    node2vec_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['node2vec_file'])
    re_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    joint_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    nmf_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    pca_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    autoencoder_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    tile2vec_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['tile2vec_file'])
    joint_ae_dw = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
                                  c['autoencoder_embedding_file'])
    msne_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'])
    msne_tile2_vec_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'], c['tile2vec_file'])
    hdge_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['hdge_file'])

elif task == 'check_in':
    input_data = region_grid.get_checkin_counts(metric="mean")
    naive_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs)
    naive_raw_feature_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, embedding=region_grid.feature_matrix,
                                               second_embedding=region_grid.weighted_mtx)
    naive_raw_feature_img_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                                   region_grid.weighted_mtx, c['kmeans_file'])
    deepwalk_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, embedding=c['deepwalk_file'])
    node2vec_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['node2vec_file'])
    re_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    nmf_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    pca_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    autoencoder_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    tile2vec_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['tile2vec_file'])
    joint_ae_dw = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
                                     c['autoencoder_embedding_file'])
    msne_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'])
    msne_tile2_vec_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['msne_file'], c['tile2vec_file'])
    hdge_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['hdge_file'])

else:
    raise NotImplementedError("User must input task: {'house_price', 'traffic', or 'checkin'")

print("K-Fold Learning - {}".format(estimator))


# Get Features
for model in models:
    print(model.name, end="\t\t")
    model.model.get_features(input_data)


k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=5555)



train_ind_arr = np.arange(naive_mod.X.shape[0])

for fold_cntr, (train_idx, test_idx) in tqdm(enumerate(k_fold.split(train_ind_arr)), "Calculating folds", total=n_folds):

    def calc_model_on_fold(m: Model):
        rmse, mae = m.model.train_eval(train_idx, test_idx, estimator)
        return rmse, mae

    for model, (rmse, mae) in zip(models, parmap(calc_model_on_fold, models)):
        model.err[fold_cntr, 0] = rmse
        model.err[fold_cntr, 1] = mae

results = []

for model in models:
    err_mean = np.mean(model.err, axis=0)
    err_std = np.std(model.err, axis=0)
    results.append([model.name, err_mean[0], err_std[0], err_mean[1], err_std[1]])


results_df = pd.DataFrame(results, columns=['model', 'cv rmse', 'std rmse', 'cv mae', 'std mae'])
print(results_df)

# results_df.to_csv(f"{REGION_ENCODER_DIR_PATH}/experiments/results/{task}-{estimator}-results.csv")