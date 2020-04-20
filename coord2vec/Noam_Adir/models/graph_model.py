import glob
import os

from catboost import CatBoostRegressor
from geopandas import GeoSeries
from shapely.geometry import Point
from sklearn.base import BaseEstimator
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import torch
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from coord2vec.Noam_Adir.manhattan.pipeline import init_pipeline
from coord2vec.Noam_Adir.models.geom_graph_builder import GeomGraphBuilder
from coord2vec.Noam_Adir.models.pyGAT.lightning_modules import LitGAT
from coord2vec.Noam_Adir.models.pygcn import sparse_mx_to_torch_sparse_tensor
from coord2vec.Noam_Adir.models.pygcn.lightning_modules import LitGCN


class GraphModel:
    """
    An API class that warps graph models
    """
    def __init__(self, hparams: Namespace):
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if hparams.cuda:
            torch.cuda.manual_seed(hparams.seed)
        self.model = None
        self.model_name = "GAT" if hparams.attention else "GCN"
        self.graph_model_class = LitGAT if hparams.attention else LitGCN

    def fit(self, X: pd.DataFrame, y, train_idx=None, restart_fit=True):
        train_idx = range(len(X)) if train_idx is None else train_idx
        # building the trainer

        checkpoint_callback = ModelCheckpoint(
            filepath=os.getcwd(),
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=self.model_name + "_"
        )
        early_stopping = EarlyStopping('val_loss', patience=hparams.early_stop_patience)
        all_relevant_ckpts = glob.glob(os.path.join(os.getcwd(), self.model_name+'*.ckpt'))
        ckpt_exsits = False if len(all_relevant_ckpts) == 0 else True  # TODO hparams
        if not restart_fit and ckpt_exsits:
            # will continue from the most recent cpkt file
            resume_from_ckpt = max(all_relevant_ckpts, key=os.path.getctime)
        else:
            resume_from_ckpt = None
        logs_path = hparams.log_path  # TODO hparams
        logger = TensorBoardLogger(save_dir=logs_path, name=f"{self.model_name}_tb")
        gpus = None if hparams.gpus == 0 else hparams.gpus  # TODO hparams
        trainer = Trainer(max_epochs=hparams.epochs, logger=logger,
                          resume_from_checkpoint=resume_from_ckpt,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stopping,
                          check_val_every_n_epoch=hparams.val_every_n_epoch,
                          gpus=gpus)

        adj = self.create_geom_graph_from_df_index(X)

        self.model = self.graph_model_class(X.to_numpy(), y, adj, train_idx, nclass=hparams.nclass, hparams=hparams)
        trainer.fit(self.model)

    def predict_proba(self, X: pd.DataFrame):
        pass  # TODO add functionality

    def predict_idx_last_graph(self, pred_idx):
        """
        The output of the model of the nodes pred_idx are of in the last graph the model was trained on
        Args:
            pred_idx: the indexes of the nodes in the last graph the model was trained on
                in most cases pred_idx=test_idx

        Returns: numpy array of predictions the length of pred_inx

        """
        if self.model is None:
            print("model should be fitted before prediction")
            return
        adj = self.model.adj.detach()
        X_tensor = self.model.X.detach()
        pred_idx = torch.LongTensor(pred_idx)
        self.model.eval()
        return (self.model.forward(X_tensor, adj)[pred_idx]).detach().numpy()

    def predict_idx(self, pred_idx):
        """
        deprecated, delete after integration

        """
        if self.model is None:
            print("model should be fitted before prediction")
            return
        adj = self.model.adj.detach()
        X_tensor = self.model.X.detach()
        pred_idx = torch.LongTensor(pred_idx)
        self.model.eval()
        return (self.model.forward(X_tensor, adj)[pred_idx]).detach().numpy()

    def predict(self, X:pd.DataFrame):
        """

        Args:
            X: pd DataFrame,  create new graph and predict according to new adj on all the indexes,

        Returns: numpy array of predictions the length of X

        """
        if self.model is None:
            print("model should be fitted before prediction")
            return
        adj = self.create_geom_graph_from_df_index(X)
        adj = self.model.procces_adj(adj)
        X_tensor = torch.tensor(X, requires_grad=False, dtype=torch.float64)
        pred_idx = torch.LongTensor(range(len(X)))
        self.model.eval()
        return (self.model.forward(X_tensor, adj)[pred_idx]).numpy()

    @staticmethod
    def create_geom_graph_from_df_index(df:pd.DataFrame, graph_building_method="DT"):
        """

        Args:
            df: a DataFrame with shpley.geometries as index
            graph_building_method: method for building the graph
                    can be:"DT", "RNG" or "fix_distance"

        Returns: the graph adjacency matrix as scipy sparse matrix

        """
        all_geometries = GeoSeries(list(df.index))
        graph_builder = GeomGraphBuilder(geometries=all_geometries, method=graph_building_method)
        graph_builder.construct_vertices()
        adj = graph_builder.get_adj_as_scipy_sparse_matrix()
        assert GeoSeries([geom for n, geom in graph_builder.graph.nodes(data="geometry")]).geom_almost_equals(
            all_geometries).all()
        return adj


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # add hear non model specific parameters
    parser.add_argument('--gpus', type=int, default=0, help='Number of gpus.')
    parser.add_argument('--use-double-precision', action='store_true', default=False, help='use-double-precision.')
    parser.add_argument('--attention', action='store_true', default=False, help='use GAT layers')
    parser.add_argument('--nclass', type=int, default=1, help='Number of classes if 1 the regression.')
    parser.add_argument('--log-path', default='/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir/tb_logs',
                        help='tensorbord log path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--val-every-n-epoch', type=int, default=10, help='Number of epochs to validate after')
    parser.add_argument('--early-stop', action='store_true', default=False, help='stop training when val_loss stop improving')
    parser.add_argument('--early-stop-patience', type=int, default=100, help='the patience of the early stop feature')

    # good practice to define LightningModule speficic params in the module
    hparams = parser.parse_args()
    graph_model_class = LitGAT if hparams.attention else LitGCN
    parser = graph_model_class.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    hparams.cuda = (hparams.gpus != 0) and torch.cuda.is_available()

    # random seed intialization
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    if hparams.cuda:
        torch.cuda.manual_seed(hparams.seed)

    # baseline
    n_catboost_iter = 150
    catboost_lr = 1
    catboost_depth = 3
    models = [CatBoostRegressor(iterations=n_catboost_iter, learning_rate=catboost_lr,
                                depth=catboost_depth, verbose=False)]
    metrics = {"r2": r2_score,
               "mse": mean_squared_error,
               "mae": mean_absolute_error}

    # get the data
    pipeline_dict = init_pipeline(models)
    task_handler = pipeline_dict["task_handler"]
    unique_coords = pipeline_dict["unique_coords"]
    unique_coords_idx = pipeline_dict["unique_coords_idx"]
    price = pipeline_dict["price"]

    all_features_unique_coords = pipeline_dict["all_features_unique_coords"]

    # fit models
    task_handler.add_graph_model(GraphModel(hparams))
    all_idx = list(range(len(all_features_unique_coords)))
    train_idx, test_idx = train_test_split(all_idx)

    unique_coords_as_points = pd.Series([Point(coord[0], coord[1]) for coord in unique_coords])
    all_features_unique_coords = all_features_unique_coords.set_index(unique_coords_as_points)
    task_handler.fit_all_models_with_idx(all_features_unique_coords, price[unique_coords_idx], train_idx)
    scores = task_handler.score_all_model_multi_metrics_idx(all_features_unique_coords, price, test_idx,
                                                            measure_funcs=metrics)
    scores_df = pd.DataFrame(scores)
    print(scores_df)

    print("continue training graph model again on a new graph")
    graph_model = list(task_handler.graph_models_dict.values())[0]
    graph_model.fit(all_features_unique_coords, price, restart_fit=False)
    scores = task_handler.score_all_model_multi_metrics_idx(all_features_unique_coords, price, test_idx,
                                                            measure_funcs=metrics)