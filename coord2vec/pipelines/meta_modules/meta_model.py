import copy
import os
import time
from typing import List, Dict

import pandas as pd
import logging

from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from joblib import parallel_backend
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
import numpy as np
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from coord2vec.config import TRUE_POSITIVE_RADIUS, DISTANCE_CACHE_DIR
from coord2vec.evaluation.evaluation_metrics.metrics import soft_auc, get_soft_auc_scorer
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.feature_extraction.feature_bundles import modiin_bundle_features, karka_bundle_features, \
    all_bundle_features
from coord2vec.feature_extraction.feature_utils import FeatureFilter
from coord2vec.pipelines.meta_modules.meta_module import MetaModule


class MetaModel(MetaModule, TransformerMixin):
    """
    Class responsible for handling all hyperparameters related to the model (type, regularization, etc.)
    Will be accessed using a YAML config file or params
    """
    hp_model = {
        'catboost': lambda self: CatBoostClassifier(loss_function='CrossEntropy',
                                                    iterations=self.model_catboost__n_estimators,
                                                    depth=self.model_catboost__depth, thread_count=4,
                                                    learning_rate=self.model_catboost__lr,
                                                    l2_leaf_reg=self.model_catboost__l2reg,
                                                    eval_metric='AUC',
                                                    early_stopping_rounds=200,
                                                    verbose=False),
        # 'SVM': lambda self: Pipeline(
        #     [('scaler', StandardScaler()), ('model', SVC(C=self.model_SVM__C, probability=True, gamma='auto'))]),
        # 'Logistic_Regression': lambda self: Pipeline(
        #     [('scaler', StandardScaler()), ('model', LogisticRegression(solver='liblinear', n_jobs=1))]),
        # 'BalancedRF': lambda self: BalancedRandomForestClassifier(n_estimators=self.model_BalancedRF__n_estimators,
        #                                                           max_depth=self.model_BalancedRF__depth),
        # 'XGBoost': lambda self: XGBClassifier(learning_rate=self.model_XGBoost__lr,
        #                                       max_depth=self.model_XGBoost__depth,
        #                                       n_estimators=self.model_XGBoost__n_estimators,
        #                                       reg_lambda=self.model_XGBoost__l2reg,
        #                                       early_stopping_round=50, nthread=4),
        # 'MLP': lambda self: MLPClassifier(hidden_layer_sizes=(32, 16)),
    }

    # catboost hyper parameters
    hp_model_catboost__n_estimators = [1000]  # [700, 1000, 1500]
    # TODO: possible bug when same name is used and also filtered, check in test sometime
    hp_model_catboost__lr = [0.03]  # def is 0.03  [0.01, 0.03, 0.1, 0.15]  [0.03, 0.05, 0.01]
    hp_model_catboost__depth = [3]#, 4]  # def is 6  [4, 6, 8]
    hp_model_catboost__l2reg = [6]  # def is 3
    hp_model_catboost__soft_radius = [100]#, 150, 200]  # [0, 50, 100, 150]

    # BalancedRF hyper parameters
    hp_model_BalancedRF__n_estimators = [300, 700, 1000]
    hp_model_BalancedRF__depth = [4, 7, 10]

    # XGBoost hyper params
    hp_model_XGBoost__lr = [0.03]  # def is 0.3  [0.03, 0.1, 0.15, 0.3]
    hp_model_XGBoost__depth = [8, 10]  # def is 6  [4, 6, 8
    hp_model_XGBoost__n_estimators = [300]  # [200, 300, 500]
    hp_model_XGBoost__l2reg = [0.5, 1, 2]  # def is 1

    # svm hyper params
    hp_model_SVM__C = [0.1, 1, 10]

    # feature selection
    hp_use_fs = [False]#[True]
    hp_use_fs_True__selection_type = {
        # 'None': lambda self, model, cv: None,
        # 'RFE': lambda self, model, cv: SFS(model, self.feature_selection_RFE__nfeatures, forward=False, verbose=2,
        #                                    scoring=get_soft_auc_scorer(), cv=cv, n_jobs=-1),
        # 'RFA': lambda self, model, cv: SFS(model, self.feature_selection_RFA__nfeatures, forward=True, verbose=2,
        #                                    scoring=get_soft_auc_scorer(), cv=cv, n_jobs=-1),
        # 'chi2_kbest': lambda self: SelectKBest(chi2, k=self.use_fs_True__nfeatures),  # only for non-negative feats
        # TODO: using a normal threshold will make it possible to select an automatic number of features.
        # 'linearSVC_importance':
        #     lambda self: SelectFromModel(LinearSVC(C=0.5, penalty='l1', dual=False), threshold=-np.inf,
        #                                  max_features=self.use_fs_True__nfeatures),
        'extra_trees_importance':
            lambda self: SelectFromModel(ExtraTreesClassifier(n_estimators=50), threshold=-np.inf,
                            max_features=self.use_fs_True__nfeatures),
        # 'mutual_info_classif_kbest': lambda self: SelectKBest(mutual_info_classif, k=self.use_fs_True__nfeatures),
        # 'f_classif_kbest': lambda self: SelectKBest(f_classif, k=self.use_fs_True__nfeatures),
    }


    hp_bundle_filter = {
        # 'lvl1': lambda self: FeatureFilter(all_bundle_features, importance=1),
        'lvl2': lambda self: FeatureFilter(all_bundle_features, importance=2),
        # 'lvl3': lambda self: FeatureFilter(all_bundle_features, importance=3),
        # 'lvl4': lambda self: FeatureFilter(all_bundle_features, importance=4),
    }

    # feature selection parameters
    hp_use_fs_True__bundle_filter_lvl1__nfeatures = [50] # [25, 50, 100]
    hp_use_fs_True__bundle_filter_lvl2__nfeatures = [50]  # [25, 50, 100]
    hp_use_fs_True__bundle_filter_lvl3__nfeatures = [100]  # [25, 50, 100]
    hp_use_fs_True__bundle_filter_lvl4__nfeatures = [150, 200, 300]  # [25, 50, 100]


    hp_neg_ratio = [1, 1.5, 2]#[1, 1.5]

    @MetaModule.save_passed_params_wrapper
    def __init__(self,
                 model: str,
                 model_catboost__lr=0.1,
                 model_catboost__depth=7,
                 model_catboost__l2reg=1,
                 model_catboost__n_estimators=300,
                 model_XGBoost__lr=0.3,
                 model_XGBoost__depth=6,
                 model_XGBoost__n_estimators=300,
                 model_XGBoost__l2reg=1,
                 model_catboost__soft_radius=50,
                 model_SVM__C=1,
                 use_fs: bool = False,
                 use_fs_True__selection_type='f_classif_kbest',
                 use_fs_True__bundle_filter_lvl1__nfeatures=0,
                 use_fs_True__bundle_filter_lvl2__nfeatures=0,
                 use_fs_True__bundle_filter_lvl3__nfeatures=0,
                 use_fs_True__bundle_filter_lvl4__nfeatures=0,
                 model_BalancedRF__n_estimators=700,
                 model_BalancedRF__depth=700,
                 bundle_filter='lvl1',
                 neg_ratio=1,
                 **kwargs):
        super().__init__(**kwargs)
        # TODO: using self.init_args and locals() I can implement check that params are in values defined by HPs
        local_vars = locals()  # hides which init param is used
        self.hp_dict = {arg: local_vars[arg] for arg in self.init_args}
        self.model_name = self._create_model_name(self.passed_kwargs)  # defined by wrapper
        self.bundle_filter = bundle_filter
        self.use_soft_labels = self._use_soft_labels(model)

        self.model_BalancedRF__n_estimators = model_BalancedRF__n_estimators
        self.model_BalancedRF__depth = model_BalancedRF__depth

        self.use_fs = use_fs
        self.use_fs_True__selection_type = use_fs_True__selection_type if use_fs else 'all'
        self.use_fs_True__nfeatures = max(use_fs_True__bundle_filter_lvl1__nfeatures, use_fs_True__bundle_filter_lvl2__nfeatures,
                                          use_fs_True__bundle_filter_lvl3__nfeatures, use_fs_True__bundle_filter_lvl4__nfeatures)
        self.model_catboost__lr = model_catboost__lr
        self.model_catboost__depth = model_catboost__depth
        self.model_catboost__l2reg = model_catboost__l2reg
        self.model_catboost__n_estimators = model_catboost__n_estimators
        self.soft_radius = model_catboost__soft_radius
        self.model_SVM__C = model_SVM__C
        self.model_XGBoost__l2reg = model_XGBoost__l2reg
        self.model_XGBoost__n_estimators = model_XGBoost__n_estimators
        self.model_XGBoost__depth = model_XGBoost__depth
        self.model_XGBoost__lr = model_XGBoost__lr

        self.neg_ratio = neg_ratio

        self.model_str = model

    def _create_model_name(self, passed_kwargs):
        # TODO: we can't write this to file, maybe add another one which is file-safe
        return "MetaModel:  " + ",".join(
            [f"{MetaModel.extract_hp_param_name(arg)}={passed_kwargs[arg]}" for arg in passed_kwargs])

    def fit(self, X, y, cv=5, y_soft=None, task=None) -> ("MetaModel", List[float]):
        """
        Fits the model according to the supplied hyperparams
        Args:
            X: Feature matrix
            y: labels
            cv: list of tuples - (train_index, test_index) or int (for normal CV with cv folds)

        Returns:
            self, list of accuracies for each fold
        """
        feat_filter = self.hp_bundle_filter[self.bundle_filter](self)
        X = feat_filter.transform(X)
        # must be last because uses other params
        self.model = self.hp_model[self.model_str](self)

        y_soft = y if y_soft is None else y_soft
        y = np.array(y).astype(int)

        # add the neg ratio
        if type(cv) != int:
            cv = self._negative_sampling(cv, y)

        if self.use_soft_labels and self.soft_radius > 0:
            cache_dir = os.path.join(DISTANCE_CACHE_DIR, str(self.soft_radius))
            y = task.get_soft_labels(gpd.GeoSeries(data=X.index.values), radius=self.soft_radius, cache_dir=cache_dir)
        if self.use_fs:
            logging.info(f"Fitting feature selection - {self.model_name}")
            selection = self.hp_use_fs_True__selection_type[self.use_fs_True__selection_type](self)
            all_train_idx = np.unique(np.concatenate([train_idx for train_idx, test_idx in cv]))
            selection.fit(X.iloc[all_train_idx], y[all_train_idx] > 0.5)  # threshold in case of soft labels
            # transform
            features_selection_X = selection.transform(X)
            new_columns = X.columns[selection.get_support()]
            X = pd.DataFrame(data=features_selection_X, index=X.index, columns=new_columns)

        logging.info(f"Fitting {self.model_name}")
        with parallel_backend("threading"):
            results = cross_validate(self.model,
                                     X, y,
                                     cv=cv,
                                     return_estimator=True, return_train_score=True,
                                     error_score='raise',
                                     n_jobs=-1)
        models = results['estimator']
        # TODO: need to see retval for both scores and normalize. was slow and didn't get to that
        self.results = self._get_results_to_save(X, models, cv, y_soft)
        return self

    def transform(self, X):
        return self.model.predict(X)

    def _negative_sampling(self, cv, y):
        new_cv = []
        for train_indices, test_indices in cv:
            y_train = y[train_indices]
            num_pos_samples = int(y_train.sum())
            neg_train_indices = train_indices[y_train == 0]
            new_neg_train_indices = np.random.choice(neg_train_indices, replace=False,
                                                     size=min(int(num_pos_samples * self.neg_ratio), len(neg_train_indices)))
            new_train_indices = np.concatenate((train_indices[y_train == 1], new_neg_train_indices))
            new_cv.append((new_train_indices, test_indices))
        return new_cv

    def _use_soft_labels(self, model):
        return model == 'catboost'

    def _get_results_to_save(self, X, models, cv, y_soft) -> Dict:

        results = {'y': y_soft, 'model_name': self.model_name, 'models': models ,'hp_dict':self.hp_dict}
        results = {'X_df': X, 'y': y_soft, 'model_name': self.model_name, 'models': models ,'hp_dict':self.hp_dict}
        results['train_idx'], results['test_idx'], results['probas'], \
        results['auc_scores'] = [], [], [], []
        if type(cv) != int:
            for i, (train_set, test_set) in enumerate(cv):

                results['train_idx'].append(train_set)
                results['test_idx'].append(test_set)
                probas = models[i].predict_proba(X)[:, 1]
                results['probas'].append(probas)
                results['auc_scores'].append(soft_auc(y_soft[test_set], probas[test_set]))

        return results

    def _calculate_test_scores(self, X, y_soft, cv, models):
        auc_scores = []
        for (train_indices, test_indices), model in zip(cv, models):
            X_test = X.iloc[test_indices]
            y_test = y_soft[test_indices]
            y_proba = model.predict_proba(X_test)
            auc_score = soft_auc(y_test, y_proba)
            auc_scores.append(auc_score)
        return auc_scores

    def predict(self, X):
        return self.results['models'][0].predict(X)  # TODO Fix this

    def predict_proba(self, X):
        return self.results['models'][0].predict_proba(X)  # TODO Fix this
