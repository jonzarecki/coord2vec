from sklearn.base import BaseEstimator
import numpy as np


class BaselineModel(BaseEstimator):
    def score_samples(self, features_df):
        assert 'number_of_building_0m' in features_df.columns
        assert 'area_of_self_0m' in features_df.columns
        assert 'building_scores_avg_0m' in features_df.columns
        assert 'building_scores_max_0m' in features_df.columns

        num_buildings = features_df['number_of_building_0m']
        area = features_df['area_of_self_0m']
        scores_max = features_df['building_scores_max_0m']
        scores_avg = features_df['building_scores_avg_0m']

        weights = [1, 1, 1, 1]
        scores = weights[0] * 1 - (((num_buildings - 6) / 6) ** 2) + \
                 weights[1] * 1 - (((area - 8000) / 8000) ** 2) + \
                 weights[2] * scores_max + \
                 weights[3] * scores_avg
        return scores.values  # low is more anomalous

    def fit(self, *args):
        pass
