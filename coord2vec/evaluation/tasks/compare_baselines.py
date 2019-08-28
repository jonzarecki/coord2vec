from typing import List

import pandas as pd
from sklearn.base import TransformerMixin

from coord2vec import config
from coord2vec.evaluation.tasks import HousePricing
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.models.baselines import *
from coord2vec.feature_extraction.features_builders import house_price_builder
from coord2vec.models.baselines.empty_model import EmptyModel


def compare_baselines(task:TaskHandler, baselines:List[TransformerMixin]):
    task_handler = task
    coords, additional_features, y = task_handler.get_data()

    scores = []
    for model in baselines:
        # get embeddings
        geo_features = model.transform(coords)
        geo_features = pd.DataFrame(geo_features, columns=[f'cord2vec{i}' for i in range(geo_features.shape[1])])
        X = pd.concat([geo_features, additional_features], axis=1)

        # get results
        task_handler.fit(X, y)
        baseline_scores = task_handler.scores()

        scores.append(baseline_scores)
        print(baseline_scores)

    return scores


if __name__ == '__main__':
    # create all the models
    coord2vec = Coord2Vec(house_price_builder, n_channels=3, multi_gpu=False).load_trained_model(
        '../../../coord2vec/models/saved_models/norm_model.pt')
    coord2features = Coord2Features(house_price_builder)
    empty_model = EmptyModel()

    baselines = [coord2features, coord2vec, empty_model]

    scores = compare_baselines(HousePricing(), baselines)

    print(scores)
