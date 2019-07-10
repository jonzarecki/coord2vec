import os
from typing import Callable, Tuple, Any

from coord2vec import config
from coord2vec.evaluation.tasks.task_handler import TaskHandler

import numpy as np
import pandas as pd

from coord2vec.feature_extraction.features_builders import house_price_builder
from coord2vec.models.baselines import *


class HousePricing(TaskHandler):
    """
    House pricing prediction task for house prices in Beijing.

    """

    def get_data(self) -> Tuple[Any, Any, Any]:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), r"Housing price in Beijing.csv"),
                         encoding="latin").iloc[:50]
        df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)

        return df['coord'].values, df[['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor', 'buildingType', 'constructionTime',
                                       'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
                                       'fiveYearsProperty', 'subway', 'district', 'communityAverage']], df[
                   'price'].values


if __name__ == '__main__':
    hp = HousePricing()
    coords, feats, y = hp.get_data()

    ############# get embeddings #############

    # X = get_example_house_pricing_features(coords)
    # print("X,y ready, starting AutoML")
    # X_all = pd.concat([X, feats], axis=1)

    coord2vec = Random()
    coord2vec.fit(cache_dir=config.CACHE_DIR, sample=False)
    # coord2vec.load_trained_model()
    X = coord2vec.predict(coords)
    ##########################################

    hp.fit(X, y)
    scores = hp.scores()
    print(scores)


    # according to https://www.kaggle.com/gavinmandias/beijing-housing-prices-analysing-and-predicting
    # final rmse can be 144.75553313583714
