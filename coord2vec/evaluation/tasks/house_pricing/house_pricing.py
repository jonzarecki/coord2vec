import os
from typing import Callable, Tuple, Any

from coord2vec.evaluation.tasks.task_handler import TaskHandler

import numpy as np
import pandas as pd

from coord2vec.feature_extraction.features_builders import house_price_builder


def get_random_embeddings(coords):
    return np.random.rand(len(coords), 100)


def get_example_house_pricing_features(coords):
    return house_price_builder.extract_coordinates(coords)

class HousePricing(TaskHandler):
    """
    House pricing prediction task for house prices in Beijing.

    """
    def get_data(self) -> Tuple[Any, Any, Any]:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), r"Housing price in Beijing.csv"), encoding="latin").iloc[:50]
        df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)

        return df['coord'].values, df[['square', 'livingRoom', 'drawingRoom', 'kitchen', \
       'bathRoom', 'floor', 'buildingType', 'constructionTime',\
       'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',\
       'fiveYearsProperty', 'subway', 'district', 'communityAverage']], df['price'].values


if __name__ == '__main__':
    hp = HousePricing()
    coords, feats, y = hp.get_data()
    X = get_example_house_pricing_features(coords)
    print("X,y ready, starting AutoML")
    X_all = pd.concat([X, feats], axis=1)
    hp.fit(X_all, y)
    scores = hp.scores()
    print(scores)
