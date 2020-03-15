import os
from typing import Tuple, Any

import pandas as pd

from coord2vec.evaluation.tasks.task_handler import TaskHandler


class HousePricing(TaskHandler):
    """
    House pricing prediction task for house prices in Seattle.
    https://www.kaggle.com/harlfoxem/housesalesprediction#kc_house_data.csv
    """

    def get_dataset(self) -> Tuple[Tuple[float, float], pd.DataFrame, Any]:
        # df = pd.read_csv(os.path.join(os.path.dirname(__file__), r"Housing price in Beijing.csv"),
        #                  encoding="latin").iloc[:10]
        # df['coord'] = df.apply(lambda row: tuple(row[['Lat', 'Lng']].values), axis=1)
        #
        #  return df['coord'].values, df[['square', 'livingRoom', 'drawingRoom', 'kitchen', \
        # 'bathRoom', 'floor', 'buildingType', 'constructionTime',\
        # 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',\
        # 'fiveYearsProperty', 'subway', 'district', 'communityAverage']], df['price'].values

        df = pd.read_csv(os.path.join(os.path.dirname(__file__), r"kc_house_data.csv"),
                         encoding="latin").iloc[:10]
        df['coord'] = df.apply(lambda row: tuple(row[['long', 'lat']].values), axis=1)
        coords = df['coord'].values
        features = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                       'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']]
        y = df['price'].values
        return coords, features, y
