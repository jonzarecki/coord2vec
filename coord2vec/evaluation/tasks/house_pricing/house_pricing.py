import os
from typing import Callable, Tuple, Any
import pandas as pd

from coord2vec.evaluation.tasks.task_handler import TaskHandler


class HousePricing(TaskHandler):
    """
    House pricing prediction task for house prices in Beijing.

    """
    def get_data(self) -> Tuple[Any, Any, Any]:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), r"Housing price in Beijing.csv"),
                         encoding="latin").iloc[:50]
        df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)

        return df['coord'].values, df[['square', 'livingRoom', 'drawingRoom', 'kitchen', \
       'bathRoom', 'floor', 'buildingType', 'constructionTime',\
       'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',\
       'fiveYearsProperty', 'subway', 'district', 'communityAverage']], df['price'].values