from coord2vec.evaluation.tasks.task_handler import TaskHandler

import numpy as np
import pandas as pd


def get_embeddings(coords):
    return np.random.rand(len(coords), 100)


class HousePricing(TaskHandler):
    """
    House pricing prediction task for house prices in Beijing.

    """
    def get_data(self):
        df = pd.read_csv(r"Housing price in Beijing.csv", encoding="latin").iloc[:50]
        df['coord'] = df.apply(lambda row: tuple(row[['Lng', 'Lat']].values), axis=1)

        embeddings = get_embeddings(df['coord'].values)
        prices = df['price'].values

        return embeddings, prices

if __name__ == '__main__':
    hp = HousePricing()
    X, y = hp.get_data()
    hp.fit(X, y)
    scores = hp.scores()
    print(scores)
