import pandas as pd

from coord2vec import config
from coord2vec.evaluation.tasks import HousePricing
from coord2vec.models.baselines import *


def compare_baselines(task, baselines, **fit_kwargs):
    task_handler = task()
    coords, features, y = task_handler.get_data()

    scores = []
    for baseline in baselines:
        # get embeddings
        model = baseline()
        model.fit(**fit_kwargs)
        X = model.predict(coords)
        X = pd.concat([X, features], axis=1)

        # get results
        task_handler.fit(X, y)
        baseline_scores = task_handler.scores()

        scores.append(baseline_scores)

    return scores

if __name__ == '__main__':
    baselines = [Coord2Vec, Coord2Featrues]
    fit_kwargs = {'cache_dir': config.CACHE_DIR}

    scores = compare_baselines(HousePricing, baselines, **fit_kwargs)

    print(scores)
