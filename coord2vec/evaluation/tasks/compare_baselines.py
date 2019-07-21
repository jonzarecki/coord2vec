import pandas as pd

from coord2vec import config
from coord2vec.evaluation.tasks import HousePricing
from coord2vec.models.baselines import *
from coord2vec.feature_extraction.features_builders import example_features_builder


def compare_baselines(task, baselines, **fit_kwargs):
    task_handler = task()
    coords, features, y = task_handler.get_data()

    scores = []
    for model in baselines:
        # get embeddings
        X = model.predict(coords)
        X = pd.DataFrame(X, columns=[f'cord2vec{i}' for i in range(X.shape[1])])
        X = pd.concat([X, features], axis=1)

        # get results
        task_handler.fit(X, y)
        baseline_scores = task_handler.scores()

        scores.append(baseline_scores)

    return scores


if __name__ == '__main__':
    coord2vec = Coord2Vec(example_features_builder, n_channels=3).load_trained_model(
        '../../../coord2vec/models/saved_models/first_model.pt')
    baselines = [coord2vec, Coord2Featrues().load_trained_model('')]
    fit_kwargs = {'cache_dir': config.CACHE_DIR}

    scores = compare_baselines(HousePricing, baselines, **fit_kwargs)

    print(scores)
