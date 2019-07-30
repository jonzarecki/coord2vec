import pandas as pd

from coord2vec import config
from coord2vec.evaluation.tasks import HousePricing
from coord2vec.models.baselines import *
from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder
from coord2vec.models.baselines.empty_model import EmptyModel


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
        # print(baseline_scores)

    return scores


if __name__ == '__main__':
    coord2vec = Coord2Vec(house_price_builder, n_channels=3, multi_gpu=False).load_trained_model(
        '../../../coord2vec/models/saved_models/norm_model.pt')
    coord2features = Coord2Features(house_price_builder).load_trained_model('')
    empty_model = EmptyModel()

    baselines = [empty_model, coord2features, coord2vec]
    fit_kwargs = {'cache_dir': config.CACHE_DIR}

    scores = compare_baselines(HousePricing, baselines, **fit_kwargs)

    print(scores)
