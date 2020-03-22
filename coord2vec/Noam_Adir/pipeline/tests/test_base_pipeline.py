from unittest import TestCase

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

from coord2vec.Noam_Adir.pipeline.base_pipeline import *


class TestPipeline(TestCase):
    def setUp(self) -> None:
        self.features_bundle = karka_bundle_features
        self.use_full_dataset = False
        self.batch_size = 1000
        self.clean_funcs = ALL_FILTER_FUNCS_LIST
        self.cleaned_features = extract_and_filter_csv_data(self.clean_funcs, use_full_dataset=self.use_full_dataset)

    def test_extract_and_filter_csv_data(self):
        unfiltered_features = get_csv_data(use_full_dataset=False)
        features = unfiltered_features.copy()
        for clean_func in self.clean_funcs:
            features = clean_func(unfiltered_features)
        self.assertEqual(len(self.cleaned_features), len(features))

    def test_extract_geographical_features(self):
        all_features = extract_geographical_features(self.cleaned_features, batch_size=self.batch_size,
                                                     features_bundle=self.features_bundle)
        self.assertEqual(self.cleaned_features.shape[0], all_features.shape[0])
        # check that the geographic features extraction does not change the non geographic features
        self.assertEqual(self.cleaned_features["floor"].values.all(), all_features["floor"].values.all())
        # check number of features is number of features in csv + number of geographic features + 1 for "coord_id"
        self.assertEqual(all_features.shape[1],
                         self.cleaned_features.shape[1] + len(create_building_features(self.features_bundle)) + 1)

    def test_all_pipe(self):
        use_full_dataset = False
        batch_size = 1000
        n_cat_iter = 150
        models = [LinearRegression(),
                  CatBoostRegressor(iterations=n_cat_iter, learning_rate=1, depth=3)]
        cleaned_features = extract_and_filter_csv_data(use_full_dataset=use_full_dataset)
        all_features = extract_geographical_features(cleaned_features, batch_size=batch_size,
                                                     features_bundle=self.features_bundle)
        training_cache = train_models(models=models, all_features=all_features)
        plot_scores(training_cache)

        # assert pipeline is running
        self.assertEqual(0, 0)

