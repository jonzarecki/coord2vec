from unittest import TestCase
from coord2vec.config import TEST_CACHE_DIR
from coord2vec.models.baselines import Coord2Vec
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.feature_extraction.features_builders import example_features_builder
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset


class TestCoord2Vec(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding_dim = 16
        cls.coord2vec = Coord2Vec(example_features_builder, n_channels=3, embedding_dim=cls.embedding_dim,
                                  tb_dir='test')

    def test_fit_predict(self):
        # test fit
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=7, use_existing=False)
        dataset = TileFeaturesDataset(TEST_CACHE_DIR)
        self.coord2vec.fit(dataset, epochs=3, batch_size=1)

        # test predict
        coord_pred = [(34.8576548, 32.1869038), (34.8583825, 32.1874658)]
        embeddings = self.coord2vec.predict(coord_pred)
        self.assertTupleEqual(embeddings.shape, (len(coord_pred), self.embedding_dim))

    # def test_fit_with_sample(self):
    #     self.coord2vec.fit(cache_dir=TEST_CACHE_DIR, epochs=3, sample=True, sample_num=7)
