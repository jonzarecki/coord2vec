from unittest import TestCase

from torch import nn

from coord2vec.config import CACHE_DIR
from coord2vec.models.baselines import Coord2Vec


class TestCoord2Vec(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        losses = [nn.L1Loss() for i in range(5)]
        cls.embedding_dim = 16
        cls.coord2vec = Coord2Vec(losses=losses, embedding_dim=cls.embedding_dim)

    def test_fit_predict(self):
        # test fit
        self.coord2vec.fit(cache_dir=CACHE_DIR, epochs=1)

        # test predict
        coord_pred = [(34.8576548, 32.1869038), (34.8583825, 32.1874658)]
        embeddings = self.coord2vec.predict(coord_pred)
        self.assertTupleEqual(embeddings.shape, (len(coord_pred), self.embedding_dim))

