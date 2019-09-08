from unittest import TestCase
import numpy as np

from coord2vec.common.mtl.metrics.reconstruction import DistanceCorrelation


class TestReconstruction_correlation(TestCase):
    def test_reconstruction_correlation(self):
        embedding = np.random.rand(100, 128)
        features = np.random.rand(100, 16)
        correlation = DistanceCorrelation().compute(embedding, features)
        self.assertGreaterEqual(correlation, -1)
        self.assertGreaterEqual(1, correlation)
