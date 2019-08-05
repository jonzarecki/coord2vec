from unittest import TestCase
import numpy as np

from coord2vec.evaluation.intristic_metrics.reconstruction import reconstruction_correlation


class TestReconstruction_correlation(TestCase):
    def test_reconstruction_correlation(self):
        embedding = np.random.rand(100, 128)
        features = np.random.rand(100, 16)
        correlation = reconstruction_correlation(embedding, features)
        self.assertGreaterEqual(correlation, -1)
        self.assertGreaterEqual(1, correlation)
