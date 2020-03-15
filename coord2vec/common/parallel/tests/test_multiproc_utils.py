from unittest import TestCase

from coord2vec.common.parallel.multiproc_util import parmap


def f(x):
    return x+1


class TestParmap(TestCase):
    def test_parmap_does_calculation_correctly(self):
        res = parmap(f, range(500))
        self.assertListEqual(res, [x+1 for x in range(500)])

    def test_parmap_does_calculation_correctly_with_chunks(self):
        res = parmap(f, range(500), chunk_size=5)
        self.assertListEqual(res, [x+1 for x in range(500)])
