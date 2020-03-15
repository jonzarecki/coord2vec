from unittest import TestCase
import numpy as np
from bokeh.models import LayoutDOM

from coord2vec.evaluation.visualizations.bokeh_plots import bokeh_pr_curve_from_y_proba


class TestBokeh_pr_curve(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_pred = np.random.choice((0, 1), size=10)
        cls.y_true = np.random.choice((0, 1), size=10)

    def test_bokeh_pr_curve(self):
        fig = bokeh_pr_curve_from_y_proba(self.y_pred, self.y_true, legend='Zarecki is special')
        self.assertIsInstance(fig, LayoutDOM)
