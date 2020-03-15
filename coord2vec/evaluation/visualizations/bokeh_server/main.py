import os
import random
import sys
import warnings

import numpy
from bokeh.io import curdoc
from bokeh.models.widgets import Panel, Tabs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


file_path = os.path.realpath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path)))))
sys.path.append(root_path)
# sys.path.append(os.path.join(root_path, 'coord2vec'))
from coord2vec.evaluation.visualizations.bokeh_server.feature_dashboard import FeatureDashboard
from coord2vec.common.itertools import flatten
from coord2vec.feature_extraction.osm.postgres_feature_factory import PostgresFeatureFactory
from coord2vec.config import BUILDINGS_FEATURES_TABLE, BUILDING_RESULTS_DIR
from coord2vec.feature_extraction.feature_bundles import create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.evaluation.visualizations.bokeh_server.building_dashboard import BuildingTaskDashboard
from xgboost import XGBClassifier
from coord2vec.common.parallel import multiproc_util


def main():
    random.seed(42)
    numpy.random.seed(42)
    multiproc_util.force_serial = True
    task = None


    # get the feature names
    feature_factory = PostgresFeatureFactory(task.embedder.features, input_gs=None)
    all_feat_names = flatten([feat.feature_names for feat in feature_factory.features])

    # create bokeh tabs
    tabs = []
    # tabs += [Panel(child=FeatureDashboard(all_feat_names).main_panel, title="Feature Exploration")]
    tabs += [Panel(child=BuildingTaskDashboard(task).main_panel, title="Task")]  # TODO: doesn't work for some reason

    tabs = Tabs(tabs=tabs)
    curdoc().add_root(tabs)


warnings.filterwarnings("ignore")
main()