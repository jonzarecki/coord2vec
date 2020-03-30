import os
import random
import sys
import warnings

import numpy
from bokeh.io import curdoc
from bokeh.models.widgets import Panel, Tabs

file_path = os.path.realpath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path)))))
sys.path.append(root_path)
# sys.path.append(os.path.join(root_path, 'coord2vec'))

from coord2vec.evaluation.visualizations.bokeh_server.pilot_dashboard import PilotDashboard
from coord2vec.common.itertools import flatten
from coord2vec.feature_extraction.osm.postgres_feature_factory import PostgresFeatureFactory
from coord2vec.config import COORD2VEC_DIR_PATH
from coord2vec.common.parallel import multiproc_util


def main():
    random.seed(42)
    numpy.random.seed(42)
    multiproc_util.force_serial = True
    task = None


    # get the feature names
    # TODO commented these 2 rows outside
    # feature_factory = PostgresFeatureFactory(task.embedder.features, input_gs=None)
    # all_feat_names = flatten([feat.feature_names for feat in feature_factory.features])

    bokeh_dir = os.path.join(COORD2VEC_DIR_PATH, 'evaluation', 'pilot_toy_data')
    model_results_path = os.path.join(bokeh_dir, 'model_results')
    pilot_results_path = os.path.join(bokeh_dir, 'pilot_results')

    # create bokeh tabs
    tabs = []
    tabs += [Panel(child=PilotDashboard(model_results_path, pilot_results_path).main_panel, title="Pilot")]
    # tabs += [Panel(child=FeatureDashboard(all_feat_names).main_panel, title="Feature Exploration")]
    # tabs += [Panel(child=BuildingTaskDashboard(task).main_panel, title="Task")]  # TODO: doesn't work for some reason

    tabs = Tabs(tabs=tabs)
    curdoc().add_root(tabs)


warnings.filterwarnings("ignore")
main()

#  bokeh serve /data/home/morpheus/coord2vec_brus/coord2vec/evaluation/visualizations/bokeh_server --address=0.0.0.0 --port=8200 --allow-websocket-origin=*:8200