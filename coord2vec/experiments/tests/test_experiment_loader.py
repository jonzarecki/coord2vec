from time import sleep
from unittest import TestCase

import pickle

import shutil
from datetime import datetime
from coord2vec.config import PROJECT_ROOT
import os

from coord2vec.experiments.experiment_loader import load_experiment_results


class TestLoadExperimentResults(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime1 = datetime.now()
        sleep(1)
        datetime2 = datetime.now()
        cls.test_path = f"{PROJECT_ROOT}/results/test"

        for i, cur_time in enumerate([cls.datetime1, datetime2]):
            sleep(0.1)
            cur_dir = os.path.join(cls.test_path, cur_time.isoformat('_', 'seconds'))
            os.makedirs(cur_dir)
            pickle_path = os.path.join(cur_dir, 'results.pickle')
            with open(pickle_path, "wb") as f:
                pickle.dump({'bla': f'bla{i}'}, f)

    def test_load_experiment_results_latest(self):
        res_dict = load_experiment_results(self.test_path)
        self.assertEqual(res_dict['bla'], 'bla1')

    def test_load_experiment_results_first(self):
        res_dict = load_experiment_results(self.test_path, self.datetime1)
        self.assertEqual(res_dict['bla'], 'bla0')

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_path)
