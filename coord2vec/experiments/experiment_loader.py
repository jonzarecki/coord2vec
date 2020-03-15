import datetime
import logging
import os
import pickle
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

from coord2vec.common.parallel.multiproc_util import parmap

from tqdm import tqdm

from coord2vec.config import BUILDING_RESULTS_DIR


def load_experiment_results(results_dir: str = BUILDING_RESULTS_DIR, exp_datetime: datetime.datetime = None) -> Dict:
    """
    load an experiment results dictionary from a directory
    Args:
        results_dir: the directory where all the experiments are in
        exp_datetime: optional. if a specific experiment is wanted. if None, will return the latest experiment

    Returns:
        a dictionary with the experiment results
    """
    all_experiments = os.listdir(results_dir)
    if exp_datetime is None:
        paths = [os.path.join(results_dir, experiment) for experiment in all_experiments]
        exp_dir = max(paths, key=os.path.getctime)
    else:
        date_str = exp_datetime.isoformat('_', 'seconds')
        exp_dir = os.path.join(results_dir, date_str)

    results_path = os.path.join(exp_dir, 'results.pickle')
    logging.info(f"reading experiment from {os.path.basename(exp_dir)}")
    with open(results_path, 'rb') as file:
        results = pickle.load(file)
        return results


def load_separate_model_results(results_dir: str = BUILDING_RESULTS_DIR,
                                exp_datetime: datetime.datetime = None) -> Dict:
    all_experiments = os.listdir(results_dir)
    if exp_datetime is None:
        paths = [os.path.join(results_dir, experiment) for experiment in all_experiments]
        exp_dir = max(paths, key=os.path.getctime)
    else:
        date_str = exp_datetime.isoformat(' ', 'seconds')
        exp_dir = os.path.join(results_dir, date_str)

    if not os.path.exists(exp_dir):
        logging.warning("Experiment path doesn't exist")
        return {'model_results': []}

    print(f"loading from {exp_dir}")

    all_files = os.listdir(exp_dir)
    chunk_size = 10
    # TODO: ugly cause pickles are large (1800 models takes 2 hours to load)

    def load_chunk(i):
        all_results = []
        for f in all_files[min(len(all_files),i*chunk_size):min(len(all_files), (i+1)*chunk_size)]:
            with open(os.path.join(exp_dir, f, 'results.pickle'), 'rb') as file:
                res = pickle.load(file)['kfold_results']
                # TODO: we might want it in the future
                # del res['X_df']
                # del res['y']
                all_results.append(res)
        return all_results

    def load_file(fpath):
        with open(os.path.join(exp_dir, fpath, 'results.pickle'), 'rb') as file:
            res = pickle.load(file)['kfold_results']
            # TODO: we might want it in the future
            # del res['X_df']
            # del res['y']
        return res

    print(f"number of files is {len(all_files)}")

    all_results = parmap(load_file, all_files, chunk_size=5,
                                 use_tqdm=True, desc="Opening all model results", unit="model")
    return {'model_results': all_results}


def get_all_existing_model_names(results_dir: str = BUILDING_RESULTS_DIR,
                            exp_datetime: datetime = None) -> List[str]:
    all_model_results = load_separate_model_results(results_dir, exp_datetime)['model_results']
    if len(all_model_results) == 0:
        return []
    all_model_names = [results['model_name'] for results in all_model_results]
    return all_model_names
