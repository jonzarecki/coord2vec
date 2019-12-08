import multiprocessing
import pickle as pkl
import traceback
import time
from os.path import basename
from typing import Dict

sleeptime = 0.5
total_waiting_time = 250
par_loaded_objs = list()  # global list for all waiting objs


class ParallelLoad(object):
    def __init__(self, loading_fun=None, pkl_path=None):
        self.model = None
        self.return_queue = None
        self.manager = multiprocessing.Manager()
        self.ret_dict = self.manager.dict()
        par_loaded_objs.append(self)
        if pkl_path is not None:
            self.pkl_path = pkl_path
            self.p = multiprocessing.Process(args=(self.pkl_path, self.ret_dict),
                                             target=lambda pth, d: d.__setitem__(1, self.load_pickle(pth)))
        else:  # loading_fun is not None
            self.p = multiprocessing.Process(args=(loading_fun, self.ret_dict),
                                             target=lambda f, d: d.__setitem__(1, f()))
        self.p.start()

    @property
    def obj(self):
        if self.model is None:
            counter = 0
            while len(self.ret_dict) == 0:
                time.sleep(sleeptime)
                counter += 1
                if counter >= total_waiting_time / sleeptime:
                    break
            if len(self.ret_dict) == 0:
                traceback.print_exc()
                raise AssertionError(f"ParallelLoad failed to load object after {total_waiting_time} seconds")
            self.model = self.ret_dict[1]
            self.p, self.return_queue = None, None
            self.manager.shutdown()
            self.manager = None
        return self.model

    def is_finished(self):
        return self.model is not None

    @staticmethod
    def load_pickle(pkl_path: str) -> Dict:
        try:
            sa = time.time()
            pkl_fname = basename(pkl_path)
            print("loading " + pkl_path)
            with open(pkl_path, 'rb') as f:
                retval = pkl.load(f)
        except Exception:
            traceback.print_exc()
            raise
        print(f"pickle {pkl_fname} load time: {int(time.time() - sa)}s")
        return retval

    @staticmethod
    def wait_for_all_loads():
        print("waiting for all loads . . .     ", end="", flush=True)
        for par_load in par_loaded_objs:
            a = par_load.obj  # read (wait) for all loads
        print("Done")
