import copy
import inspect
from abc import ABC
from functools import wraps
from typing import List, Dict, Tuple, Set

# def verify_init_params(func):
#     def verify_wrapper_func(*args, **kwargs):  # problematic as we don't know the name in *args
#
#
#         return func(*args, **kwargs)


class MetaModule(ABC):
    """
    Abstract class for meta modules.
        Define all hyper params as class variables
        All hyper-parameters class variables should start with hp_

        hps that assume a specific value of another hp will encode that in their name:
        i.e. if param2 is only relevant when param1=='a' then
             its name will be hp_param1_a__param2 = ..

             each condition like that is separated by '__'

    Currently there is a problem to inherit a class which inherits MetaModule (look at verify hp)
    TODO: still does not verify params passed in __init__ are defined in the hyperparams
            Need to define wrapper to __init__
    """

    def __init__(self, **kwargs):
        self.init_args = self.verify_hyperparameters()
        assert len(kwargs) == 0, f"all params should be caught by inheriting classes: \n {kwargs}"

    @staticmethod
    def save_passed_params_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert len(args) == 1, "all params should be passed as kwargs"
            self = args[0]
            assert isinstance(self, MetaModule), "args[0] should be self"
            self.passed_kwargs = kwargs
            return func(*args, **kwargs)

        return wrapper

    def clear_all_hp_variables(self):
        for p in self.init_args:
            del self.__dict__[f'hp_{p}']

    @classmethod
    def verify_hyperparameters(cls):
        # verify all param values are immutable or functions
        all_hp_keys = []
        for key, val in cls.__dict__.items():
            if 'hp_' == key[:3]:
                all_hp_keys.append(key[3:])
                if isinstance(val, list):
                    all_vals = val
                elif isinstance(val, dict):
                    all_vals = val.values()
                else:
                    raise AssertionError(f"invalid type {type(val)}")

                assert isinstance(key, (int, float, str)), f"hp {key} cannot be of type {type(key)}" \
                                                           f"should be a primitive"  # can possibly change

                for v in all_vals:
                    assert isinstance(v, (int, float, str)) or v is None or callable(v), \
                        f"hp {key}'s value cannot be of type {type(v)}" \
                        f"cannot ensure it is immutable"

        # verify all hp are defined in inheriting class
        init_args = list(inspect.signature(cls.__init__, follow_wrapped=True).parameters.keys())
        init_args.remove('self')  # won't work in class that inherit classes that inherit MetaModule
        init_args.remove('kwargs')  # won't work in class that inherit classes that inherit MetaModule

        assert len(set(init_args).symmetric_difference(set(all_hp_keys))) == 0, \
            "all init params should be defined as hp, and in reverse\n" \
            f"init_args: \t\t {set(init_args)}\n" \
            f"all_hp_keys: \t {set(all_hp_keys)}"

        return init_args

    @staticmethod
    def extract_hp_param_name(key):
        return key.split('__')[-1]  # last __ means param name

    @staticmethod
    def is_all_dependencies_ok(params, curr_key_deps):
        """
            Checks if all deps (defined in $curr_key_deps) are valid in $params
            curr_key_deps were defined using the dependency syntax discussed in the class documentation.
        """
        return all(str(params[k]) == curr_key_deps[k] for k in curr_key_deps.keys())

    @staticmethod
    def get_key_dependencies(key: str, all_keys: List[str]):
        return [k for k in all_keys if MetaModule.extract_hp_param_name(k) in key and k != key]

    @classmethod
    def get_all_hyperparameter_combinations(cls) -> List[dict]:
        """

        Returns:
            Returns all possible combinations to the model a list of dicts
            afterwards it can be use as MetaAA(*retval[0])
        """
        cls.verify_hyperparameters()
        possible_hp_lists = []  # list of (key,val) for each key for all it's possible values
        possible_hp_dict = {}
        all_hp_keys = []
        for key, val in cls.__dict__.items():
            if 'hp_' != key[:3]:
                continue
            all_vals = None
            if isinstance(val, list):
                all_vals = val
            elif isinstance(val, dict):
                all_vals = list(val.keys())

            key_name = key[3:]
            all_hp_keys.append(key_name)
            possible_hp_lists.append([(key_name, v) for v in all_vals])
            possible_hp_dict[key_name] = all_vals

        all_opts = cls.build_combinations(possible_hp_dict=possible_hp_dict)
        return [dict(comb) for comb in all_opts]

    @classmethod
    def build_combinations(cls, possible_hp_dict: Dict[str, List[object]]) -> Set[Tuple]:
        """
        Build all possible combinations as list of tuples (of variable length)
        uses recusion
        Args:
            possible_hp_dict: dict of hp_param name to list of values

        Returns:
            List of tuple with all combs
        """
        all_keys = possible_hp_dict.keys()
        clean_keys = [k for k in all_keys if '__' not in k]
        if len(possible_hp_dict) == 0 or len(clean_keys) == 0:
            return set()  # no keys left, or only keys with bad deps (no clean-keys)

        all_opt_set = set()
        filt_key = clean_keys[0]
        all_filt_key_vals = possible_hp_dict[filt_key]

        if len(all_keys) == 1:  # only one key left
            assert len(clean_keys) == 1, "all keys are clean"
            filt_key = clean_keys[0]
            return set([((filt_key, v),) for v in all_filt_key_vals])

        possible_hp_dict = copy.copy(possible_hp_dict)
        possible_hp_dict.pop(filt_key)  # don't go over filt_key
        # TODO: unreadable AF
        for v in all_filt_key_vals:
            new_possible_hp_dict = {}  # a new dict for recursion
            new_keys_map = {}  # maps shortened names (without deps) to full names
            for key in possible_hp_dict:
                curr_cond = f"{filt_key}_{str(v)}__"
                # running example for: filt_key = param1, v = a, key= param0_0__param1_a__param2
                if filt_key in key and curr_cond not in key:  # dependency is bad (e.g. param1_b)
                    continue
                else:
                    if curr_cond in key:
                        idx = key.index(filt_key)  # idx of param1
                        cleared_key = f"{key[:idx]}{key[(idx+len(curr_cond)):]}"  # param0_0__param2
                        new_keys_map[cleared_key] = key  # maps short to full
                        new_possible_hp_dict[cleared_key] = possible_hp_dict[key]  # all vals for key
                    else:
                        new_possible_hp_dict[key] = possible_hp_dict[key]  # as normal
            curr_key_opt_set = cls.build_combinations(new_possible_hp_dict)
            # map param-names back to full length (with deps)
            map_tpl = lambda tpl: [(new_keys_map.setdefault(pname, pname), val) for pname, val in tpl]
            if len(curr_key_opt_set) == 0:
                all_opt_set.add(((filt_key, v),))  # nothing more, add one with curr param (which is OK)
            else:
                all_opt_set.update({tuple([(filt_key, v)]+map_tpl(tpl)) for tpl in curr_key_opt_set})
        return all_opt_set

    @classmethod
    def init_from_yaml_config(cls, path: str):
        """
        Initializes a MetaModule using a .yaml file
        Converts yaml values to kwargs to initialize with
        Args:
            path: Path to yaml file

        Returns:
            Initialized MetaModule
        """
        # TODO: convert yaml to **kwargs
        # filter only to params defined by define_hyperparams_dicts()
        kwargs = {}
        return cls(**kwargs)
