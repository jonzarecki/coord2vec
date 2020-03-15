import inspect
import logging
import random
import re
from typing import Any, List, Union, Callable

import numpy
from lagoon import Task
from lagoon.utils.hash_utils import hash_value


class LambdaTask(Task):
    """
        Lagoon helper for decorating functions into pipelines tasks.

        Args:
            func: the function (lambda/other) we want to run in this task
            func_output_names: The name of the outputs returned by the function
            override_input_names: Optional parameter if an input param changes during the func and
                                    we want this change to pass onward
            seed: random seed set inside run()
    """
    def __init__(self, func: Callable, func_output_names: Union[List[str], str], override_input_names: List[str] = None,
                 seed=42, passed_vars={} ,**kwargs):
        super().__init__(**kwargs)
        self.passed_vars = passed_vars
        self.func = func
        self.seed = seed
        self.func_input_names = inspect.getfullargspec(func).args

        self.relevant_params = {inp_name: self.params[inp_name] for inp_name in self.func_input_names if inp_name in self.params}
        self.func_input_names = set(self.func_input_names).difference(self.params.keys())

        self.override_input_names = override_input_names if override_input_names is not None else []

        func_output_names = func_output_names if isinstance(func_output_names, list) else [func_output_names]

        # pass_input_names are names that pass from the input as is, so the need to appear in the input
        #       (unless they are in override_input_names)
        self.pass_input_names = [name for name in func_output_names if
                                 (name in self.func_input_names and name not in self.override_input_names)]
        self.func_output_names = [name for name in func_output_names if name not in self.pass_input_names]

    @property
    def name(self) -> str:
        class_name = self.__class__.__name__
        func_name = self.func.__name__
        if 'lambda' in func_name:
            func_name = re.sub('\s+', ' ', ''.join(inspect.getsourcelines(self.func)[0]).strip().replace('\n', '  '))
        obj_hash = hash_value(f"{class_name}{func_name}{self._get_params_hash()}")
        return f"{class_name}_{func_name}-{obj_hash}"

    def run(self) -> Any:
        numpy.random.seed(self.seed); random.seed(self.seed)

        # TODO: input names can also come from pipelines variable
        # if not all([inp_name in self.input for inp_name in self.func_input_names]):
        #     a=1
        self.input.update(self.passed_vars)

        assert all([inp_name in self.input for inp_name in self.func_input_names]), \
            f"""self.func_input_names {self.func_input_names} ---- self.input {self.input.keys()}
                {self.func.__code__}
            
            """
        func_input = {inp_name: self.input[inp_name] for inp_name in self.func_input_names}
        func_input.update(self.relevant_params)
        output_tpl = self.func(**func_input)

        if output_tpl is None:
            output_tpl = tuple()
        if not isinstance(output_tpl, tuple):
            output_tpl = (output_tpl, )

        assert len(output_tpl) == len(self.func_output_names), \
            f"""should return the same number of items: received {len(output_tpl)} expected out_names: {self.func_output_names}
                {self.func.__code__}
            
            """
        out_dict = {name: output_tpl[i] for i, name in enumerate(self.func_output_names)}
        pass_dict = {name: self.input[name] for name in self.pass_input_names}

        out_dict.update(pass_dict)

        return out_dict