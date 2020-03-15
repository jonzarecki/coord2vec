import copy
from typing import Any, List, Union, Callable

from lagoon import Task, Stage

from coord2vec.pipelines.lagoon_utils.auto_stage import AutoStage
from coord2vec.pipelines.lagoon_utils.lambda_task import LambdaTask


class _ForMergeTask(LambdaTask):
    """
        Helper task for merging for results to dict of param -> ordered list of param results
        Should not be used outside this file
    """
    def __init__(self, iter_num: int, **kwargs):
        # no need for output functionality as this is an internally used task
        super().__init__(func=lambda: 1, func_output_names=[], **kwargs)
        self.iter_num = iter_num

    def run(self) -> Any:
        for_output_names = [p for p in set('~'.join(inp_name.split('~')[:-1]) for inp_name in self.input.keys()) if len(p)!=0]
        output_dict = {}
        for o_name in for_output_names:
            output_dict[o_name] = [self.input[f"{o_name}~{i}"] for i in range(self.iter_num)]

        return output_dict


class ForInputTask(LambdaTask):
    """
        Helper task for wrapping an input LambdaTask which will be used as a "for" iterator
        adds:
             iter_params: the name of the params used inside the for
             iter_num: the number of elements inside the iterator returned by the task (which the for iterates)
    """
    def __init__(self, func: Callable, iter_params: List[str], iter_num: int, **kwargs):
        super().__init__(func=func, func_output_names=["itr"], **kwargs)
        self.iter_num = iter_num
        self.iter_params = iter_params


class ForCalcTask(LambdaTask):
    """
        Helper task for wrapping an input LambdaTask which will be used as the body of a "for"
        adds:
             dependencies: list of dependencies to the task, these will be added for each duplicated "for" body task
    """
    def __init__(self, func: Callable, func_output_names: Union[List[str], str], dependencies: List[Task], *args, **kwargs):
        super().__init__(func=func, func_output_names=func_output_names, **kwargs)
        self.dependencies = dependencies
        self.curr_itr_num = None

    def set_curr_itr_num(self, curr_itr_num: int):
        self.curr_itr_num = curr_itr_num
        self.func_output_names = [f"{p}~{self.curr_itr_num}" for p in self.func_output_names]

    def run(self) -> Any:
        output_dict = super().run()
        if self.curr_itr_num is not None:
            pass_num_dict = {f"{n}~{self.curr_itr_num}": output_dict[n] for n in self.pass_input_names}
            new_out_dict = {n: output_dict[n] for n in self.func_output_names}
            new_out_dict.update(pass_num_dict)

            return new_out_dict

        else:
            return output_dict


def define_for_dependencies(S_program: Stage, for_calc_task: ForCalcTask,
                            for_input_task: ForInputTask, for_merge_task: LambdaTask) -> LambdaTask:
    """

    Args:
        S_program: The program stage
        for_calc_task: The ForCalcTask with the function containing the body of the "for"
        for_input_task: The ForInputTask which returns the iterator for the "for"
        for_merge_task: The LambdaTask which merges all for output from "p": [p0, p1, p2] to one output

    Returns:
        $for_merge_task, as it is the end of the chain
    """
    # extract useful varaibles
    iter_num, iter_params = for_input_task.iter_num, for_input_task.iter_params
    calc_deps = for_calc_task.dependencies

    for_merge_to_list_task = _ForMergeTask(iter_num)

    for i in range(iter_num):  # define dependencies
        for_input_transform = LambdaTask(lambda _i, itr: itr[_i], iter_params, _i=i)  # same "itr" from ForInputTask

        S_program.add_dependency(for_input_task, for_input_transform)

        calc_task_copy = copy.deepcopy(for_calc_task)
        calc_task_copy.set_curr_itr_num(i)
        S_program.add_dependency(calc_deps + [for_input_transform], calc_task_copy)

        S_program.add_dependency(calc_task_copy, for_merge_to_list_task)

    S_program.add_dependency(for_merge_to_list_task, for_merge_task)

    return for_merge_task

