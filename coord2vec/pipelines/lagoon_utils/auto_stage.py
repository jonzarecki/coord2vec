from typing import Union, List

from lagoon import Stage, Task

from coord2vec.pipelines.lagoon_utils.lambda_task import LambdaTask


class AutoStage(Stage):
    def __init__(self, name: str, **kwargs):
        super().__init__(name,  **kwargs)
        self.output_param_to_task = dict()

    def update_output_params(self, task):
        # TODO: kind-of ugly, uses internal _dict_graph
        if isinstance(task, LambdaTask) and task not in self._dict_graph:
            for output_param in (task.pass_input_names + task.func_output_names):
                self.output_param_to_task[output_param] = task

    def add_auto(self, task: LambdaTask):
        relevant_connections = set()
        for input_param in task.func_input_names:
            if input_param in self.output_param_to_task:
                relevant_connections.add(self.output_param_to_task[input_param])
            else:
                pass  # can come from pipelines variable
                # raise AssertionError(f"input {input_param} not presented before")

        if len(relevant_connections) == 0:
            self.add(task)
        else:
            self.add_dependency(list(relevant_connections), task)

    def add_dependency(
        self, current_task: Union[Task, List[Task]], next_task: Union[Task, List[Task]]
    ) -> "Stage":
        if not isinstance(current_task, list):
            current_task = [current_task]
        if not isinstance(next_task, list):
            next_task = [next_task]

        for task in (next_task + current_task):
            self.update_output_params(task)    # will try for all NEW tasks

        return super(AutoStage, self).add_dependency(current_task, next_task)

def add_to_DAG(task: LambdaTask, s: AutoStage):
    s.add_auto(task)