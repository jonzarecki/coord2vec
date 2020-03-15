import os
from typing import Any, List, Union

from lagoon import Task
from lagoon.io.file import lagoon_open

class ExprSaverTask(Task):
    def __init__(self, save_path: str, save_params: Union[List[str], str], **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.save_params = save_params

    def run(self):
        # TODO: input names can also come from pipelines variable/constants
        with lagoon_open(os.path.join(self.save_path, "results.pickle"), "w") as f:
            f.save({name: self.input[name] for name in self.save_params})


