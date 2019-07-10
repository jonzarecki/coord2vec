import unittest
import h2o
from coord2vec.evaluation.tasks.task_handler import TaskHandler


class TestTaskHandler(unittest.TestCase):
    @unittest.mock.patch.multiple(TaskHandler, __abstractmethods__=set())
    def test_h2o_runs(self):
        h2o.demo("glm")




if __name__ == '__main__':
    unittest.main()
