import logging
from datetime import datetime
from functools import wraps

from .timer import Timer


# logging.basicConfig(
#     filename="logFile.txt",
#     filemode="a",
#     format="%(asctime)s %(levelname)s-%(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
def timed_logged_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        print(f"Starting {func.__name__}  ", end="")
        res = func(*args, **kwargs)
        duration = timer.stop()

        # logger = logging.getLogger(self.logger_name)
        logger = logging

        logger.info(f"Finished {func.__name__} for in {int(duration)}s")
        return res
    return wrapper
