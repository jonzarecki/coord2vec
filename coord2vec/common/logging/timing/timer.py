import time


class Timer:
    """
    Object used for timing for logs
    """
    def __init__(self):
        self.start_time = 0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            raise Exception("Timer is already running, please stop or reset before starting again")

    def stop(self):
        if self.running:
            self.running = False
            seconds_elapsed = time.time() - self.start_time
            self.start_time = 0
            return seconds_elapsed * 1000
        return 0

    def reset(self):
        self.running = False
        self.start_time = 0