import time


# perhaps the timeit library would be more suitable

class Stopwatch:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.process_time()

    def lap(self):
        c_time = time.process_time()
        # Hopefully this simple way is equivalent to the CPU time as used in satzilla...

        return c_time - self.start_time

    def reset(self):
        self.start_time = None

    def stop(self):
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        self.start_time = None

    def get_elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        return time.process_time() - self.start_time

    def get_elapsed_time_str(self):
        return f"{self.get_elapsed_time():.2f} seconds"

    def get_elapsed_time_minutes(self):
        return self.get_elapsed_time() / 60

    def get_elapsed_time_minutes_str(self):
        return f"{self.get_elapsed_time_minutes():.2f} minutes"
