import time

# perhaps the timeit library would be more suitable

class Stopwatch:

    def start(self):
        self.start_time = time.process_time()

    def lap(self):
        c_time = time.process_time()
        # Hopefully this simple way is equivalent to the CPU time as used in satzilla...

        return c_time - self.start_time
