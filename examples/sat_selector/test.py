import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


def foo():
    cnt = 0
    while (True):
        # print("Hello")
        cnt += 1


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


try:
    with time_limit(10):
        foo()
except TimeoutException as e:
    print("Timed out!")
