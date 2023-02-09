__author__ = "T. Fornal"

import functools
import time

def timer(function):
    """Prints the execution time of given function."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = function(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\nExecution finished in {run_time: .2f}s")
        return value

    return wrapper
