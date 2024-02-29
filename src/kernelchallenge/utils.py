import time
from functools import wraps
from typing import Callable


def timeit(func: Callable) -> Callable:
    """ Decorator for timing function execution time """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        time0 = time.time()
        output = func(*args, **kwargs)
        timef = time.time()
        print(f"Function {func.__name__} took {timef - time0:.2f} seconds")
        return output

    return timeit_wrapper
