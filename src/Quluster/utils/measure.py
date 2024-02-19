import time
from functools import wraps

def measure(func):
    @wraps(func)
    def wrapper(self, *args, **kargs):
        start_time = time.perf_counter()
        func(self, *args, **kargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        fn = func.__name__
        self.timing[fn] = execution_time if fn not in self.timing else self.timing[fn]+execution_time
    return wrapper