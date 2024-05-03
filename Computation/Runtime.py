import time
import functools
import numpy as np

def complexity_estimator(sizes):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            print(f"Running complexity estimator for '{func.__name__}' with sizes: {sizes}")
            times = []
            for size in sizes:
                # Setup the object for each size
                self.__init__('0' * size + '1' * size)  # Reinitialize with balanced 0s and 1s
                start_time = time.time()
                _ = func(self, *args, **kwargs)  # Call the original function
                times.append(time.time() - start_time)

            # You might want to apply more sophisticated analysis here
            # For simplicity, just print times:
            print("Times recorded:", times)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class RuntimeChecker:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        print(f"Execution time of '{self.func.__name__}': {end_time - start_time:.6f} seconds")
        return result
    
def log_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        print(f"Execution time of '{func.__name__}': {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def log_runtime_iterations(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args[0].operation_count = 0  # Assuming the first arg is always 'self' for an instance method
        result = func(*args, **kwargs)
        print(f"Runtime Complexity (operation count) of '{func.__name__}': {args[0].operation_count} operations")
        return result
    return wrapper
    
if __name__ == "__main__":
    # Example function to be decorated
    @RuntimeChecker
    def test_function(x):
        sum = 0
        for i in range(x):
            sum += i
        return sum

    # Call the decorated function
    print("Result:", test_function(10000))