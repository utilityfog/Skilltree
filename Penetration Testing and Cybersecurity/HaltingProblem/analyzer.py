import inspect
import re
import signal

from contextlib import contextmanager

class TimeoutException(Exception):
    """Exception to raise on a timeout"""

@contextmanager
def time_limit(seconds):
    """Raise TimeoutException in the context after a certain time limit."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def analyze_halt(Callable, Input=None):
    """Method that analyzes whether a given callable will halt or loop when processing input by examining its source code.
    
    Args:
        Callable: A callable function that takes a single argument.
        Input: The input to pass to the callable function (unused in this static analysis version).

    Returns:
        True if the function appears to halt, False if it appears to loop indefinitely based on source code analysis.
    """
    try:
        source_code = inspect.getsource(Callable)
        print(f"Source code of {Callable.__name__}:\n{source_code}")

        # Look for simple infinite loop patterns e.g., "while True:" or "for _ in itertools.count():"
        if re.search(r"while True:", source_code) or re.search(r"for _ in itertools.count\(", source_code):
            return False  # Predicts looping indefinitely
        # Additional patterns can be added here

        return True  # No obvious infinite loops found

    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"
    
def test_loop(Input: bool):
    while True:
        pass
    
def test_halt(Input):
    pass