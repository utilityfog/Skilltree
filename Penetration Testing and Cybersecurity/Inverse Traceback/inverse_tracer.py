import functools
import inspect
import uuid
from contextvars import ContextVar

call_chain_id_var = ContextVar("call_chain_id", default=None)
call_chain_var = ContextVar("call_chain", default=[])

class MethodWrapper:
    def __init__(self, method, owner):
        self.method = method
        self.owner = owner

    def __call__(self, *args, **kwargs):
        # Retrieve the current call chain and ID
        current_call_chain_id = call_chain_id_var.get()
        call_chain = call_chain_var.get()

        method_name = self.method.__name__
        print(f"Executing {method_name} because it's part of the call chain.")

        # Check for direct continuation in the call chain
        if current_call_chain_id is not None:
            call_chain.append(method_name)
            call_chain_var.set(call_chain)
        
        # Execute the wrapped method
        result = self.method(self.owner, *args, **kwargs)

        # Attempt to wrap the result if it's a class instance
        if inspect.isclass(type(result)):
            result = Proxy(result)

        if current_call_chain_id is not None:
            call_chain.pop()
            call_chain_var.set(call_chain)

        return result

class Proxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if callable(attr):
            return MethodWrapper(attr, self._target)
        return attr

def entry_point(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        # Initialize a new call chain for this entry point
        call_chain_id = uuid.uuid4()
        call_chain_id_var.set(call_chain_id)
        call_chain_var.set([method.__name__])

        # Execute the method, wrap the result if it's a class instance
        result = method(*args, **kwargs)
        if inspect.isclass(type(result)):
            result = Proxy(result)

        # Cleanup after execution
        call_chain_id_var.set(None)
        call_chain_var.set([])

        return result
    return wrapper
