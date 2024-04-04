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
        current_call_chain_id = call_chain_id_var.get()
        call_chain = call_chain_var.get()

        method_name = f"{self.owner.__class__.__name__}.{self.method.__name__}"
        print(f"Executing {method_name} because it's part of the call chain.")

        # Nested / Chained method handling
        if current_call_chain_id is not None:
            call_chain.append(method_name)
            call_chain_var.set(call_chain)
        else:
            # This is a new call chain, possibly due to direct method invocation outside the initial entry point
            call_chain_id = uuid.uuid4()
            call_chain_id_var.set(call_chain_id)
            call_chain_var.set([method_name])

        result = self.method(self.owner, *args, **kwargs)

        if inspect.isclass(type(result)) or inspect.ismethod(result) or callable(result):
            result = Proxy(result)

        # Reset call chain only if this was the initiation of a new chain
        if current_call_chain_id is None:
            call_chain_id_var.set(None)
            call_chain_var.set([])

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

        result = method(*args, **kwargs)

        if inspect.isclass(type(result)) or inspect.ismethod(result) or callable(result):
            result = Proxy(result)

        call_chain_id_var.set(None)
        call_chain_var.set([])

        return result
    return wrapper