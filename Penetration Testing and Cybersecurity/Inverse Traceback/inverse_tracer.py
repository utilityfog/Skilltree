import asyncio
import functools
import inspect
import uuid
import sys

from contextvars import ContextVar

call_chain_id_var = ContextVar("call_chain_id", default=None)
call_chain_var = ContextVar("call_chain", default=[])

# Ignore this decorator because I most likely do not want to decorate every single method
# def log_if_part_of_chain(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         current_call_chain_id = call_chain_id_var.get()
#         call_chain = call_chain_var.get()
#         if current_call_chain_id is not None:
#             method_name = func.__name__
#             print(f"Executing {method_name} because it's part of the call chain.")
#             call_chain.append(method_name)
#             call_chain_var.set(call_chain)
#         result = func(*args, **kwargs)
#         if current_call_chain_id is not None:
#             call_chain.pop()
#             call_chain_var.set(call_chain)
#         return result
#     return wrapper

def trace_calls(frame, event, arg):
    """The most important method. Instruction for how sys.setprofile should log method calls."""
    # Only interested in call events
    if event != "call":
        return trace_calls
    
    # Fetch the context of the current execution path
    current_call_chain_id = call_chain_id_var.get()
    if current_call_chain_id is None:
        # If there's no active execution path, stop tracing this path
        return
    
    # Extract function or method name
    co = frame.f_code
    func_name = co.co_name
    if func_name in {'<module>', 'write'}:
        # Ignore module-level executions and write calls
        return trace_calls

    # Construct a method identifier
    filename = co.co_filename
    lineno = frame.f_lineno
    method_identifier = f"{func_name} in {filename}:{lineno}"

    # Log the call if part of the current execution path
    call_chain = call_chain_var.get()
    call_chain.append(method_identifier)
    call_chain_var.set(call_chain)
    print(f"Executing {method_identifier} because it's part of the call chain.")
    
    return trace_calls

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
    
# class AsyncMethodChainProxy:
#     def __init__(self, awaitable):
#         self._awaitable = awaitable
#         self._future = asyncio.Future()
#         self._future.set_result(awaitable)

#     def __await__(self):
#         result = yield from self._awaitable.__await__()
#         return self._wrap_result(result)

#     def _wrap_result(self, result):
#         if inspect.iscoroutine(result) or asyncio.isfuture(result):
#             return AsyncMethodChainProxy(result)
#         return result

#     def __getattr__(self, name):
#         async def _method(*args, **kwargs):
#             nonlocal name
#             result = await self  # Await the current awaitable and get the result
#             next_attr = getattr(result, name)
#             if asyncio.iscoroutinefunction(next_attr) or callable(next_attr):
#                 # If it's a coroutine function, call it and return a new proxy if needed
#                 called = next_attr(*args, **kwargs)
#                 if asyncio.iscoroutine(called) or asyncio.isfuture(called):
#                     return AsyncMethodChainProxy(called)
#                 return called
#             else:
#                 return next_attr
#         return _method

class AsyncMethodChainProxy:
    def __init__(self, obj):
        self._obj = obj
    
    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        
        if asyncio.iscoroutinefunction(attr):
            async def _method(*args, **kwargs):
                result = await attr(*args, **kwargs)
                # Conditional logging based on the result type
                if inspect.isclass(type(result)) or inspect.ismethod(result) or callable(result):
                    print(f"Dynamic condition met for: {attr.__name__}")
                return result
            return _method
        else:
            return attr

# def entry_point(method):
#     @functools.wraps(method)
#     def wrapper(*args, **kwargs):
#         # Initialize a new call chain for this entry point
#         call_chain_id = uuid.uuid4()
#         call_chain_id_var.set(call_chain_id)
#         call_chain_var.set([method.__name__])

#         result = method(*args, **kwargs)

#         if inspect.isclass(type(result)) or inspect.ismethod(result) or callable(result):
#             result = Proxy(result)

#         call_chain_id_var.set(None)
#         call_chain_var.set([])

#         return result
#     return wrapper

# def entry_point(method):
#     @functools.wraps(method)
#     def wrapper(*args, **kwargs):
#         # Initialize a new call chain for this entry point
#         call_chain_id = uuid.uuid4()
#         call_chain_id_var.set(call_chain_id)
#         call_chain_var.set([method.__qualname__])  # Using __qualname__ for more descriptive method names

#         # Start tracing
#         old_trace = sys.gettrace()
#         sys.settrace(trace_calls)
#         try:
#             result = method(*args, **kwargs)
#         finally:
#             # Stop tracing and reset context
#             sys.settrace(old_trace)
#             call_chain_id_var.set(None)
#             call_chain_var.set([])

#         return result
#     return wrapper

def entry_point(method):
    """Entry Point Decorator that uses sys.setprofile to conditionally log method calls that are part of the initial method call's execution chain"""
    if asyncio.iscoroutinefunction(method):
        @functools.wraps(method)
        async def async_wrapper(*args, **kwargs):
            # Initialize a new call chain for this entry point
            call_chain_id = uuid.uuid4()
            call_chain_id_var.set(call_chain_id)
            call_chain_var.set([method.__qualname__])  # For more descriptive method names

            # Start tracing
            old_trace = sys.getprofile()
            sys.setprofile(trace_calls)
            try:
                result = await method(*args, **kwargs)  # Await the coroutine
            finally:
                # Stop tracing and reset context
                sys.setprofile(old_trace)
                call_chain_id_var.set(None)
                call_chain_var.set([])

            return result
        return async_wrapper
    else:
        @functools.wraps(method)
        def sync_wrapper(*args, **kwargs):
            # Initialize a new call chain for this entry point
            call_chain_id = uuid.uuid4()
            call_chain_id_var.set(call_chain_id)
            call_chain_var.set([method.__qualname__])

            # Start tracing
            old_trace = sys.getprofile()
            sys.setprofile(trace_calls)
            try:
                result = method(*args, **kwargs)
            finally:
                # Stop tracing and reset context
                sys.setprofile(old_trace)
                call_chain_id_var.set(None)
                call_chain_var.set([])

            return result
        return sync_wrapper