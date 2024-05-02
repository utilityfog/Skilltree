from analyzer import analyze_halt, time_limit, TimeoutException

def Negate(Callable, **Inputs):
    print(f"Negate called with input: {Inputs}")
    if analyze_halt(Callable, **Inputs):
        # If Analyze_Halt says the given Callable will halt on a certain input, loop
        try:
            with time_limit(10):  # Timeout set for 10 seconds
                while(True):
                    pass
        except TimeoutException:
            return f"Negate loops because analyze_halt says {Callable} will halt."
    else:
        # If Analyze_Halt says the given Callable will loop on a certain input, halt
        return f"Negate halts because analyze_halt says {Callable} will loop."