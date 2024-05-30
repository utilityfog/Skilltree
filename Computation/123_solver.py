def sum_count_solver():
    """Receives a string like:
    3 = number of test cases
    4
    7
    10
    
    Outputs:
    7
    44
    274
    """
    import sys
    import math
    # from io import StringIO
    
    # Process input
    input = sys.stdin.read().strip() # StringIO(test_input)
    lines = input.split("\n")
    N = list(map(int, lines[1:]))
    
    for n in N:
        if n==0:
            break
        
        # Initialize M, a list of counts of all combinations of 1,2,3 that add up to each index i
        M = [0]*(n+3)
        M[1] = 1
        M[2] = 2
        M[3] = 4
        for i in range(4, n+1):
            M[i] = M[math.floor(i-1)] + M[math.floor(i-2)] + M[math.floor(i-3)]
        print(M[n])
        
sum_count_solver()