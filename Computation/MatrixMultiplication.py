def matrix_chain_order_solver():
    import sys
    
    # Process input
    numlines = int(sys.stdin.readline().strip())
    p = list(map(int, sys.stdin.readline().strip().split(" ")))
    while numlines > 1:
        p.append(int(sys.stdin.readline().strip().split(" ")[1]))
        numlines-=1
    n = len(p) - 1
    
    # Initialize m and s with 0's
    m = [[0 for _ in range(n+1)] for _ in range(n+1)]
    s = [[0 for _ in range(n+1)] for _ in range(n+1)]
    
    # The code snippet `for i in range(1, n+1):
    #         m[i][i] = 0` is initializing the diagonal elements of the matrix `m` with 0.
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = 2**31
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    print(m[1][n])
    matrix_chain_order_tracer(s, 1, n, 1, n)
    
def matrix_chain_order_tracer(s, i: int, j: int, initial: int, final: int):
    if i == j:
        return f"A_{i}"
    else:
        group = f"({matrix_chain_order_tracer(s, i, s[i][j], i, j)}*{matrix_chain_order_tracer(s, s[i][j] + 1, j, i, j)})"
        if i == initial and j == final:
            print(group)
        return group
    
matrix_chain_order_solver()