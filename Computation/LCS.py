def lcs_solver():
    X = [1,3,4,5,2]
    Y = [2,6,3,9,2]
    Z = [1,6,3,6,1,2]
    m = len(X)
    n = len(Y)
    k = len(Z)
    C = [[[0 for _ in range(k+1)] for _ in range(n+1)] for _ in range(m+1)]
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for l, z in enumerate(Z):
                C[i+1][j+1][l+1] = C[i][j][l] + 1 if x==y==z else max(C[i][j+1][l+1], C[i+1][j][l+1], C[i+1][j+1][l])
            
    print(C[m][n][k]) # Final Output; this prints the length of the LCS between X[1:m], Y[1:n]
    
lcs_solver()