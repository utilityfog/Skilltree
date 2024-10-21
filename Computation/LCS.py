def lcs_solver():
    X = [1,2,2,1,2] #,2,2,1,2] # 이게 왜 되는지 궁금합니다! [1,3,4,5,2,3]
    Y = [2,2,1,1,1] #,1,2,1,2] # 이게 왜 되는지 궁금합니다! [2,6,4,3,9,3,2,3]
    Z = [1,2,2,1,2] #,2,1,1] # 이게 왜 되는지 궁금합니다! [1,6,4,3,6,1,3,2,3]
    m = len(X)
    n = len(Y)
    k = len(Z)
    C = [[[0 for _ in range(k+1)] for _ in range(n+1)] for _ in range(m+1)]
    L = [[[0 for _ in range(k+1)] for _ in range(n+1)] for _ in range(m+1)]
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for l, z in enumerate(Z):
                if x==y==z:
                    C[i+1][j+1][l+1] = C[i][j][l] + 1
                    L[i+1][j+1][l+1] = 3
                else:
                    C[i+1][j+1][l+1] = max(C[i][j+1][l+1], C[i+1][j][l+1], C[i+1][j+1][l])
                    if max(C[i][j+1][l+1], C[i+1][j][l+1], C[i+1][j+1][l]) == C[i][j+1][l+1]:
                        L[i+1][j+1][l+1] = 2
                    elif max(C[i][j+1][l+1], C[i+1][j][l+1], C[i+1][j+1][l])
            
    print(f"length of lcs: {C[m][n][k]}") # Final Output; this prints the length of the LCS between X[1:m], Y[1:n], Z[1:k]
    print(f"LCS: {L}")
    
lcs_solver()

# def lcs_solver_baekjoon():
#     import sys
    
#     X = sys.stdin.readline().strip()
#     Y = sys.stdin.readline().strip()
#     m = len(X)
#     n = len(Y)
#     C = [[[0 for _ in range(k+1)] for _ in range(n+1)] for _ in range(m+1)]
#     L=[]
    
#     for i, x in enumerate(X):
#         for j, y in enumerate(Y):
#             for l, z in enumerate(Z):
#                 if x==y==z:
#                     C[i+1][j+1][l+1] = C[i][j][l] + 1
#                     # try:
#                     #     assert L[C[i+1][j+1][l+1]] is not None
#                     # except IndexError:
#                     if len(L) < C[i+1][j+1][l+1]:
#                         L.append(x)
#                 else:
#                     C[i+1][j+1][l+1] = max(C[i][j+1][l+1], C[i+1][j][l+1], C[i+1][j+1][l])
            
#     print(f"length of lcs: {C[m][n][k]}") # Final Output; this prints the length of the LCS between X[1:m], Y[1:n], Z[1:k]
#     print(f"LCS: {L}")