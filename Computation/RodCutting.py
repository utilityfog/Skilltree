# Rod Cutting Problem Definition:
# Given a Rod of length n and a list of prices where each element represents the price paid for rod of length index,
# How can we cut the rod such that we maximize the revenue earned by selling its parts?

# prices = 첫 원소부터 시작해서 길이가 그 원소의 인덱스 만한 막대기의 가격이 각 원소 값인 배열
prices = [0,1,3,3,2,7,5,13,12,11,14] # e.g. max(n) = 10

def cut_rod(n: int, prices: [int]):
    """
    Method that solves the rod cutting problem.
    """
    n_minus_one = n-1 # Produces error otherwise
    revenues_without_one = [0]*n_minus_one # initialize with n-1 0's
    revenues_one = [prices[1]]
    revenues = [0] + revenues_one + revenues_without_one
    q = -1
    for i in range(1, n+1):
        for j in reversed(range(int(i/2), i+1)): # when i == 2, iterates 2->1
            q = max(q, prices[j] + revenues[i-j]) if i == j else max(q, revenues[j] + revenues[i-j]) # 설명: 중복 방지; e.g. revenues[i] >= prices[i] -> revenues[3]+prices[1]을 하면 revenues[1]+prices[3] 같은건 할 필요가 없음
        revenues[i] = q
    return revenues[n]

def rod_cutter():
    """
    Method that solves https://www.acmicpc.net/problem/11052
    """
    import sys
    
    # Process input
    readline = sys.stdin.readline
    N = int(readline().strip())
    prices = list(map(int, readline().split()))
    prices = [0] + prices
    
    # Initialize revenues list and max_value
    dp = [0]*(N+1)
    max_value = -1
    
    # DP problem solving
    for i in range(1, N + 1):
        max_value = prices[i] # Required in order to compare the price of each DP chunk separately
        for j in range(1, i // 2 + 1):
            max_value = max(max_value, dp[j] + dp[i - j])
        dp[i] = max_value

    return dp[N]

# Test
# print(cut_rod(10, prices))

# How to extract the individual pieces of the cut

# Problem: revenues[i] is a summarized form of the individual cuts we had to make to maximize revenue for a rod length of i
# There must be some way to store the 

def rod_cutting_with_chunk_extraction():
    """
    Method that solves https://www.acmicpc.net/problem/11052 augmented with the actual solution to how the rod needs to be cut
    """
    # Process input
    N = int(input())
    prices = [0]
    for str_in in input().split(" "):
        prices.append(int(str_in))
    
    # Initialize revenues list and max_value
    dp = [0] * (N + 1)
    max_value = -1
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            max_value = max(max_value, prices[j] + dp[i - j])
        dp[i] = max_value
    
    print(dp[N])

output = rod_cutter()
print(output)
