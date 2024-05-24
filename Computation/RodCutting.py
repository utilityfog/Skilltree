
# Rod Cutting Problem Definition:
# Given a Rod of length n and a list of prices where each element represents the price paid for rod of length index,
# How can we cut the rod such that we maximize the revenue earned by selling its parts?

# prices = 첫 원소부터 시작해서 길이가 그 원소의 인덱스 만한 막대기의 가격이 각 원소 값인 배열
prices = [0,1,3,3,6,7,8,11,12,12,14] # e.g. max(n) = 10

def cut_rod(n: int, prices: [int]):
    """
    Method that solves the rod cutting problem.
    Assumptions: Individual rod-length prices increase or remain the same with respect to increases in length.
    """
    n_minus_one = n-1 # Produces error otherwise
    revenues_without_one = [0]*n_minus_one # initialize with n-1 0's
    revenues_one = [prices[1]]
    revenues = [0] + revenues_one + revenues_without_one
    q = -1
    for i in range(1, n+1):
        for j in reversed(range(int(i/2), i+1)): # when i == 2, iterates 2->1
            q = max(q, prices[j] + revenues[i-j]) if i == j else max(q, revenues[j] + prices[i-j]) # 설명: 중복 방지; e.g. revenues[i] >= prices[i] -> revenues[3]+prices[1]을 하면 revenues[1]+prices[3] 같은건 할 필요가 없음
        revenues[i] = q
    return revenues[n]

# Test
print(cut_rod(10, prices))