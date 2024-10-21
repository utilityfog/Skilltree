# 1. Squaring a number:
x = 10
n = 2
x ** n

# 2. Reading input from STDIN: 
import sys

for line in sys.stdin:
    line.strip()
    
    if 'q' == line.strip():
        break
    print(f'Input : {line}')

print("Exit")

# 3. Printing output to STDOUT: print by default

# 4. Coerce to int: int(x)

# 5. Coerce to string: str(x)

# 6. Remove duplicates in a string x: ''.join(dict.fromkeys(x))

# 7. Queue in python: 
from queue import Queue
q = Queue(maxsize = 3)
print(q.qsize())
q.put('a')
q.put('b')
q.put('c')
print("\nFull: ", q.full()) 
print("\nElements dequeued from the queue")
print(q.get())
print(q.get())
print(q.get())
print("\nEmpty: ", q.empty())
q.put(1)
print("\nEmpty: ", q.empty()) 
print("Full: ", q.full())

# 8. a
