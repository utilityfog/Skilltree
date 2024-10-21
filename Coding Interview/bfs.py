# BFS for Adjacency Matrix using visited Queue (neighboring nodes are interpreted as neighbors in a mxn grid instead of only interpreting matrix[i][j] where i==j as nodes)
def bfs(matrix, row, col, visited):
    nodes = [(row, col)]  # BFS queue initialized with starting node
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Directions to move (down, up, right, left)
    
    while nodes:
        row, col = nodes.pop(0)
        
        # Ensure we are within matrix bounds
        if row < 0 or row >= len(matrix) or col < 0 or col >= len(matrix[0]):
            continue
        
        if (row, col) not in visited:
            visited.append((row, col))  # Mark the node as visited
            
            # For each direction, explore the neighbors
            for dx, dy in directions:
                new_row, new_col = row + dx, col + dy
                if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
                    if matrix[new_row][new_col] == 1:  # If it's a fresh orange or a valid node
                        nodes.append((new_row, new_col))

def bfs_wrapper(matrix):
    visited = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i, j) not in visited and matrix[i][j] == 1:
                # One bfs call completes exhaustive bfs for all connected regions
                bfs(matrix, i, j, visited)
    return visited

# Example matrix to run
matrix = [
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1]
]

print(bfs_wrapper(matrix))

# BFS for Adjacency List using visited Queue
def bfs_list(graph, start_node):
    """
    Performs a breadth-first search on a graph represented as an adjacency list.

    Args:
        graph: The adjacency list representation of the graph.
        start_node: The node to start the BFS from.

    Returns:
        A list of nodes visited in BFS order.
    """

    visited = set()
    queue = [start_node]
    result = []

    while queue:
        node = queue.pop(0)

        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return result

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(bfs_list(graph, 'A'))  # Output: ['A', 'B', 'C', 'D', 'E', 'F']

# Solution to: https://leetcode.com/problems/rotting-oranges/
class Solution(object):
    def bfs(self, matrix, nodes):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Directions for up, down, left, right
        minutes = 0  # Track the minutes

        while nodes:
            new_nodes = []  # To store the next set of nodes (like new_queue in previous solution)
            for row, col in nodes:
                for dx, dy in directions:
                    new_row, new_col = row + dx, col + dy
                    # Ensure we're within bounds of the matrix and check if it's a fresh orange
                    if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
                        if matrix[new_row][new_col] == 1:  # If it's a fresh orange
                            matrix[new_row][new_col] = 2  # Mark as rotten (visit)
                            new_nodes.append((new_row, new_col))  # Add to next round of processing

            if new_nodes:
                minutes += 1  # Increase minute count when we have more oranges to process
            nodes = new_nodes  # Process the next set of rotten oranges in the next round

        return minutes

    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        nodes = []  # Same as `queue` in the previous solution
        fresh_oranges = 0  # Count fresh oranges to track if we can complete the task

        # Step 1: Find all rotten oranges and count fresh oranges
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    nodes.append((i, j))  # Treat rotten oranges as starting nodes
                elif grid[i][j] == 1:
                    fresh_oranges += 1  # Count fresh oranges

        # If no fresh oranges are present, return 0 (nothing to rot)
        if fresh_oranges == 0:
            return 0

        # Step 2: Perform BFS to rot all reachable fresh oranges
        minutes_passed = self.bfs(grid, nodes)

        # Step 3: Check if there are any fresh oranges left after BFS
        for row in grid:
            if 1 in row:  # If any fresh oranges are left
                return -1

        return minutes_passed