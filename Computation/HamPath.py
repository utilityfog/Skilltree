import networkx as nx
import matplotlib.pyplot as plt
import time

def is_hamiltonian_path(G, path, visited, v, t, visualizer):
    visualizer(G, path)  # Visualize the current state
    if len(path) == len(G) and path[-1] == t:
        return True
    
    for w in G[v]:
        if not visited[w]:
            visited[w] = True
            path.append(w)
            visualizer(G, path)  # Visualize after adding to path
            
            if is_hamiltonian_path(G, path, visited, w, t, visualizer):
                return True
            
            visited[w] = False
            path.pop()
            visualizer(G, path, backtrack=True)  # Visualize backtrack
    
    return False

def hamiltonian_path(G, s, t, visualizer):
    visited = {node: False for node in G}
    path = [s]
    visited[s] = True
    
    if is_hamiltonian_path(G, path, visited, s, t, visualizer):
        return path
    return None

def visualize_graph_with_path(G, path, backtrack=False):
    graph = nx.DiGraph()
    for node in G:
        graph.add_node(node)
        for neighbor in G[node]:
            graph.add_edge(node, neighbor)
    
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=15)
    
    if path:
        path_edges = list(zip(path, path[1:]))
        edge_colors = 'red' if not backtrack else 'orange'  # Red for normal, orange for backtrack
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='lightgreen')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=edge_colors, width=2)
    
    plt.show()

# Graph configuration with a Hamiltonian Path
G = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D', 'E'],
    'D': ['B', 'C', 'E', 'F'],
    'E': ['C', 'D', 'F', 'G'],
    'F': ['D', 'E', 'G', 'H'],
    'G': ['E', 'F', 'H', 'I'],
    'H': ['F', 'G', 'I'],
    'I': ['G', 'H']
}

s, t = 'A', 'I'
path = hamiltonian_path(G, s, t, visualize_graph_with_path)
print("Hamiltonian Path:", path)