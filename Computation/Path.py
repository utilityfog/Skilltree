import networkx as nx
import matplotlib.pyplot as plt

from collections import deque

class GraphVisualization:
    def __init__(self, vertices):
        self.graph = nx.DiGraph()  # Directed graph
        self.graph.add_nodes_from(range(vertices))

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def bfs(self, start, end):
        # Use networkx BFS to find path from start to end
        try:
            predecessors = dict(nx.bfs_predecessors(self.graph, source=start))
            # Reconstruct the path from end to start using predecessors
            path = [end]
            while path[-1] != start:
                path.append(predecessors[path[-1]])
            path.reverse()
            print(f"Path found from {start} to {end}: {path}")
            return path
        except KeyError:
            print(f"No path found from {start} to {end}")
            return None

    def show_graph(self, path=None):
        pos = nx.spring_layout(self.graph)  # Positions for all nodes
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=15)

        if path:
            edge_labels = dict([((u, v,), f'{u}->{v}')
                                for u, v, d in self.graph.edges(data=True)])
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='lightgreen')
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='r', width=2)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()

if __name__ == "__main__":
    gv = GraphVisualization(4)
    gv.add_edge(0, 1)
    gv.add_edge(0, 2)
    gv.add_edge(1, 2)
    gv.add_edge(2, 0)
    gv.add_edge(2, 3)
    gv.add_edge(3, 3)

    path = gv.bfs(0, 3)
    gv.show_graph(path=path)  # Visualize the graph and highlight the path