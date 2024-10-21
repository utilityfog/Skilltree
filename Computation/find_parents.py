class DAG:
    index = 0
    parent = None
    children = []
    visited = False

def find_parents():
    import queue
    
    dag_raw = [[1,6],[6,3],[3,5],[4,1],[2,4],[4,7]]
    node_list = []
    for node_raw in dag_raw:
        dag = DAG()
        # print(node_raw)
        dag.index = node_raw[0]
        dag.children.append(node_raw[1])
        print(dag.children)
        node_list.append(dag)
    
    for node in node_list:
        print(node.children)
    
    Q = queue.Queue()
    root = node_list[0]
    visited_num = 0
    root.parent = -1
    root.visited = True
    Q.put(root)
    
    while(visited_num >= len(node_list) or Q.empty()):
        current_node = Q.get()
        if current_node.visited == False:
            for child in current_node.children:
                if child.parent is None:
                    child.parent = current_node
                    Q.put(child)
            current_node.visited = True
            visited_num += 1
    
    for node in node_list:
        print(node.parent)
        
find_parents()