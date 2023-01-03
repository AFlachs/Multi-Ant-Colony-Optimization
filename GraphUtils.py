import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#from AntColonyOptimizer import inf

inf = 1000
graph1 = np.array([
        [0, 1, 3, inf],
        [1, 0, 1, 3],
        [3, 1, 0, 1],
        [inf, 3, 1, 0]
        ])
"""
    1
  / | \ 
0   |   3
  \ | / 
    2
"""


def mat_to_graph(matrix : np.ndarray) -> nx.Graph:
    """
    Convert a np.array of shape=(2,2) to a Graph object of the networkx library.

    :param matrix: adjacency matrix representing the graph.
    :return: Graph object.
    """
    matrix[(matrix==inf) | (matrix==np.Infinity)]=0
    return nx.from_numpy_array(matrix)

def get_min_cut_from_mat(matrix : np.ndarray) -> tuple:
    """
    Return the minimum cut of a graph given its adjacency matrix.

    :param matrix: adjacency matrix representing the graph.
    :return: tuple composed of (the cut value, pair of 2 sets representing the partition).
    """
    start, end = 0, len(matrix)-1
    G = mat_to_graph(matrix)
    return nx.minimum_cut(G,start,end,capacity='weight')

def display_from_mat(matrix : np.ndarray, cut=False) -> None:
    """
    Return the minimum cut of a graph given its adjacency matrix.

    :param matrix: adjacency matrix representing the graph.
    :return: 
    """    
    G = mat_to_graph(matrix)
    labels = {e: G.edges[e]['weight'] for e in G.edges}
    pos = nx.spring_layout(G)
    if cut:
        C = get_min_cut_from_mat(matrix)
        cols=[]
        for n in G.nodes:
            if n in C[1][0]:
                cols.append(0.9)
            else:
                cols.append(0.1)
        cols[0]=1.0 #start in ...
        cols[len(cols)-1]=0.0 #end in ... 
        nx.draw(G, pos, with_labels=True, node_color= cols, font_weight='bold')
    else:
        nx.draw(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def get_min_disjoint_path(matrix : np.ndarray,get_subsets=False) -> int:
    """
    Return the minimum number of disjoint path.

    :param matrix: adjacency matrix representing the graph.
    :param get_subsets: boolean specifying the type of the return.
    :return: if get_subsets, return a tuple composed of (the cut value, pair of 2 sets representing the partition),
            otherwise, return just the cut value representing the number of disjoint paths. 
    """
    matrix[matrix>0]=1
    C = get_min_cut_from_mat(matrix)
    if get_subsets:
        return C
    else:
        return C[0] #first element is the value of the cut (here, in a graph with all edge labelled with 0) 

def exist_shortest_path(matrix : np.ndarray) -> bool:
    """
    Return a boolean indicating if a shortest path exists, given an adjacency matrix,
    between the starting node (0 by default) and an ending node (last one by default) 

    :param matrix: adjacency matrix representing the graph.
    :return: True if a shortest path is found, False otherwise. 
    """
    G=mat_to_graph(matrix)
    return nx.has_path(G,0,len(matrix)-1)

def graph_to_mat(graph : nx.Graph,with_inf_val=False) -> np.ndarray:
    """
    
    """
    mat = nx.to_numpy_array(graph,dtype=int)
    if with_inf_val:
        for i in range(len(mat)):
            for j in range(i+1,len(mat)):
                if mat[i][j]==0:
                    mat[i][j],mat[j][i]=inf,inf #TODO: replace with np.Infinity
    return mat    

def display_from_graph(graph,cut=False) -> None:
    """
    
    """
    labels = {e: graph.edges[e]['weight'] for e in graph.edges}
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()
    if cut:
        raise(NotImplementedError)


mat1=np.array([
    [0,1,1,0,0],
    [1,0,1,0,0],
    [1,1,0,0,0],
    [0,0,0,0,2],
    [0,0,0,2,0]
])

#graph1=mat1

display_from_mat(graph1,cut=True)

G=mat_to_graph(graph1)
m=graph_to_mat(G)
print("coucou")
print(m)