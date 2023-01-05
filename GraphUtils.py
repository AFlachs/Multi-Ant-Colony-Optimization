import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
#from AntColonyOptimizer import inf




inf = 1000

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
    
    try:
        labels = {e: graph.edges[e]['weight'] for e in graph.edges}
    except:
        labels = {e: 1 for e in graph.edges}
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()
    if cut:
        raise(NotImplementedError)

def GraphFactory_barabasi_albert(N : int, M : int,resample=True) -> nx.Graph:
    """
    This method generates randomly a barabasi-albert (scale free) graph.
    NOTE that for i in range(N) : if i<=M, then there is only one new connecting-edge built. 
    If i>M (until i<N), then there are progressively M new connecting-edges built with previous nodes for this method. 

    :param N: number of nodes in the graph that has to be generated.
    :param M: number of edges to attach from a new node to an existing one.
    :return: new randomly generated graph
    """
    G = nx.barabasi_albert_graph(N,M)
    if resample:
        # resample labels to allow 'random' definition of start & end nodes when taking 1st & last in array.
        map = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: np.random.random())))
        G = nx.relabel_nodes(G, map)
        # 2 previous  lines from https://stackoverflow.com/questions/59739750/how-can-i-randomly-permute-the-nodes-of-a-graph-with-python-in-networkx
    return G

def GraphFactory_erdos_renyi(N : int) -> nx.Graph:
    #TODO: docstring and auto N's.
    return nx.erdos_renyi_graph(N,1.001*np.log(N)/N)

def GraphFactory_watts_strogatz(N=12,K=4,P=0.0):
    #ex usage:
    return nx.watts_strogatz_graph(N,K,P)
    #TODO: implement with interesting values.
    raise(NotImplementedError)

def GraphFactory_barbell(N_nodes : int, N_links = 1, type='mat') -> nx.Graph | np.ndarray:
    """
    Generate a barbell graph with 'N_nodes' divided in 2 cliques (of 'N_nodes//2' nodes), and with 'N_links' between those 2 cliques. 

    :param N_nodes: Total number of vertices.
    :param N_links: Total number of edges between the 2 cliques.
    :param type: type of the return graph (either 'np.ndarray' or 'nx.Graph').
    :return: Barbell graph
    """
    G = nx.barbell_graph(N_nodes//2,N_nodes%2)
    if N_links > 1:
        n_links_to_implement=N_links-1
        while n_links_to_implement>0:
            start, dest = np.random.randint(0,(N_nodes//2)),np.random.randint((N_nodes+1)//2,N_nodes)
            if (start,dest) not in G.edges:
                G.add_edge(start,dest,weight=1)
                n_links_to_implement-=1
        
    if type=='mat':
        return graph_to_mat(G)
    elif type=='graph':
        return G
    else:
        exit("ERROR : 'type' parameter in 'GraphFactory_barbell' should be either equal to 'mat' or 'graph'.")

def GraphFactory_from_paper(num=1,bis=False) -> np.ndarray:
    """
    
    """
    if num==1:
        mat=np.array([
        [0, 1, 3, inf],
        [1, 0, 1, 3],
        [3, 1, 0, 1],
        [inf, 3, 1, 0]
        ])
    elif num==2:
        X=1 # value of the bridge (3,5)
        if bis:
            X=3
        mat=np.array([
            #0 1 2 3 4 5 6 7 8 9
            [0,1,1,0,0,0,0,0,0,0],#v0
            [1,0,1,1,1,0,0,0,0,0],#v1
            [1,1,0,1,1,0,0,0,0,0],#v2
            [0,1,1,0,1,X,0,0,0,0],#v3
            [0,1,1,1,0,0,1,0,0,0],#v4
            [0,0,0,X,0,0,1,1,1,0],#v5
            [0,0,0,0,1,1,0,1,1,0],#v6
            [0,0,0,0,0,1,1,0,1,1],#v7
            [0,0,0,0,0,1,1,1,0,1],#v8
            [0,0,0,0,0,0,0,1,1,0] #v9
        ])
    elif num==3:
        pass
    else:
        raise(NotImplementedError)
    return mat

#G = GraphFactory_barabasi_albert(N=10,M=1,resample=False)
#G = GraphFactory_barabasi_albert(N=10,M=2,resample=False)
#G = GraphFactory_barabasi_albert(N=10,M=3,resample=False)

#G = GraphFactory_erdos_renyi(40)

#G = GraphFactory_watts_strogatz()
#G = GraphFactory_watts_strogatz(K=2)

G = GraphFactory_barbell(12,N_links=3,type='graph')
#G = GraphFactory_barbell(10)

m=graph_to_mat(G)
print("# edges = ",len(G.edges))
print("# nodes = ",len(G.nodes))
display_from_graph(G)
