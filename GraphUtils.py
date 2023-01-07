import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from networkx.algorithms.connectivity import minimum_st_edge_cut

### CONVERSION functions (from/to networkx.Graph/numpy.array)
def mat_to_graph(matrix : np.ndarray) -> nx.Graph:
    """
    Convert a np.array of shape=(2,2) to a Graph object of the networkx library.

    :param matrix: adjacency matrix representing the graph.
    :return: Graph object.
    """
    tmp=np.copy(matrix)
    tmp[(tmp==np.inf)]=0
    return nx.from_numpy_array(tmp)

def graph_to_mat(graph : nx.Graph,with_inf_val=False) -> np.ndarray:
    """
    
    """
    mat = np.copy(nx.to_numpy_array(graph))
    if with_inf_val:
        for i in range(len(mat)):
            for j in range(i+1,len(mat)):
                if mat[i][j]==0:
                    mat[i][j],mat[j][i]=np.inf,np.inf
    return mat

### UTILS functions checking some properties of a graph
def get_min_cut_from_mat(matrix : np.ndarray,src=None,dst=None) -> tuple:
    """
    Return the minimum cut of a graph given its adjacency matrix.

    :param matrix: adjacency matrix representing the graph.
    :return: tuple composed of (the cut value, pair of 2 sets representing the partition).
    """
    if src==None : src=0
    if dst==None : dst=len(matrix)-1

    G = mat_to_graph(matrix)
    return nx.minimum_cut(G,src,dst,capacity='weight')

def get_edges_from_min_cut(matrix : np.ndarray) -> np.ndarray:
    G=mat_to_graph(matrix)
    cut_val=get_min_disjoint_path(matrix)
    res=np.zeros((cut_val,2))
    cut = minimum_st_edge_cut(G,0,len(matrix)-1)
    for i,e in enumerate(cut):
        res[i,0]=e[0]#edge, start side
        res[i,1]=e[1]#edge, arrival side
    return res

def get_all_disjoint_paths(matrix : np.ndarray,src=None,dst=None) -> list :
    if src==None : src=0
    if dst==None : dst=len(matrix)-1

    G=mat_to_graph(matrix)
    return list(nx.edge_disjoint_paths(G,src,dst))

def get_min_disjoint_path(matrix : np.ndarray,ret_subsets=False) -> int:
    """
    Return the minimum number of disjoint path.

    :param matrix: adjacency matrix representing the graph.
    :param get_subsets: boolean specifying the type of the return.
    :return: if get_subsets, return a tuple composed of (the cut value, pair of 2 sets representing the partition),
            otherwise, return just the cut value representing the number of disjoint paths. 
    """
    tmp=np.zeros(shape=matrix.shape)
    tmp[matrix==np.inf]=0
    tmp[matrix>0]=1
    tmp[matrix==np.inf]=0
    C = get_min_cut_from_mat(tmp)
    if ret_subsets:
        return C
    else:
        return int(C[0]) #first element is the value of the cut (here, in a graph with all edge labelled with 0) 

def exist_shortest_path(matrix : np.ndarray,src=None, dst=None) -> bool:
    """
    Return a boolean indicating if a shortest path exists, given an adjacency matrix,
    between the starting node (0 by default) and an ending node (last one by default) 

    :param matrix: adjacency matrix representing the graph.
    :return: True if a shortest path is found, False otherwise. 
    """
    if src==None : src=0
    if dst==None : dst=len(matrix)-1
    G=mat_to_graph(matrix)
    return nx.has_path(G,src,dst)

### DISPLAY functions (uses matplotlib.pyplot)
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



### FACTORIES (using some functions of networkx)
def GraphFactory_barabasi_albert(N : int, M : int,resample=True,ret_type='mat') -> nx.Graph | np.ndarray:
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
        Gnew = nx.relabel_nodes(G, map).copy()
        Gret=nx.Graph()
        Gret.add_nodes_from(sorted(Gnew.nodes(data=True)))
        Gret.add_edges_from(Gnew.edges(data=True))
        # 2 previous  lines from https://stackoverflow.com/questions/59739750/how-can-i-randomly-permute-the-nodes-of-a-graph-with-python-in-networkx
        if ret_type=='mat':
            return graph_to_mat(Gret)
        else:
            return Gret
    if ret_type=='mat':
        return graph_to_mat(G)
    else:
        return G

def GraphFactory_erdos_renyi(N : int) -> nx.Graph | np.ndarray:
    #TODO: docstring, multiple type return.
    return nx.erdos_renyi_graph(N,1.001*np.log(N)/N)

def GraphFactory_watts_strogatz(N=12,K=4,P=0.0, ret_type='mat') -> nx.Graph | np.ndarray:
    """
    
    """
    G = nx.watts_strogatz_graph(N,K,P)
    
    if ret_type=='mat':
        return graph_to_mat(G)
    elif ret_type=='graph':
        return G
    else:
        raise TypeError("ERROR : 'ret_type' parameter in 'GraphFactory_watts_strogatz' should be either equal to 'mat' or 'graph'.")

def GraphFactory_barbell(N_nodes : int, N_links = 1, ret_type='mat', with_inf_val=False) -> nx.Graph | np.ndarray:
    """
    Generate a barbell graph with 'N_nodes' divided in 2 cliques (of 'N_nodes//2' nodes), and with 'N_links' between those 2 cliques. 

    :param N_nodes: Total number of vertices.
    :param N_links: Total number of edges between the 2 cliques.
    :param type: type of the return graph (either 'np.ndarray' or 'nx.Graph').
    :return: Barbell graph
    """
    #if N_nodes%2==1: raise TypeError("Please specify an odd number of nodes for Barbell factory.")
    G = nx.barbell_graph(N_nodes//2,N_nodes%2)
    links=np.ones((N_links,2))*(-1) # allow to have bridge starting/arriving from/to different bridges. 
    links[0,:]=[N_nodes//2-1,N_nodes//2]
    if N_links > 1:
        n_links_to_implement=N_links-1
        while n_links_to_implement>0:
            #exclude source and sink nodes
            start, dest = np.random.randint(1,(N_nodes//2)),np.random.randint((N_nodes+1)//2,N_nodes-1)
            if (start,dest) not in G.edges and start not in links[:,0] and dest not in links[:,1]:
                G.add_edge(start,dest,weight=1)
                links[N_links-n_links_to_implement,:]=[start,dest]
                n_links_to_implement-=1
    
    if -1 in links:
        raise TypeError("Something happened in barbell factory...")
    
    if ret_type=='mat':
        return graph_to_mat(G,with_inf_val=with_inf_val)
    elif ret_type=='graph':
        return G
    else:
        raise TypeError("ERROR : 'ret_type' parameter in 'GraphFactory_barbell' should be either equal to 'mat' or 'graph'.")

def GraphFactory_from_paper(num=1,bis=False,ret_type='mat') -> np.ndarray:
    """
    Generates graphes from the scientific paper reproduced.

    :param num: number of the graph, between 1 and 3.
    :param bis: if True, the value of a bridge in graph 2 is increased.
    :param ret_type: return type of the function.
    :return:
    """
    M=np.inf
    if num==1:
        mat=np.array([
        [0, 1, 3, M],
        [1, 0, 1, 3],
        [3, 1, 0, 1],
        [M, 3, 1, 0]
        ])
    elif num==2:
        X=1 # value of the bridge (3,5)
        if bis:
            X=3
        mat=np.array([
            #0 1 2 3 4 5 6 7 8 9
            [0,1,1,M,M,M,M,M,M,M],#v0
            [1,0,1,1,1,M,M,M,M,M],#v1
            [1,1,0,1,1,M,M,M,M,M],#v2
            [M,1,1,0,1,X,M,M,M,M],#v3
            [M,1,1,1,0,M,1,M,M,M],#v4
            [M,M,M,X,M,0,1,1,1,M],#v5
            [M,M,M,M,1,1,0,1,1,M],#v6
            [M,M,M,M,M,1,1,0,1,1],#v7
            [M,M,M,M,M,1,1,1,0,1],#v8
            [M,M,M,M,M,M,M,1,1,0] #v9
        ])
    elif num==3:
        mat=np.array([
            #0 1 2 3 4 5 6 7 8 9 . 1 2 3 4 5
            [0,1,M,M,M,5,M,M,5,M,M,1,M,M,M,M],#V0
            [1,0,1,M,M,M,M,M,M,M,M,M,M,M,M,M],#V1
            [M,1,0,1,M,M,1,M,M,M,M,M,M,M,M,M],#V2
            [M,M,1,0,2,3,M,M,M,M,M,M,M,M,M,M],#V3
            [M,M,M,2,0,M,M,M,M,M,M,M,M,M,M,2],#V4
            [5,M,M,3,M,0,1,M,M,6,M,M,M,M,M,M],#V5
            [M,M,1,M,M,1,0,1,6,M,M,M,M,M,M,M],#V6
            [M,M,M,M,M,M,1,0,M,M,M,M,M,M,M,1],#V7
            [5,M,M,M,M,M,6,M,0,1,M,M,M,3,M,M],#V8
            [M,M,M,M,M,6,M,M,1,0,1,M,1,M,M,M],#V9
            [M,M,M,M,M,M,M,M,M,1,0,M,M,M,M,1],#V10
            [1,M,M,M,M,M,M,M,M,M,M,0,1,M,M,M],#V11
            [M,M,M,M,M,M,M,M,M,1,M,1,0,1,M,M],#V12
            [M,M,M,M,M,M,M,M,3,M,M,M,1,0,2,M],#V13
            [M,M,M,M,M,M,M,M,M,M,M,M,M,2,0,2],#V14
            [M,M,M,M,2,M,M,1,M,M,1,M,M,M,2,0]#V15
        ])
    else:
        raise TypeError("ERROR : 'num' parameter in 'GraphFactory_from_paper' should represent an existing graph.")
    
    if ret_type=='mat':
        return mat
    elif ret_type=='graph':
        return mat_to_graph(G)
    else:
        raise TypeError("ERROR : 'ret_type' parameter in 'GraphFactory_from_paper' should be either equal to 'mat' or 'graph'.")


if __name__ == '__main__':
    #G = GraphFactory_barabasi_albert(N=10,M=1,resample=False)
    #G = GraphFactory_barabasi_albert(N=10,M=2,resample=False)
    #G = GraphFactory_barabasi_albert(N=10,M=3,resample=False)

    #G = GraphFactory_erdos_renyi(40)

    #G = GraphFactory_watts_strogatz()
    #G = GraphFactory_watts_strogatz(K=2)

    #G = GraphFactory_barbell(N_nodes=12,N_links=3,type='graph')
    #G = GraphFactory_barbell(10)

    m = GraphFactory_from_paper(3,ret_type='mat')
    print(m)
    C=get_min_disjoint_path(m,True)
    print(m)
    print("C[0] = ",C[0])

    G=mat_to_graph(m)
    print("# edges = ",len(G.edges))
    print("# nodes = ",len(G.nodes))
    display_from_graph(G)
