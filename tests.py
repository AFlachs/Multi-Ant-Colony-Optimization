import AntColonyOptimizer
import GraphUtils
import numpy as np
import time

LIST_NODES = [10,50,100,500]


def test_barbell_modified():

    pass

def test_small_world():
    """
    Uses watts-strogarz model to generate new graphs.
    """
    pass

def test_scale_free():
    """
    Uses barbasi-albert model to generate new graphs.
    """    
    pass

def test_freq_opt(num_graph,gamma,q0,iters=100):
    sum = 0
    for _ in range(iters):
        optimizer = AntColonyOptimizer(ants=5, types=2, init_pheromones=0.05, alpha=1, beta=2,
                                    beta_evaporation_rate=0, choose_best=q0, gamma=gamma, rho=0.1)
        best = optimizer.fit(num_graph, 20, verbose=False)
        if num_graph==1 and best == [[0, 2, 3], [0, 1, 3]] or best == [[0, 1, 3], [0, 2, 3]]:
            sum += 1
        elif num_graph==2 and True:
            sum += 1
            raise NotImplementedError
        elif num_graph==3 and True:
            sum += 1
            raise NotImplementedError
    return sum
        

def test_freq_bridge_passages():
    """
    For graph n°2, test 
    """
    pass


def test_paper(num_graph):
    Qs=[i*0.1 for i in range(0,6)]
    gammas=[i*0.1 for i in range(0,10)]
    res = np.ndarray(shape=(len(gammas),len(Qs)))

    for ig,g in enumerate(gammas):
        for iq,q in enumerate(Qs):
            res[ig,iq]=test_freq_opt(num_graph,g,q)
    #header are not writed in the .csv (g's\q's)->(vertical\horizontal) 
    np.savetxt("test_paper_graph%d.csv",res,fmt="%d",delimiter=",")

if __name__ == '__main__':
    pass