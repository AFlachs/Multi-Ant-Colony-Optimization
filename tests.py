from csv import writer
from AntColonyOptimizer import AntColonyOptimizer
import numpy as np
import GraphUtils as gu
import time

def seq_in_list(list, seq):
    for i in range(len(list) - len(seq) + 1):
        if list[i:i + len(seq)] == seq:
            return True
    return False


def test_paper_graph1():
    f = open("Files/graph1_paper.csv", "a")
    writer_csv = writer(f)
    gamma_list = np.arange(0, 6)
    q_0_list = np.arange(0, 1, 0.1)
    for gamma in gamma_list:
        for q_0 in q_0_list:
            sum = 0
            for i in range(100):
                optimizer = AntColonyOptimizer(ants=5, types=2, init_pheromones=0.05, beta=2, choose_best=q_0,
                                               gamma=gamma, rho=0.1)
                best = optimizer.fit(1, 20, verbose=False)
                if best == [[0, 2, 3], [0, 1, 3]] or best == [[0, 1, 3], [0, 2, 3]]:
                    sum += 1
            print(sum)
            writer_csv.writerow([gamma, q_0, sum])


def test_paper_graph_2_no_weight_method0():
    """
    Fonction pour les tests à reproduire sur le graph2 avec les deux bridges = 1. On utilise la première
    méthode de calcul pour la fonction objective.
    """
    f = open("Files/graph2_paper_noweight.csv", "a")
    writer_csv = writer(f)
    nb_types = [*range(2, 7)]
    bridge1 = [3, 5]
    for nb_type in nb_types:
        nb_ants_bridge1 = 0
        for i in range(100):
            optimizer = AntColonyOptimizer(ants=12, types=nb_type, init_pheromones=0.05, beta=2, choose_best=1,
                                           gamma=2, rho=0.1)
            best = optimizer.fit(2, 1000, verbose=False)
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        writer_csv.writerow([nb_type, percentage, 0])


def test_paper_graph_2_no_weight_method1():
    """
    Fonction pour les tests à reproduire sur le graph2 avec les deux bridges = 1. On utilise la deuxième
    méthode de calcul pour la fonction objective.
    """
    f = open("Files/graph2_paper_noweight.csv", "a")
    writer_csv = writer(f)
    nb_types = [*range(2, 7)]
    bridge1 = [3, 5]
    for nb_type in nb_types:
        nb_ants_bridge1 = 0
        for i in range(100):
            optimizer = AntColonyOptimizer(ants=12, types=nb_type, init_pheromones=0.05, beta=2, choose_best=1,
                                           gamma=2, rho=0.1, method_score=1)
            best = optimizer.fit(2, 1000, verbose=False)
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        writer_csv.writerow([nb_type, percentage, 1])


def test_paper_graph_2_weight_method0():
    """
    Fonction pour les tests à reproduire sur le graph2 avec les deux bridges qui ont des weights diff. On utilise la première
    méthode de calcul pour la fonction objective.
    """
    f = open("Files/graph2_paper_weight.csv", "a")
    writer_csv = writer(f)
    nb_types = [*range(2, 7)]
    bridge1 = [3, 5]
    for nb_type in nb_types:
        nb_ants_bridge1 = 0
        for i in range(100):
            optimizer = AntColonyOptimizer(ants=12, types=nb_type, init_pheromones=0.05, beta=2, choose_best=1,
                                           gamma=2, rho=0.1)
            best = optimizer.fit(3, 1000, verbose=False)
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        writer_csv.writerow([nb_type, percentage, 0])


def test_paper_graph_2_weight_method1():
    """
    Fonction pour les tests à reproduire sur le graph2 avec les deux bridges qui ont des weights diff. On utilise la deuxième
    méthode de calcul pour la fonction objective.
    """
    f = open("Files/graph2_paper_weight.csv", "a")
    writer_csv = writer(f)
    nb_types = [*range(2, 7)]
    bridge1 = [3, 5]
    for nb_type in nb_types:
        nb_ants_bridge1 = 0
        for i in range(100):
            optimizer = AntColonyOptimizer(ants=12, types=nb_type, init_pheromones=0.05, beta=2, choose_best=1,
                                           gamma=2, rho=0.1, method_score=1)
            best = optimizer.fit(3, 1000, verbose=False)
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        writer_csv.writerow([nb_type, percentage, 1])


def test_paper_graph3():
    f = open("Files/graph3_paper.csv", "a")
    writer_csv = writer(f)
    nb_types = [2, 3, 4]
    q_0_list = np.arange(0, 1, 0.1)
    for nb_type in nb_types:
        for q_0 in q_0_list:
            nb_disjoint = 0
            for i in range(100):
                optimizer = AntColonyOptimizer(ants=12, types=nb_type, init_pheromones=0.05, beta=2, choose_best=q_0,
                                               gamma=2, rho=0.1)
                best = optimizer.fit(4, 1000, verbose=False)
                if check_disjoints(best):
                    nb_disjoint+=1


def check_disjoints(best):
    for j in range(len(best) - 1):
        for k in range(len(best[j]) - 1):
            seq = [best[j][k], best[j][k + 1]]
            # Il faut chercher si la séquence se trouve dans un autre path de best
            for path in best[j + 1:]:
                if seq_in_list(path, seq):
                    return False
    return True




def test_barbell_modified(verbose=True):
    #VARIABLES (for tests)
    LIST_NODES = [10,15,20,25,30]#TODO: should be [10, 50, 100, 500]
    N_LINKS = 3
    LIST_N_ANT_TYPES = [2,3,4] #TODO: AT LEAST 2 ANT TYPES (1 causes bug)
    N_PAIRS = 1 #TODO: should be 5, but would be identical in case of Barbell...
    N_GRAPHS = 1 #TODO: should be 3, but would be identical in case of Barbell...

    #CONSTANTS (from optimizer)
    N_TESTS = 50
    ANT_NUMBER=10
    q_0, g = 0, 1

    filename="barbell_001"
    f = open("Files/%s.csv"%(filename),'a',newline='')
    ww = writer(f)
    ww.writerow(["# nodes","# links","no graph","# ant types","% success/{:d}tests".format(N_TESTS),"time(s)"])
    tic=time.perf_counter()
    itic=tic
    for nnodes in LIST_NODES:
        for ing in range(N_GRAPHS):
            if verbose:print("Initialization of Barbell graph n° ",ing," with ",nnodes," nodes and " ,N_LINKS," bridges...")
            graph = gu.GraphFactory_barbell(nnodes,N_LINKS,ret_type='mat')
            best_paths = gu.get_all_disjoint_paths(graph)
            for ntypes in LIST_N_ANT_TYPES:
                if verbose:print("  Tests with ",ntypes," ant types")
                #nb_type=nlinks #the number of types of ants is the number of link for barbel
                mincut=N_LINKS#same for the value of the minimum cut
                sum_successes = 0
                for _ in range(N_TESTS):
                    if verbose:print("    :test",_,": began...")
                    optimizer = AntColonyOptimizer(ants=ANT_NUMBER, types=ntypes, init_pheromones=0.05, beta=2, choose_best=q_0,
                                               gamma=g, rho=0.1)
                    best = optimizer.fit(0,graph_mat=graph,iterations=100, verbose=False,verbis=False)
                    i=0
                    for j in range(len(best)):
                        if best[j] in best_paths:
                            i+=1
                    if i==len(best):
                        if verbose:print("    :test",_,": succeeded !\n     sol=",best)
                        sum_successes+=1
                    else:
                        if verbose:print("    :test",_,": failed !\n     sol=",best)
                #CSV    #nodes,#links,n°graph,#types,%
                tac=time.perf_counter()
                ww.writerow([nnodes,N_LINKS,ing,ntypes,sum_successes*(100/N_TESTS),round(tac-itic,2)])
                itic=tac
                
    f.close()
    tac=time.perf_counter()
    print(f"Barbell took {tac - tic:0.1f} seconds to run.")

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
    #test_paper_graph1()
    test_barbell_modified()