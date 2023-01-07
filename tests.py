from csv import writer
from AntColonyOptimizer import AntColonyOptimizer
import numpy as np
import GraphUtils as gu
import time
import rust_macs

M = 0
init_pheromones = 0.05
beta = 2
rho = 0.1
source = 0
cl_len = 5

# Function to calculate the coefficient of variance for graph 3
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100


graph1 = np.array([[0, 1, 3, M],
                   [1, 0, 1, 3],
                   [3, 1, 0, 1],
                   [M, 3, 1, 0]])

graph2_no_weight = np.array([
    #0 1 2 3 4 5 6 7 8 9
    [0,1,1,M,M,M,M,M,M,M],#v0
    [1,0,1,1,1,M,M,M,M,M],#v1
    [1,1,0,1,1,M,M,M,M,M],#v2
    [M,1,1,0,1,1,M,M,M,M],#v3
    [M,1,1,1,0,M,1,M,M,M],#v4
    [M,M,M,1,M,0,1,1,1,M],#v5
    [M,M,M,M,1,1,0,1,1,M],#v6
    [M,M,M,M,M,1,1,0,1,1],#v7
    [M,M,M,M,M,1,1,1,0,1],#v8
    [M,M,M,M,M,M,M,1,1,0] #v9
])

graph2_weight = np.array([
    #0 1 2 3 4 5 6 7 8 9
    [0,1,1,M,M,M,M,M,M,M],#v0
    [1,0,1,1,1,M,M,M,M,M],#v1
    [1,1,0,1,1,M,M,M,M,M],#v2
    [M,1,1,0,1,3,M,M,M,M],#v3
    [M,1,1,1,0,M,1,M,M,M],#v4
    [M,M,M,3,M,0,1,1,1,M],#v5
    [M,M,M,M,1,1,0,1,1,M],#v6
    [M,M,M,M,M,1,1,0,1,1],#v7
    [M,M,M,M,M,1,1,1,0,1],#v8
    [M,M,M,M,M,M,M,1,1,0] #v9
])

graph3 = np.array([
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


def seq_in_list(list, seq):
    for i in range(len(list) - len(seq) + 1):
        if list[i:i + len(seq)] == seq:
            return True
    return False


def test_paper_graph1(writing=False):
    f = open("Files/graph1_paper.csv", "a")
    writer_csv = writer(f)
    gamma_list = np.arange(0, 6)
    q_0_list = np.arange(0, 1, 0.1)
    for gamma in gamma_list:
        for q_0 in q_0_list:
            sum = 0
            for i in range(100):
                if i % 10 == 0:
                    print(f"Iter {i} with gamma={gamma} and q0={q_0}")
                ants, types, init_pheromones, beta, rho = 5, 2, 0.05, 2, 0.1
                source, dest = 0, 3
                t_max, cl_len = 20, 5
                results = rust_macs.optimize_macs(graph1, ants, types, init_pheromones, beta, q_0, gamma, rho,
                                                source, dest, t_max, cl_len, 0)
                best = [i[0] for i in results]
                if best == [[0, 2, 3], [0, 1, 3]] or best == [[0, 1, 3], [0, 2, 3]]:
                    sum += 1
            if writing:
                writer_csv.writerow([gamma, q_0, sum])
            else:
                print([gamma, q_0, sum])
                input()


def test_paper_graph_2_no_weight_method0(writing=False):
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
            if i % 10 == 0:
                print(f"Iteration {i} with {nb_type} types")
            ants, types, init_pheromones, beta, rho = 12, nb_type, 0.05, 2, 0.1
            source, dest = 0, 9
            t_max, cl_len = 1000, 5
            q_0, gamma = 1, 2
            results = rust_macs.optimize_macs(graph2_no_weight, ants, types, init_pheromones, beta, q_0, gamma, rho,
                                          source, dest, t_max, cl_len, 0)
            best = [i[0] for i in results]
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        if writing:
            writer_csv.writerow([nb_type, percentage, 0])
        else:
            print([nb_type, percentage, 0])
            input()


def test_paper_graph_2_no_weight_method1(writing=False):
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
            if i % 10 == 0:
                print(f"Iteration {i} with {nb_type} types")
            ants, types, init_pheromones, beta, rho = 12, nb_type, 0.05, 2, 0.1
            source, dest = 0, 9
            t_max, cl_len = 1000, 5
            q_0, gamma = 1, 2
            results = rust_macs.optimize_macs(graph2_no_weight, ants, types, init_pheromones, beta, q_0, gamma, rho,
                                          source, dest, t_max, cl_len, 1)
            best = [i[0] for i in results]
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        if writing:
            writer_csv.writerow([nb_type, percentage, 1])
        else:
            print([nb_type, percentage, 1])
            input()


def test_paper_graph_2_weight_method0(writing=False):
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
            if i % 10 == 0:
                print(f"Method 0 : iteration {i} with {nb_type} types")
            ants, types, init_pheromones, beta, rho = 12, nb_type, 0.05, 2, 0.1
            source, dest = 0, 9
            t_max, cl_len = 1000, 5
            q_0, gamma = 1, 2
            results = rust_macs.optimize_macs(graph2_weight, ants, types, init_pheromones, beta, q_0, gamma, rho,
                                          source, dest, t_max, cl_len, 0)
            best = [i[0] for i in results]
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type

        if writing:
            writer_csv.writerow([nb_type, percentage, 0])
        else:
            print([nb_type, percentage, 0])
            input()


def test_paper_graph_2_weight_method1(writing=False):
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
            if i % 10 == 0:
                print(f"Method 1 : iteration {i} with {nb_type} types")
            ants, types, init_pheromones, beta, rho = 12, nb_type, 0.05, 2, 0.1
            source, dest = 0, 9
            t_max, cl_len = 1000, 5
            q_0, gamma = 1, 2
            results = rust_macs.optimize_macs(graph2_weight, ants, types, init_pheromones, beta, q_0, gamma, rho,
                                          source, dest, t_max, cl_len, 1)
            best = [i[0] for i in results]
            for path in best:
                if seq_in_list(path, bridge1):
                    nb_ants_bridge1 += 1
        percentage = nb_ants_bridge1 / nb_type
        if writing:
            writer_csv.writerow([nb_type, percentage, 1])
        else:
            print([nb_type, percentage, 1])
            input()


def test_paper_graph3(writing=False):
    n_ants = 12
    gamma = 2
    dest = 15
    t_max = 1000 # Tah Jul
    disjoint_val_method = 0
    f = open("Files/graph3_paper.csv", "a")
    writer_csv = writer(f)
    nb_types = [2, 3, 4]
    q_0_list = np.arange(0, 1, 0.1)
    for nb_type in nb_types:
        for q_0 in q_0_list:
            min_length_per_run = []
            for i in range(100):
                if i%10 == 0:
                    print(f"Iter {i} with {nb_type} types and q0={q_0}")
                result = rust_macs.optimize_macs(graph3, n_ants, nb_type, init_pheromones, beta, q_0, gamma, rho, source, dest, t_max, cl_len, disjoint_val_method)
                result_paths = [i[0] for i in result]
                result_lengths = [i[2] for i in result]
                if check_disjoints(result_paths):
                    min_length_per_run.append(min(result_lengths))
            optimal_sol = min(min_length_per_run)
            perc_opt = min_length_per_run.count(optimal_sol)/len(min_length_per_run) #On compte combien de runs disjoints contiennent la meilleure solution
            coef_var = cv(min_length_per_run)
            if writing:
                writer_csv.writerow([nb_type,q_0,optimal_sol,len(min_length_per_run),perc_opt,coef_var])
            else:
                print([nb_type,q_0,optimal_sol,len(min_length_per_run),perc_opt,coef_var])
                input()


def test_paper_graph3_q0_varying(writing=False):
    n_ants = 12
    gamma = 2
    dest = 15
    t_max = 1000  # Tah Jul
    disjoint_val_method = 0
    f = open("Files/graph3_paper_q0_varying.csv", "a")
    writer_csv = writer(f)
    nb_types = [2, 3, 4]
    for nb_type in nb_types:
        min_length_per_run = []
        for i in range(100):
            if i % 10 == 0:
                print(f"Iter {i} with {nb_type} types")
            result = rust_macs.optimize_macs(graph3, n_ants, nb_type, init_pheromones, beta, 0.0, gamma, rho, source,
                                             dest, t_max, cl_len, disjoint_val_method, 0.1, 0.9)
            result_paths = [i[0] for i in result]
            result_lengths = [i[2] for i in result]
            if check_disjoints(result_paths):
                min_length_per_run.append(min(result_lengths))
        optimal_sol = min(min_length_per_run)
        perc_opt = min_length_per_run.count(optimal_sol) / len(
            min_length_per_run)  # On compte combien de runs disjoints contiennent la meilleure solution
        coef_var = cv(min_length_per_run)
        if writing:
            writer_csv.writerow([nb_type, "variation_of_q_0", optimal_sol, len(min_length_per_run), perc_opt, coef_var])
        else:
            print([nb_type, "variation_of_q_0", optimal_sol, len(min_length_per_run), perc_opt, coef_var])
            input()


def check_disjoints(best):
    for j in range(len(best) - 1):
        for k in range(len(best[j]) - 1):
            seq = [best[j][k], best[j][k + 1]]
            # Il faut chercher si la séquence se trouve dans un autre path de best
            for path in best[j + 1:]:
                if seq_in_list(path, seq):
                    return False
    return True




def test_barbell_modified(numtest,adv=True,verbose=True, use_rust_macs=True):
    #CONSTANTS (from optimizer)
    ANT_NUMBER = 10
    q_0, g = 0, 0#1
    t_max, cl_len = 100, 5
    #VARIABLES (for tests)
    filename="barbell_00{:d}".format(numtest)
    if filename=="barbell_000":
        LIST_NODES = [10,20]
        LIST_N_ANT_TYPES = [2,3,4] 
        N_LINKS = 3
        N_TESTS = 10
    if filename=="barbell_002":
        LIST_NODES = [10,15,20,25,30]#TODO: should be [10, 50, 100, 500]
        LIST_N_ANT_TYPES = [2,3,4] #TODO: AT LEAST 2 ANT TYPES (1 causes bug)
        N_LINKS = 2
        N_TESTS = 100
    if filename=="barbell_003":
        LIST_NODES = [10,50,100]#,500]
        LIST_N_ANT_TYPES = [2,4,6,8] 
        N_LINKS = 4
        N_TESTS = 50
    if filename=="barbell_004":
        LIST_NODES = [10,15,20,25,30,35,40,45,50]#TODO: should be [10, 50, 100, 500]
        LIST_N_ANT_TYPES = [2,3,4] #TODO: AT LEAST 2 ANT TYPES (1 causes bug)
        N_LINKS = 3
        N_TESTS = 100
        t_max=200
    
    N_PAIRS = 1 #should be 5, but would be identical in case of Barbell...
    N_GRAPHS = 1 #should be 3, but would be identical in case of Barbell...
    #N_PAIRS & N_GRAPHS are unuseful for Barbell

    f = open("Files/%s.csv"%(filename),'a',newline='')
    ww = writer(f)
    ww.writerow(["# nodes","# links","no graph","# ant types","% success/{:d}tests".format(N_TESTS),"% partial success","time(s)"])
    f.close()
    tic=time.perf_counter()
    itic=tic
    for nnodes in LIST_NODES:
        for ing in range(N_GRAPHS):
            if adv or verbose:print("Initialization of Barbell graph n° ",ing," with ",nnodes," nodes and " ,N_LINKS," bridges...")
            graph = gu.GraphFactory_barbell(nnodes,N_LINKS,ret_type='mat')
            disj_best_paths = gu.get_all_disjoint_paths(graph)
            print(graph)
            for ntypes in LIST_N_ANT_TYPES:
                if adv:print("  Tests with ",ntypes," ant types (",nnodes," nodes)")
                #nb_type=nlinks #the number of types of ants is the number of link for barbel
                if N_LINKS==len(disj_best_paths):
                    mincut=N_LINKS#same for the value of the minimum cut
                sum_successes = 0
                part_successes = 0
                for _ in range(N_TESTS):
                    if adv and _%10==0:print("iter",_)
                    if use_rust_macs:
                        results = optimize_macs(graph, ANT_NUMBER, ntypes, 0.05, 2, q_0, g, 0.1, 0, graph.shape[0]-1, t_max, cl_len, disjoint_val_method=0)
                        best = [i[0] for i in results]
                        
                    else:
                        raise TypeError("Use rust_macs instead...")
                        optimizer = AntColonyOptimizer(ants=ANT_NUMBER, types=ntypes, init_pheromones=0.05, beta=2, choose_best=q_0, gamma=g, rho=0.1)
                        best = optimizer.fit(0,graph_mat=graph,iterations=100, verbose=False,verbis=False)
                    n_good_path=0

                    for j in range(len(best)):
                        if best[j] in best_paths:
                            i+=1
                    if i==len(best):
                        if verbose:print("    :test",_,": succeeded !\n     sol=",best)
                        sum_successes+=1
                    else:
                        if n_good_path>0:part_successes+=1 #if at least one disjoint path is found
                        if verbose:print("    :test",_,": failed !\n     sol=",best)
                #CSV    #nodes,#links,n°graph,#types,%
                tac=time.perf_counter()
                f = open("Files/%s.csv"%(filename),'a',newline='')
                ww = writer(f)
                ww.writerow([nnodes,N_LINKS,ing,ntypes,sum_successes*(100/N_TESTS),part_successes*(100/N_TESTS),round(tac-itic,2)])
                f.close()
                itic=tac
                
    f.close()
    tac = time.perf_counter()
    print(f"Barbell took {tac - tic:0.1f} seconds to run.")


def test_small_world():
    """
    Uses watts-strogarz model to generate new graphs.
    """
    pass


def test_scale_free(numtest=1,adv=True,verbose=False,use_rust_macs=True):
    """
    Uses barbasi-albert model to generate new graphs.
    """
    #CONSTANTS (from optimizer)
    ANT_NUMBER = 10
    q_0, g = 0, 0#1
    t_max, cl_len = 100, 5
    #VARIABLES (for tests)
    filename="barabasi_albert_00{:d}".format(numtest)
    if filename=="barabasi_albert_000":
        LIST_NODES = [10,20]
        LIST_N_ANT_TYPES = [2,3] #AT LEAST 2 ANT TYPES (1 causes bug)
        N_TESTS = 20
        N_PAIRS = 2 
        N_GRAPHS = 2 
        t_max=20
    if filename=="barabasi_albert_001":
        LIST_NODES = [10,15,20,25,30,35,40,45,50] 
        LIST_N_ANT_TYPES = [2,3,4] #AT LEAST 2 ANT TYPES (1 causes bug)
        N_TESTS = 100
        N_PAIRS = 5 
        N_GRAPHS = 3 
        t_max=200
    

    f = open("Files/%s.csv"%(filename),'a',newline='')
    ww = writer(f)
    ww.writerow(["# nodes","no graph","no pair","# ant types","% success/{:d}tests".format(N_TESTS),"% partial success","time(s)","(mincut","src","dst",])
    f.close()
    tic=time.perf_counter()
    itic=tic

    for nnodes in LIST_NODES:
        for ing in range(N_GRAPHS):
            if adv or verbose:print("Initialization of Barabasi-Albert graph n° ",ing," with n=",nnodes," nodes (& m=2)")
            graph = gu.GraphFactory_barabasi_albert(nnodes,2,ret_type='mat')

            for ip in range(N_PAIRS):#to reduce variance
                src,dst=None,None
                while src==dst and gu.exist_shortest_path(graph,src,dst):
                    src=np.random.randint(0,len(graph)-1)
                    dst=np.random.randint(0,len(graph)-1)
                disj_best_paths = gu.get_all_disjoint_paths(graph,src,dst)
                mincut = gu.get_min_cut_from_mat(graph,src,dst)[0]
                if adv or verbose:print(" info : src=",src," and dst=",dst,", # disj paths=",len(disj_best_paths),", mincut=",mincut," \ndisj_path=",disj_best_paths)

                print(graph)
                for ntypes in LIST_N_ANT_TYPES:
                    if adv:print("  Tests with ",ntypes," ant types (",nnodes," nodes)")
                    #nb_type=nlinks #the number of types of ants is the number of link for barbel
#                    mincut=N_LINKS#same for the value of the minimum cut
                    sum_successes = 0
                    part_successes = 0
                    for _ in range(N_TESTS):
                        if adv and _%10==0:print("iter",_)
                        if use_rust_macs:
                            results = optimize_macs(graph, ANT_NUMBER, ntypes, 0.05, 2, q_0, g, 0.1, src, dst, t_max, cl_len, disjoint_val_method=0)
                            best = [i[0] for i in results]
                            
                        else:
                            raise TypeError("Use rust_macs instead...")
                            optimizer = AntColonyOptimizer(ants=ANT_NUMBER, types=ntypes, init_pheromones=0.05, beta=2, choose_best=q_0, gamma=g, rho=0.1)
                            best = optimizer.fit(0,graph_mat=graph,iterations=100, verbose=False,verbis=False)
                        i=0#number of good disjoint paths found that are part of the opt. solution
                        for j in range(len(best)):
                            if best[j] in disj_best_paths:
                                i+=1
                        if (ntypes>=mincut and i>=mincut)or(ntypes<mincut and i==ntypes):
                        #if i>=len(disj_best_paths):
                            if verbose:print("    :test",_,": succeeded !\n     sol=",best)
                            sum_successes+=1
                        else:
                            if i>0:part_successes+=1 #if at least one disjoint path is found
                            if verbose:print("    :test",_,": failed !\n     sol=",best)
                    tac=time.perf_counter()
                    f = open("Files/%s.csv"%(filename),'a',newline='')
                    ww = writer(f)
                    ww.writerow([nnodes,ing,ip,ntypes,sum_successes*(100/N_TESTS),part_successes*(100/N_TESTS),round(tac-itic,2),mincut,src,dst])
                    f.close()
                    itic=tac
                
    f.close()
    tac=time.perf_counter()
    print(f"Scale free took {tac - tic:0.1f} seconds to run.")    
    pass


def test_freq_opt(num_graph, gamma, q0, iters=100):
    sum = 0
    for _ in range(iters):
        optimizer = AntColonyOptimizer(ants=5, types=2, init_pheromones=0.05, alpha=1, beta=2,
                                       beta_evaporation_rate=0, choose_best=q0, gamma=gamma, rho=0.1)
        best = optimizer.fit(num_graph, 20, verbose=False)
        if num_graph == 1 and best == [[0, 2, 3], [0, 1, 3]] or best == [[0, 1, 3], [0, 2, 3]]:
            sum += 1
        elif num_graph == 2 and True:
            sum += 1
            raise NotImplementedError
        elif num_graph == 3 and True:
            sum += 1
            raise NotImplementedError
    return sum


def test_freq_bridge_passages():
    """
    For graph n°2, test 
    """
    pass


def test_paper(num_graph):
    Qs = [i * 0.1 for i in range(0, 6)]
    gammas = [i * 0.1 for i in range(0, 10)]
    res = np.ndarray(shape=(len(gammas), len(Qs)))

    for ig, g in enumerate(gammas):
        for iq, q in enumerate(Qs):
            res[ig, iq] = test_freq_opt(num_graph, g, q)
    # header are not writed in the .csv (g's\q's)->(vertical\horizontal)
    np.savetxt("test_paper_graph%d.csv", res, fmt="%d", delimiter=",")


if __name__ == '__main__':
    #test_paper_graph1()

    #test_barbell_modified(0,adv=True,verbose=False)#ok
    #test_barbell_modified(2,adv=True,verbose=False)#ok
    #test_barbell_modified(3,adv=True,verbose=False)#TODO
    test_barbell_modified(4,adv=True,verbose=False)#TODO

    #test_scale_free(0,adv=True,verbose=False)#ok
    #test_scale_free(1,adv=True,verbose=False)#TODO
    
    """
    print("Testing graph 1")
    test_paper_graph1(True)
    print("Testing graph 2 no weight 0")
    test_paper_graph_2_no_weight_method0(True)
    print("Testing graph 2 no weight 1")
    test_paper_graph_2_no_weight_method1(True)
    print("Testing graph 2 weight 0")
    test_paper_graph_2_weight_method0(True)
    print("Testing graph 2 weight 1")
    test_paper_graph_2_weight_method1(True)
    print("Testing graph 3 q0 varies")
    test_paper_graph3_q0_varying(True)
    print("Testing graph 3 q0 fixed")
    test_paper_graph3(True)
    # test_barbell_modified()
    """
