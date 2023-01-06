from csv import writer
from AntColonyOptimizer import AntColonyOptimizer
import numpy as np

LIST_NODES = [10, 50, 100, 500]


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
        writer_csv.writerow([nb_type, percentage, 0])

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


if __name__ == '__main__':
    test_paper_graph1()
