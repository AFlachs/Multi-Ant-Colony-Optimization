import numpy as np
import matplotlib.pyplot as plt
import time

import warnings

warnings.filterwarnings("ignore")

graph1 = np.array([[0, 1, 3, np.inf],
                   [1, 0, 1, 3],
                   [3, 1, 0, 1],
                   [np.inf, 3, 1, 0]])


class Graph:
    def __init__(self, nb_graph):
        if nb_graph == 1:
            self.distance_matrix = graph1
        else:
            raise TypeError("You didn't give a right number of graph.")

    def candidate_list(self, node):
        line_of_interest = self.distance_matrix[node]
        candidates = []
        nb_elems = 0
        for i, elem in enumerate(line_of_interest):
            if nb_elems >= 5:
                break
            if elem != 0 and elem != np.inf:
                candidates.append(i)
                nb_elems += 1
        return candidates

    def candidate_list_available(self, node, available_nodes):
        line_of_interest = self.distance_matrix[node]
        candidates = []
        nb_elems = 0
        for i, elem in enumerate(line_of_interest):
            if nb_elems >= 5:
                break
            if elem != 0 and elem != np.inf and (i in available_nodes):
                candidates.append(i)
                nb_elems += 1
        return candidates


class Type_of_ant:
    def __init__(self, nb_ants):
        self.nb_ants = nb_ants
        self.paths = []
        self.best_path_per_i = None
        self.sum_to_minimze = None
        self.best_path = None
        self.best_score = None
        self.best_series = []
        self.length = None
        self.best_length = None

    def initialize_pheromones(self, init_pheromones, nb_nodes, nb_types, heuristic_matrix, heuristic_beta, gamma):
        self.other_ants = np.zeros((nb_nodes, nb_nodes, nb_types-1))
        self.pheromone_table = np.full((nb_nodes, nb_nodes), init_pheromones)
        self.foreign_pheromone_table = np.full((nb_nodes, nb_nodes), init_pheromones * (nb_types - 1))
        self.probability_matrix = (self.pheromone_table) * (
                heuristic_matrix ** heuristic_beta) * ((
                                                               1 / self.foreign_pheromone_table) ** gamma)  # element by element multiplication


class AntColonyOptimizer:
    def __init__(self, ants, types, init_pheromones, alpha=1.0, beta=0.0, beta_evaporation_rate=0,
                 choose_best=.1, gamma=0.0, rho=0.0):
        """
        Ant colony optimizer.  Traverses a graph and finds either the max or min distance between nodes.
        :param ants: number of ants per type
        :param types: number of types of ants
        :param init_pheromones: initial value of the pheromones
        :param alpha: weighting of pheromone
        :param beta: weighting of heuristic (1/distance)
        :param beta_evaporation_rate: rate at which beta decays (optional)
        :param choose_best: probability to choose the best route
        :param gamma: weighting of foreign pheromones
        :param rho: pheromone decay
        """
        # Parameters
        self.nb_ants = ants
        self.ants = [Type_of_ant(ants) for i in range(types)]
        self.init_pheromones = init_pheromones
        self.heuristic_alpha = alpha  # On n'en aura plus besoin
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.exploration_rate = choose_best
        self.heuristic_matrix = None
        self.gamma = gamma
        self.rho = rho

        self.set_of_available_nodes = None

        # Internal stats
        self.best = []
        self.fitted = False
        self.best_path = None
        self.fit_time = None

        # Plotting values
        self.stopped_early = False

    def __str__(self):
        string = "Ant Colony Optimizer"
        string += "\n--------------------"
        string += "\nDesigned to optimize either the minimum or maximum distance between nodes in a square matrix that behaves like a distance matrix."
        string += "\n--------------------"
        string += "\nNumber of ants:\t\t\t\t{}".format(self.ants)
        string += "\nEvaporation rate:\t\t\t{}".format(self.rho)
        string += "\nAlpha Heuristic:\t\t\t{}".format(self.heuristic_alpha)
        string += "\nBeta Heuristic:\t\t\t\t{}".format(self.heuristic_beta)
        string += "\nBeta Evaporation Rate:\t\t{}".format(self.beta_evaporation_rate)
        string += "\nExploration rate:\t\t{}".format(self.exploration_rate)
        string += "\n--------------------"
        string += "\nUSAGE:"
        string += "\nNumber of ants influences how many paths are explored each iteration."
        string += "\nThe alpha and beta heuristics affect how much influence the pheromones or the distance heuristic weigh an ants' decisions."
        string += "\nBeta evaporation reduces the influence of the heuristic over time."
        string += "\nChoose best is a percentage of how often an ant will choose the best route over probabilistically choosing a route based on pheromones."
        string += "\n--------------------"
        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string

    def _initialize(self, nb_graph):
        """
        Initializes the model by creating the various matrices and generating the list of available nodes
        """
        self.graph = Graph(nb_graph)
        self.num_nodes = len(self.graph.distance_matrix)
        self.heuristic_matrix = 1 / self.graph.distance_matrix
        for ant_type in self.ants:
            ant_type.initialize_pheromones(self.init_pheromones, self.num_nodes, len(self.ants), self.heuristic_matrix,
                                           self.heuristic_beta, self.gamma)
        self.set_of_available_nodes = list(range(self.num_nodes))
        self.terminal_node = self.set_of_available_nodes[-1]

    def _reinstate_nodes(self):
        """
        Resets available nodes to all nodes for the next iteration
        """
        self.set_of_available_nodes = list(range(self.graph.distance_matrix.shape[0]))

    def _remove_node(self, node):
        self.set_of_available_nodes.remove(node)

    def _update_probabilities(self):
        """
        After evaporation and intensification, the probability matrix needs to be updated.  This function
        does that.
        """
        for ant_type in self.ants:
            ant_type.probability_matrix = ant_type.pheromone_table * (self.heuristic_matrix ** self.heuristic_beta) * (
                    (1 / ant_type.foreign_pheromone_table) ** self.gamma)

    def _update_pheromones_ant(self):
        """
        After each ant, the pheromone table needs to be updated (before intensification)
        """
        for ant_type in self.ants:
            ant_type.pheromone_table = (1 - self.rho) * ant_type.pheromone_table + self.rho * (
                    1 / self.graph.distance_matrix)

    def update_foreign_pheromones(self):
        for type1 in self.ants:
            type1.foreign_pheromone_table = np.zeros(self.graph.distance_matrix.shape)
            for type2 in self.ants:
                if type1 != type2:
                    type1.foreign_pheromone_table += type2.pheromone_table

    def _choose_next_node(self, ant_type, from_node):
        """
        Chooses the next node based on probabilities.  If p < p_choose_best, then the best path is chosen, otherwise
        it is selected from a probability distribution weighted by the pheromone.
        :param from_node: the node the ant is coming from
        :return: index of the node the ant is going to
        """
        neighbours = self.graph.candidate_list_available(from_node, self.set_of_available_nodes)
        numerator = ant_type.probability_matrix[from_node, neighbours]

        if np.random.random() <= self.exploration_rate:
            next_node_index = np.argmax(numerator)
            next_node = neighbours[next_node_index]
        else:
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            next_node_index = np.random.choice(range(len(probabilities)), p=probabilities)
            next_node = neighbours[next_node_index]
        return next_node

    def update_other_ants_visits(self, other_ants, from_node, next_node):
        for i, type in enumerate(other_ants):
            type.other_ants[from_node, next_node, i] = 1
        return next_node

    def _evaluate(self, mode):
        """
        Evaluates the solutions of the ants by adding up the distances between nodes.
        :param mode: max or min
        :return: x and y coordinates of the best path as a tuple, the best path, and the best score
        """
        for type in self.ants:
            scores = np.zeros(len(type.paths))
            disjoints = np.zeros(len(type.paths))
            for index, path in enumerate(type.paths):
                score = 0
                disjoint = 0
                for i in range(len(path) - 1):
                    score += self.graph.distance_matrix[path[i], path[i + 1]]
                    disjoint = disjoint + score * np.sum(type.other_ants[path[i], path[i + 1]])
                scores[index] = score
                disjoints[index] = disjoint
            if mode == 'min':
                min_indices = np.where(disjoints==disjoints.min())[0]
                #if len(min_indices) > 1:
                    #scores_min_sum = scores[min_indices]
                    #best = min_indices[np.argmin(scores_min_sum)]
                #else:
                best = np.argmin(disjoints)
            elif mode == 'max':
                best = np.argmax(scores)
            else:
                raise TypeError("You didn't give a correct mode")
            type.best_path_per_i = type.paths[best]
            type.sum_to_minimze = disjoints[best]
            type.length = scores[best]

    def _update_pheromones_iteration(self):
        """
        Evaporate some pheromone as the inverse of the evaporation rate.  Also evaporates beta if desired.
        """
        for ant_type in self.ants:
            ant_type.pheromone_table = (1 - self.rho) * ant_type.pheromone_table + self.rho * np.full(
                (self.num_nodes, self.num_nodes), self.init_pheromones)

    def _intensify(self, type, best_path, best_score):
        """
        Increases the pheromone by some scalar for the best route.
        """
        for k in range(len(best_path) - 1):
            i = best_path[k]
            j = best_path[k + 1]
            type.pheromone_table[i, j] = (1 - self.rho) * type.pheromone_table[i, j] + 1 / best_score

    def fit(self, nb_graph, iterations=100, mode='min', early_stopping_count=20, verbose=True):
        """
        Fits the ACO to a specific map.  This was designed with the Traveling Salesman problem in mind.
        :param iterations: number of iterations
        :param mode: whether to get the minimum path or maximum path
        :param early_stopping_count: how many iterations of the same score to make the algorithm stop early
        :return: the best score
        """
        if verbose: print("Beginning ACO Optimization with {} iterations...".format(iterations))
        start = time.time()
        self._initialize(nb_graph)
        num_equal = 0

        for i in range(iterations):
            start_iter = time.time()
            path = []
            for type in self.ants:
                type.other_ants = np.zeros(
                    (self.graph.distance_matrix.shape[0], self.graph.distance_matrix.shape[1], (len(self.ants) - 1)))
            for ant in range(self.nb_ants):
                for type in self.ants:
                    other_types = [t for t in self.ants if t != type]
                    current_node = 0
                    self._remove_node(current_node)
                    while True:
                        path.append(current_node)
                        if current_node == self.terminal_node:
                            break
                        elif len(self.graph.candidate_list_available(current_node, self.set_of_available_nodes)) != 0:
                            current_node_index = self._choose_next_node(type, current_node)
                            current_node = self.update_other_ants_visits(other_types, current_node, current_node_index)

                            self._remove_node(current_node)
                        else:
                            current_node_index = np.random.choice(self.graph.candidate_list(current_node))
                            current_node = self.update_other_ants_visits(other_types, current_node, current_node_index)

                    self._reinstate_nodes()
                    type.paths.append(path)
                    path = []
                    self._update_pheromones_iteration()

            self._evaluate(mode)

            for type in self.ants:
                if i == 0:
                    type.best_score = type.sum_to_minimze
                    type.best_path = type.best_path_per_i
                    type.best_length = type.length
                else:
                    if mode == 'min':
                        if type.sum_to_minimze < type.best_score:
                            type.best_score = type.sum_to_minimze
                            type.best_path = type.best_path_per_i
                            type.best_length = type.length
                    elif mode == 'max':
                        if type.sum_to_minimze > type.best_score:
                            type.best_score = type.sum_to_minimze
                            type.best_path = type.best_path_per_i

                if type.sum_to_minimze == type.best_score:
                    num_equal += 1
                else:
                    num_equal = 0

                type.best_series.append(type.sum_to_minimze)
                self._intensify(type, type.best_path, type.best_length)

                if verbose: print("Best score at iteration {}: {}; Best path {}; overall: {} ({}s)"
                                  "".format(i, round(type.sum_to_minimze, 2), type.best_path, round(type.best_score, 2),
                                            round(time.time() - start_iter)))

                if type.sum_to_minimze == type.best_score and num_equal == early_stopping_count:
                    self.stopped_early = True
                    print("Stopping early due to {} iterations of the same score.".format(early_stopping_count))
                    break

            self.update_foreign_pheromones()
            self._update_probabilities()

        self.fit_time = round(time.time() - start)
        self.fitted = True

        for type in self.ants:
            if mode == 'min':
                self.best.append(type.best_series[np.argmin(type.best_series)])
                if verbose: print(
                    "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
                # return self.best
            elif mode == 'max':
                self.best = type.best_series[np.argmax(type.best_series)]
                if verbose: print(
                    "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
                # return self.best
            else:
                raise ValueError("Invalid mode!  Choose 'min' or 'max'.")
        self._evaluate(mode)
        to_return = []
        for type in self.ants:
            to_return.append(type.best_path)
        return to_return

    def plot(self):
        """
        Plots the score over time after the model has been fitted.
        :return: None if the model isn't fitted yet
        """
        if not self.fitted:
            print("Ant Colony Optimizer not fitted!  There exists nothing to plot.")
            return None
        else:
            for i in range(len(self.ants)):
                fig, ax = plt.subplots(figsize=(20, 15))
                ax.plot(self.ants[i].best_series, label="Best Run")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Performance")
                ax.text(.8, .6,
                        'Ant type: {}\nEvap Rate: {}\nIntensify: {}\nAlpha: {}\nBeta: {}\nBeta Evap: {}\nChoose Best: {}\n\nFit Time: {}m{}'.format(
                            i, self.evaporation_rate, self.pheromone_intensification, self.heuristic_alpha,
                            self.heuristic_beta, self.beta_evaporation_rate, self.exploration_rate, self.fit_time // 60,
                            ["\nStopped Early!" if self.stopped_early else ""][0]),
                        bbox={'facecolor': 'gray', 'alpha': 0.8, 'pad': 10}, transform=ax.transAxes)
                ax.legend()
                plt.title("Ant Colony Optimization Results (best: {})".format(np.round(self.best, 2)))
                plt.show()

sum = 0
for i in range(100):
    optimizer = AntColonyOptimizer(ants=5, types=2, init_pheromones=0.05, alpha=1, beta=2,
                                   beta_evaporation_rate=0, choose_best=0, gamma=0, rho=0.1)
    best = optimizer.fit(1, 20, verbose=False)
    print(best)
    if best == [[0, 2, 3], [0, 1, 3]] or best == [[0, 1, 3], [0, 2, 3]]:
        sum+=1
print(sum)
