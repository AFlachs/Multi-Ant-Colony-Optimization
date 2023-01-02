import numpy as np
import matplotlib.pyplot as plt
import time

import warnings

warnings.filterwarnings("ignore")

inf = 1000

graph1 = np.array([[0, 1, 3, inf],
          [1, 0, 1, inf],
          [3, 1, 0, 1],
          [inf, 3, 1, 0]])

class Graph:
    def __init__(self, nb_graph):
        if nb_graph == 1:
            self.distance_matrix = graph1
        else:
            raise TypeError("You didn't give a right number of graph.")


    def candidate_list(self, node):
        line_of_interest = self.distance_matrix[node]
        candidates = []
        for i in range(len(line_of_interest)):
            if line_of_interest[i] != 0 and line_of_interest != inf:
                candidates.append(i)
        return candidates





class Type_of_ant:
    def __init__(self, nb_ants):
        self.nb_ants = nb_ants
        self.paths = []
        self.best_path_per_i = None
        self.length = None
        self.best_path = None
        self.best_score = None
        self.best_series = []
        self.best_path_coords = None

    def initialize_pheromones(self, nb_nodes, heuristic_matrix, heuristic_beta, gamma):
        self.pheromone_table = np.ones((nb_nodes, nb_nodes))
        # Remove the diagonal since there is no pheromone from node i to itself
        self.pheromone_table[np.eye(nb_nodes) == 1] = 0
        self.foreign_pheromone_table = np.ones((nb_nodes, nb_nodes))
        # Remove the diagonal since there is no pheromone from node i to itself
        self.foreign_pheromone_table[np.eye(nb_nodes) == 1] = 0
        self.probability_matrix = (self.pheromone_table) * (
                heuristic_matrix ** heuristic_beta) * ((1/self.foreign_pheromone_table) ** gamma)  # element by element multiplication


class AntColonyOptimizer:
    def __init__(self, ants, types, evaporation_rate, intensification, alpha=1.0, beta=0.0, beta_evaporation_rate=0,
                 choose_best=.1, gamma=0, rho=0):
        """
        Ant colony optimizer.  Traverses a graph and finds either the max or min distance between nodes.
        :param ants: number of ants per type
        :param types: number of types of ants
        :param evaporation_rate: rate at which pheromone evaporates
        :param intensification: constant added to the best path
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
        self.evaporation_rate = evaporation_rate
        self.pheromone_intensification = intensification
        self.heuristic_alpha = alpha  # On n'en aura plus besoin
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.exploration_rate = choose_best
        self.heuristic_matrix = None
        self.gamma = gamma
        self.rho = rho

        self.map = None
        self.set_of_available_nodes = None

        # Internal stats
        self.best = None
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
        string += "\nEvaporation rate:\t\t\t{}".format(self.evaporation_rate)
        string += "\nIntensification factor:\t\t{}".format(self.pheromone_intensification)
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
        num_nodes = len(self.graph.distance_matrix)
        self.heuristic_matrix = 1/self.graph.distance_matrix
        for ant_type in self.ants:
            ant_type.initialize_pheromones(num_nodes, self.heuristic_matrix, self.heuristic_beta, self.gamma)
        self.set_of_available_nodes = list(range(num_nodes))

    def _reinstate_nodes(self):
        """
        Resets available nodes to all nodes for the next iteration
        """
        self.set_of_available_nodes = list(range(self.map.shape[0]))

    def _update_probabilities(self):
        """
        After evaporation and intensification, the probability matrix needs to be updated.  This function
        does that.
        """
        for ant_type in self.ants:
            ant_type.probability_matrix = ant_type.pheromone_table * (self.heuristic_matrix ** self.heuristic_beta)

    def _update_pheromones(self):
        """
        After each ant, the pheromone table needs to be updated (before intensification)
        """
        for ant_type in self.ants:
            ant_type.pheromone_table = (1 - self.evaporation_rate)*ant_type.pheromone_table + self.rho*(1/self.graph.distance_matrix)


    def _choose_next_node(self, ant_type, from_node):
        """
        Chooses the next node based on probabilities.  If p < p_choose_best, then the best path is chosen, otherwise
        it is selected from a probability distribution weighted by the pheromone.
        :param from_node: the node the ant is coming from
        :return: index of the node the ant is going to
        """
        numerator = ant_type.probability_matrix[from_node, self.set_of_available_nodes]
        if np.random.random() <= self.exploration_rate:
            next_node = np.argmax(numerator)
        else:
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            next_node = np.random.choice(range(len(probabilities)), p=probabilities)
        return next_node

    def _remove_node(self, node):
        self.set_of_available_nodes.remove(node)

    def _evaluate(self, mode):
        """
        Evaluates the solutions of the ants by adding up the distances between nodes.
        :param paths: solutions from the ants
        :param mode: max or min
        :return: x and y coordinates of the best path as a tuple, the best path, and the best score
        """
        for type in self.ants:
            scores = np.zeros(len(type.paths))
            coordinates_i = []
            coordinates_j = []
            for index, path in enumerate(type.paths):
                score = 0
                coords_i = []
                coords_j = []
                for i in range(len(path) - 1):
                    coords_i.append(path[i])
                    coords_j.append(path[i + 1])
                    score += self.map[path[i], path[i + 1]]
                scores[index] = score
                coordinates_i.append(coords_i)
                coordinates_j.append(coords_j)
            if mode == 'min':
                best = np.argmin(scores)
            elif mode == 'max':
                best = np.argmax(scores)
            else:
                raise TypeError("You didn't give a correct mode")
            type.best_path_per_i = type.paths[best]
            type.length = scores[best]
            type.best_path_coords=(coordinates_i[best], coordinates_j[best])

    def _evaporation(self):
        """
        Evaporate some pheromone as the inverse of the evaporation rate.  Also evaporates beta if desired.
        """
        for ant_type in self.ants:
            ant_type.pheromone_table *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.beta_evaporation_rate)

    def _intensify(self, type, best_coords):
        """
        Increases the pheromone by some scalar for the best route.
        :param best_coords: x and y (i and j) coordinates of the best route
        """
        i = best_coords[0]
        j = best_coords[1]
        type.pheromone_table[i, j] += self.pheromone_intensification

    def fit(self, nb_graph, iterations=100, mode='min', early_stopping_count=20, verbose=True):
        """
        Fits the ACO to a specific map.  This was designed with the Traveling Salesman problem in mind.
        :param map_matrix: Distance matrix or some other matrix with similar properties
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

            for ant in range(self.nb_ants):
                for type in self.ants:
                    current_node = 0
                    start_node = current_node
                    while True:
                        path.append(current_node)
                        self._remove_node(current_node) # TODO: Revoir la condition de fin de la boucle (ici c'est jusqu'à ce qu'il n'y ait plus rien dans la candidate list, vérifier que c'est pareil qu'arriver à la fin
                        if len(self.set_of_available_nodes) != 0:
                            current_node_index = self._choose_next_node(type, current_node)
                            current_node = self.set_of_available_nodes[current_node_index]
                        else:
                            break

                    path.append(start_node)  # go back to start
                    self._reinstate_nodes()
                    type.paths.append(path)
                    path = []
                    self._update_pheromones()

            self._evaluate(mode)

            for type in self.ants:
                if i == 0:
                    type.best_score = type.length
                else:
                    if mode == 'min':
                        if type.length < type.best_score:
                            type.best_score = type.length
                            type.best_path = type.best_path_per_i
                    elif mode == 'max':
                        if type.length > type.best_score:
                            type.best_score = type.length
                            type.best_path = type.best_path_per_i

                if type.length == type.best_score:
                    num_equal += 1
                else:
                    num_equal = 0

                type.best_series.append(type.length)
                self._intensify(type, type.best_path_coords)

                if verbose: print("Best score at iteration {}: {}; overall: {} ({}s)"
                                  "".format(i, round(type.length, 2), round(type.best_score, 2),
                                            round(time.time() - start_iter)))

                if type.length == type.best_score and num_equal == early_stopping_count:
                    self.stopped_early = True
                    print("Stopping early due to {} iterations of the same score.".format(early_stopping_count))
                    break

            self._evaporation()
            self._update_probabilities()

        self.fit_time = round(time.time() - start)
        self.fitted = True

        for type in self.ants:
            if mode == 'min':
                self.best = type.best_series[np.argmin(type.best_series)]
                if verbose: print(
                    "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
                return self.best
            elif mode == 'max':
                self.best = type.best_series[np.argmax(type.best_series)]
                if verbose: print(
                    "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
                return self.best
            else:
                raise ValueError("Invalid mode!  Choose 'min' or 'max'.")

    def plot(self):
        """
        Plots the score over time after the model has been fitted.
        :return: None if the model isn't fitted yet
        """
        # TODO : A modifier encore
        if not self.fitted:
            print("Ant Colony Optimizer not fitted!  There exists nothing to plot.")
            return None
        else:
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.plot(self.best_series, label="Best Run")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Performance")
            ax.text(.8, .6,
                    'Ants: {}\nEvap Rate: {}\nIntensify: {}\nAlpha: {}\nBeta: {}\nBeta Evap: {}\nChoose Best: {}\n\nFit Time: {}m{}'.format(
                        self.ants, self.evaporation_rate, self.pheromone_intensification, self.heuristic_alpha,
                        self.heuristic_beta, self.beta_evaporation_rate, self.choose_best, self.fit_time // 60,
                        ["\nStopped Early!" if self.stopped_early else ""][0]),
                    bbox={'facecolor': 'gray', 'alpha': 0.8, 'pad': 10}, transform=ax.transAxes)
            ax.legend()
            plt.title("Ant Colony Optimization Results (best: {})".format(np.round(self.best, 2)))
            plt.show()
