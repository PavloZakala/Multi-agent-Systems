import copy
import numpy as np

from abc import abstractmethod
from matplotlib import pyplot as plt


def generate_city_list(city_count=30):
    city_location = np.random.rand(city_count, 2)
    return city_location


def get_dist_matrix(city_locations):
    x, y = city_locations.T
    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)

    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def get_distance_by_path(dist_matrix, path):
    total_dist = 0.0
    for i in range(len(path) - 1):
        total_dist += dist_matrix[path[i], path[i + 1]]

    return total_dist


def draw_map_with_path(city_locations, best_path):
    city_locations_path = city_locations[best_path]
    for (start_x, start_y), (end_x, end_y) in zip(city_locations_path[:-1], city_locations_path[1:]):
        plt.plot([start_x, end_x], [start_y, end_y], color="r")
    plt.scatter(city_locations[:, 0], city_locations[:, 1], marker="o", color="b")

    for i, (x, y) in enumerate(city_locations):
        plt.annotate(f"{i}", (x, y))

    plt.show()


def random_search(city_locations, dist_matrix, iteration_count=1000):
    location_count = len(dist_matrix)

    best_dist = float("inf")
    best_path = []

    for _ in range(iteration_count):
        indexes = np.arange(location_count)
        np.random.shuffle(indexes)
        current_path = indexes.tolist()
        current_path.append(current_path[0])

        current_dist = get_distance_by_path(dist_matrix, current_path)
        if current_dist < best_dist:
            best_dist = current_dist
            best_path = current_path

    return best_dist, best_path


def greedy_search(city_locations, dist_matrix):
    start_idx = np.argmin(city_locations[:, 0])
    indexes = np.arange(len(city_locations)).tolist()

    # Element x_min_idx on x_min_idx position
    indexes.pop(start_idx)

    dist = 0.0
    path = [start_idx]
    current_idx = start_idx
    while len(indexes) != 0:
        next_idx_pos = np.argmin(dist_matrix[current_idx][indexes])
        next_idx = indexes[next_idx_pos]
        indexes.pop(next_idx_pos)

        path.append(next_idx)
        dist += dist_matrix[current_idx][next_idx]

        current_idx = next_idx

    path.append(start_idx)
    dist += dist_matrix[current_idx][start_idx]

    return dist, path


def ant_colony_optimization(city_locations, dist_matrix, iteration_count=1000, alpha=3.0, beta=1.0, ro=0.6,
                            Q=7):
    POPULATION_SIZE = 16

    pheromone = np.ones_like(dist_matrix)

    def run_ant():

        available_cities = np.arange(len(city_locations))
        np.random.shuffle(available_cities)

        available_cities = available_cities.tolist()
        start_index = available_cities.index(0)
        start_city = available_cities.pop(start_index)

        ant_path = [start_city]
        current_city = start_city
        dist = 0.0

        while len(available_cities) != 0:
            weights = (pheromone[current_city][available_cities] ** alpha) * \
                      ((1 / dist_matrix[current_city][available_cities]) ** beta)

            weights = weights / np.sum(weights)

            next_index = np.random.choice(np.arange(len(available_cities)), p=weights)
            next_city = available_cities[next_index]
            dist += dist_matrix[current_city][next_city]

            ant_path.append(next_city)
            available_cities.pop(next_index)
            current_city = next_city

        dist += dist_matrix[current_city][next_city]
        ant_path.append(start_city)

        return ant_path, dist

    def dPheromone(population_paths, population_distances):
        d_pheromone = np.zeros_like(pheromone)

        for path_k, L_k in zip(population_paths, population_distances):
            dPh_k = Q / L_k
            d_pheromone[path_k[:-1], path_k[1:]] += dPh_k

        return d_pheromone

    for _ in range(iteration_count):

        population_paths = []
        population_distances = []
        for _ in range(POPULATION_SIZE):
            ant_path, dist = run_ant()
            population_paths.append(ant_path)
            population_distances.append(dist)

        # Update pheromone
        pheromone = pheromone * (1 - ro) + dPheromone(population_paths, population_distances)

    _, best_path = greedy_search(city_locations, pheromone)
    best_dist = np.sum(dist_matrix[best_path[:-1], best_path[1:]])

    return best_dist, best_path


class BasePath:

    @staticmethod
    def mutation(vec):
        assert False

    @staticmethod
    def crossover(vec1, vec2):
        assert False

    def __init__(self, vector_size):
        self.vector_size = vector_size

    def copy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def get_value(self):
        assert False


class AdjacencyMatrixPath(BasePath):

    @staticmethod
    def mutation(vec):
        def swap(matrix, idx1, idx2):
            matrix[[idx1, idx2]] = matrix[[idx2, idx1]]

        idx1, idx2 = np.random.randint(vec.vector_size, size=(2))
        idx3 = np.argmax(vec._matrix[idx1])
        idx4 = np.argmax(vec._matrix[idx2])

        swap(vec._matrix, idx1, idx2)
        swap(vec._matrix, idx3, idx4)

    @staticmethod
    def crossover(vec1, vec2):
        idx1, idx2 = np.random.randint(len(vec1._vector), size=(2))
        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
        # vec._matrix[idx1:idx2]

    def __init__(self, vector_size):
        super(AdjacencyMatrixPath, self).__init__(vector_size)

        indexes = np.arange(vector_size)
        np.random.shuffle(indexes)
        indexes = indexes.tolist()

        indexes += indexes[:1]

        self._matrix = np.zeros((vector_size, vector_size), dtype=np.int32)
        self._matrix[indexes[:-1], indexes[1:]] = 1

    def get_value(self):
        start_index = 0
        path = [start_index]

        next_index = start_index
        while True:
            next_index = np.argmax(self._matrix[next_index])
            path.append(next_index)
            if next_index == start_index:
                break

        return path


class OrderlyVectorPath(BasePath):

    @staticmethod
    def mutation(vec):
        idx1, = np.random.randint(len(vec._vector), size=(1))
        shift = np.random.choice([-3, -2, -1, 1, 2, 3])

        vec._vector[idx1] = (vec._vector[idx1] + shift) % (vec.vector_size - idx1)

    @staticmethod
    def crossover(vec1, vec2):
        idx1, idx2 = np.random.randint(len(vec1._vector), size=(2))
        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)

        vec1._vector[idx1:idx2] = vec2._vector[idx1:idx2]
        vec2._vector[idx1:idx2] = vec1._vector[idx1:idx2]

    def __init__(self, vector_size):
        super(OrderlyVectorPath, self).__init__(vector_size)
        self._vector = np.array([np.random.randint(vector_size - i) for i in range(vector_size)])

    def get_value(self):
        L = list(range(self.vector_size))

        res_vector = []
        for v in self._vector:
            res_vector.append(L.pop(v))
        return res_vector + res_vector[:1]


def genetic_algorithm(city_locations, dist_matrix,
                                           iteration_count=100, population_size=64,
                                           PathClass=BasePath):
    def path_length(path):
        return np.sum(dist_matrix[path[:-1], path[1:]])

    SPLIT = 4

    assert population_size % SPLIT == 0

    sub_part_size = population_size // SPLIT
    cities_count = len(city_locations)

    population = [PathClass(cities_count) for _ in range(population_size)]

    for _ in range(iteration_count):
        values = [path_length(vec.get_value()) for vec in population]
        sorted_indexes = np.argsort(values)[:sub_part_size]

        next_population = [population[idx].copy() for _ in range(SPLIT) for idx in sorted_indexes]

        indexes = np.arange(len(next_population) // 2, len(next_population))
        np.random.shuffle(indexes)

        # # Crossover
        # for i in range(0, len(next_population) // 2 - 1, 2):
        #     vec1 = next_population[indexes[i]]
        #     vec2 = next_population[indexes[i + 1]]
        #
        #     PathClass.crossover(vec1, vec2)

        # Mutation
        for i, vec in enumerate(next_population):
            for _ in range(i // sub_part_size):
                PathClass.mutation(vec)

        population = next_population

    vec = population[sorted_indexes[0]]

    return values[sorted_indexes[0]], vec.get_value()


if __name__ == '__main__':
    from datetime import datetime

    np.random.seed(0)

    for city_count in [5, 10, 30, 50]:
        print("City Count {}".format(city_count))

        city_locations = generate_city_list(city_count)
        dist_matrix = get_dist_matrix(city_locations)

        time = datetime.now()
        best_dist, best_path = random_search(city_locations, dist_matrix, 500000)
        print("Random search", best_dist, datetime.now() - time, best_path)

        time = datetime.now()
        best_dist, best_path = greedy_search(city_locations, dist_matrix)
        print("Greedy search", best_dist, datetime.now() - time, best_path)

        time = datetime.now()
        best_dist, best_path = ant_colony_optimization(city_locations, dist_matrix, 100,
                                                       alpha=3.0, beta=1.0, ro=0.9, Q=45)
        print("Ant colony optimization", best_dist, datetime.now() - time, best_path)

        time = datetime.now()
        best_dist, best_path = genetic_algorithm(city_locations, dist_matrix,
                                                  population_size=256, iteration_count=100,
                                                  PathClass=OrderlyVectorPath)
        print("Genetic algorithm (OrderlyVectorPath)", best_dist, datetime.now() - time, best_path)


        time = datetime.now()
        best_dist, best_path = genetic_algorithm(city_locations, dist_matrix,
                                                  population_size=256, iteration_count=100,
                                                  PathClass=AdjacencyMatrixPath)
        print("Genetic algorithm (AdjacencyMatrixPath)", best_dist, datetime.now() - time, best_path)

        # draw_map_with_path(city_locations, best_path)
