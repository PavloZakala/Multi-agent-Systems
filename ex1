import math
import copy
import time

import numpy as np
from matplotlib import pyplot as plt

# Завдання 1. Розробити  програму для знаходження екстремуму (мінімум) функції, реалізувати алгоритми:
# * Випадкового пошуку,
# * Алгоритм імітації відпалу,
# * Генетичний алгоритм.
# Порівняти результати (представити у вигляді таблиці)

f1_range = (-3, 3)


def f1(x):  # -3 <= x_i <= 3

    x1_cos = math.cos(2 * math.pi * x[0])
    x2_cos = math.cos(4 * math.pi * x[1])

    return 10 + (x[0] ** 2 - 10 * x1_cos) + (x[1] ** 2 - 7 * x2_cos)


def f2(x):
    return -x[0] * math.sin(math.sqrt(abs(x[0]))) + x[1] * math.cos(math.sqrt(abs(x[1])))


def f3(x):
    return 5 * x[0] ** 2 + 3 * x[1] ** 2 - math.cos(x[0] / 2.0) * math.cos(x[1] / math.sqrt(3)) + 1


def f4(x):
    return x[0] ** 2.0 + x[1] ** 2.0


def f5(x):
    return math.sin(x[0]) * math.sin(x[1] ** 2 / math.pi) + math.sin(x[1]) * math.sin(2 * x[1] ** 2 / math.pi)


def f6(x):
    return 5 * x[0] ** 2 + 3 * x[1] ** 2 - 7 * math.cos(x[0] / 4) + 3 * math.cos(x[1] / math.sqrt(5)) + 1


def f7(x):
    return x[0] ** 2 - 2 * x[1] * 3


def f8(x):
    return math.sin(x[0] / 3.0) * math.cos(x[0] ** 2.0 / np.pi) + \
           math.sin(x[1]) * math.sin(2 * x[1] ** 2 / math.pi)


def f9(x):
    return math.cos(x[0] / 3.0) * math.cos(x[1]) * math.exp(-(x[0] ** 2 + 2 * x[1] ** 2))


def f10(x):
    return 2 * math.sin(x[0] / (2 * math.pi)) * math.cos(x[1]) * math.exp(-(5 * x[0] ** 2 + x[1] ** 2))


def random_search(func, x1_range, x2_range, iteration_count=10000):
    np.random.seed(0)

    rand_points = np.random.rand(iteration_count, 2)
    rand_points = rand_points * [x1_range[1] - x1_range[0], x2_range[1] - x2_range[0]] + [x1_range[0],
                                                                                          x2_range[0]]

    values = list(map(func, rand_points.tolist()))
    idx_min_value = np.argmin(values)

    return values[idx_min_value], rand_points[idx_min_value]


class Chromosome:

    def __init__(self, vector_size, range=(0, 1)):
        self.vector_size = vector_size
        self._range = range
        numpy_vector = np.random.randint(2, size=(vector_size))
        self._chromosome = "".join(str(x) for x in numpy_vector)
        self._max_value = int("1" * vector_size, 2)

    def get_value(self):
        return int(self._chromosome, 2) / self._max_value * (self._range[1] - self._range[0]) + self._range[0]

    def get_max_value(self):
        return self._max_value

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def mutation(chromo):
        mutation_idx = np.random.randint(chromo.vector_size)
        new_char = '1' if chromo._chromosome[mutation_idx] == '0' else '0'

        chromo._chromosome = chromo._chromosome[:mutation_idx] + new_char + chromo._chromosome[
                                                                            mutation_idx + 1:]

    @staticmethod
    def crossover(chromo1, chromo2):
        vector_size = chromo1.vector_size
        left = np.random.randint(vector_size)
        right = left + np.random.randint(vector_size - left) + 1

        new_chromo1_value = chromo1._chromosome[:left] + \
                            chromo2._chromosome[left:right] + chromo1._chromosome[right:]

        new_chromo2_value = chromo2._chromosome[:left] + \
                            chromo1._chromosome[left:right] + chromo2._chromosome[right:]

        chromo1._chromosome = new_chromo1_value
        chromo2._chromosome = new_chromo2_value


def genetic_algorithm(func, x1_range, x2_range, population_size=64, iteration_count=1000):
    np.random.seed(0)

    CHROMOSOME_VECTOR_SIZE = 16
    SPLIT = 4

    assert population_size % SPLIT == 0

    sub_part_size = population_size // SPLIT

    population = [(Chromosome(CHROMOSOME_VECTOR_SIZE, range=x1_range),
                   Chromosome(CHROMOSOME_VECTOR_SIZE, range=x2_range)) for _ in range(population_size)]

    for i in range(iteration_count):
        values = [func([x1.get_value(), x2.get_value()]) for x1, x2 in population]
        sorted_indexes = np.argsort(values)[:sub_part_size]
        # print("Best value {} Mean value {}".format(values[sorted_indexes[0]], np.mean(values)))

        next_population = [(population[idx][0].copy(), population[idx][1].copy())
                           for _ in range(SPLIT) for idx in sorted_indexes]

        indexes = np.arange(len(next_population) // 2, len(next_population))
        np.random.shuffle(indexes)

        # Crossover
        for i in range(0, len(next_population) // 2 - 1, 2):
            chromo1_x1, chromo1_x2 = next_population[indexes[i]]
            chromo2_x1, chromo2_x2 = next_population[indexes[i + 1]]

            Chromosome.crossover(chromo1_x1, chromo2_x1)
            Chromosome.crossover(chromo1_x2, chromo2_x2)

        # Mutation
        for i, (chromo_x1, chromo_x2) in enumerate(next_population):
            for _ in range(i // sub_part_size):
                Chromosome.mutation(chromo_x1)
                Chromosome.mutation(chromo_x2)

        population = next_population

    x1, x2 = population[sorted_indexes[0]]

    return values[sorted_indexes[0]], np.array([x1.get_value(), x2.get_value()])


def draw_func(func, x1_range, x2_range, frequency=30, points={}):
    x1 = np.linspace(x1_range[0], x1_range[1], frequency)
    x2 = np.linspace(x2_range[0], x2_range[1], frequency)

    X, Y = np.meshgrid(x1, x2)
    grid_points = np.stack([X, Y]).reshape(2, -1).T
    Z = np.array(list(map(func, grid_points.tolist()))).reshape(X.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')

    for label, (p, color) in points.items():
        ax.scatter(p[0], p[1], p[2], c=color)

    plt.show()


def simulated_annealing(func, x1_range, x2_range, t_min=0.001, t_max=100.0):
    np.random.seed(0)

    # def gradient(func, state, eps=1e-5):
    #     x1, x2 = state
    #
    #     dx1 = (func([x1 - eps, x2]) - func([x1 + eps, x2])) / (2.0 * eps)
    #     dx2 = (func([x1, x2 - eps]) - func([x1, x2 + eps])) / (2.0 * eps)
    #
    #     return np.array([dx1, dx2])
    #
    # def next_state(state, alpha=0.001):
    #     dx = gradient(func, state)
    #     return state + alpha * dx

    def neighbour(state, alpha=5.0):
        dx = alpha * (np.random.rand(2) - 0.5)
        return state + dx

    def probability(dE, t_i):
        return np.e ** (-dE / t_i)

    def T(i, init_temperature):
        return 0.1 * init_temperature / i

    state = np.random.rand(2) * [x1_range[1] - x1_range[0], x2_range[1] - x2_range[0]] + [x1_range[0],
                                                                                          x2_range[0]]
    i = 1
    temp = T(i, t_max)
    while temp > t_min:
        # print(func(state))
        current_state = neighbour(state)
        current_state = np.clip(current_state, [x1_range[0], x2_range[0]], [x1_range[1], x2_range[1]])
        dE = func(current_state) - func(state)
        if dE <= 0.0:
            state = current_state
        else:
            value = np.random.rand()

            if value <= probability(dE, temp):
                state = current_state
        i += 1
        temp = T(i, t_max)

    return func(state), state


if __name__ == '__main__':

    functions = [
        (f1, ((-3, 3), (-3, 3))),
        (f2, ((-30, 30), (-10, 10))),
        (f3, ((-10, 10), (-10, 10))),
        (f4, ((-1, 1), (-1, 1))),
        (f5, ((0, math.pi), (0, math.pi))),
        (f6, ((-10, 30), (-10, 10))),
        (f7, ((-1, 1), (-1, 1))),
        (f8, ((0, math.pi), (0, math.pi))),
        (f9, ((-math.pi, 3 * math.pi), (-math.pi, 3 * math.pi))),
        (f10, ((-math.pi, math.pi), (-math.pi, math.pi)))
    ]

    for func, (x1_range, x2_range) in functions:
        print(func.__name__)
        rs_best_value, rs_best_point = random_search(func, x1_range, x2_range, iteration_count=10000)
        print("Random_search: {:.4f} ({:.4f}, {:.4f})".format(rs_best_value, rs_best_point[0], rs_best_point[1]))

        sa_best_value, sa_best_point = simulated_annealing(func, x1_range, x2_range, t_min=0.0001, t_max=50.0)
        print("Simulated annealing: {:.4f} ({:.4f}, {:.4f})".format(sa_best_value, sa_best_point[0], sa_best_point[1]))

        ga_best_value, ga_best_point = genetic_algorithm(func, x1_range, x2_range, iteration_count=700)
        print("Genetic algorithm: {:.4f} ({:.4f}, {:.4f})".format(ga_best_value, ga_best_point[0], ga_best_point[1]))

        points = {
            "rs": ((rs_best_point[0], rs_best_point[1], rs_best_value), "g"),
            "ga": ((ga_best_point[0], ga_best_point[1], ga_best_value), "r"),
            "sa": ((sa_best_point[0], sa_best_point[1], sa_best_value), "y")
        }

        draw_func(func, x1_range, x2_range, frequency=50, points=points)
