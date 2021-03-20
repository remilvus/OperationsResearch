import random
from corpus_preparation import load_corpus
from genetic.fitness_function import calculate_invDB_fitness
from heapq import nsmallest, nlargest
import numpy as np

from genetic.timer import Timer


def random_chromosome(Kmin, Kmax, nr_docs):
    k = random.randint(Kmin, Kmax)
    return random.sample(range(nr_docs), k=k)

class GeneticAlgorithm:
    def __init__(self, doc_by_term, Kmin, Kmax):
        self.doc_by_term = doc_by_term
        self.N = doc_by_term.shape[0]
        self.Kmin=Kmin
        self.Kmax=Kmax
        self.timer = Timer()

    def run(self, population_size=100, iterations=1000):
        population = self.init_population(population_size)
        for i in range(iterations):
            # print(population[:2])
            parents = self.SUS(population, population_size//2)
            population = parents+parents

    def init_population(self, size):
        return np.array(np.array([np.array(a) for a in
                                  [random_chromosome(self.Kmin, self.Kmax, self.N) for _ in range(size)]], dtype=object),
                        dtype=object)

    def SUS(self, population, size):
        """ stochastic universal sampling """
        self.timer()
        fitness = calculate_invDB_fitness(self.doc_by_term, population)
        print(sum(fitness))
        self.timer.stop('fitness batch')
        sum_fitness = sum(fitness)
        pointers_dist = sum_fitness/size
        start_pointer = random.uniform(0, pointers_dist)
        pointers = [start_pointer + i*pointers_dist for i in range(size)]
        keep = []
        curr_sum = fitness[0]
        i = 0
        for point in pointers:
            while curr_sum < point:
                i += 1
                curr_sum += fitness[i]
            keep.append(population[i])
        return keep # todo scramble?

    def point_crossover(self, parent1, parent2):
        is_valid = False
        while not is_valid:
            crossover_point = random.randrange(1, self.N)
            child1 = np.concatenate((parent1[parent1<=crossover_point], parent2[parent2>crossover_point]))
            child2 = np.concatenate((parent2[parent2<=crossover_point], parent1[parent1>crossover_point]))
            is_valid = self.valid(child1) and self.valid(child2)
            print(is_valid)
        return child1, child2

    def valid(self, chromosome):
        return self.Kmin <= len(chromosome) <= self.Kmax

    def roulette_selection(self, fitness):
        sum_fitness = sum(fitness)
        selection_point = random.uniform(0, sum_fitness)
        curr_sum = 0
        selection_idx = 0
        while curr_sum < selection_point:
            curr_sum += fitness[selection_idx]
            selection_idx += 1
        if selection_idx == len(fitness):
            selection_idx -= 1
        return selection_idx

if __name__ == '__main__':
    corpus = load_corpus("..\\corpus4k60.csv")
    print(corpus.shape)
    ga = GeneticAlgorithm(corpus, 5, 30)
    ga.run(100)
