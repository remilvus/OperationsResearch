import os
import random
from corpus_preparation import load_corpus, load_corpus_with_labels
from genetic.fitness_function import idb_score, idb_score_multi, score_by_labels
import numpy as np
import ray
from genetic.timer import Timer
from heapq import nlargest

class GeneticAlgorithm:
    def __init__(self, doc_by_term, labels, k_min, k_max, mutation_chance=0.01):
        self.doc_by_term = doc_by_term
        self.N = doc_by_term.shape[0]
        self.k_min = k_min
        self.k_max = k_max
        self.mutation_chance = mutation_chance
        self.labels = labels
        self.timer = Timer()
        if os.name == 'posix':  # use ray for multiprocessing
            ray.init()
            self.fitness_calculator = idb_score_multi
        else:
            self.fitness_calculator = idb_score

    def init_population(self, size):
        assert size % 4 == 0, "population size must by divisible by 4"
        population2 = self.random_population(size*2)
        population = self.choose_best(size, population2)
        return population

    def choose_best(self, size, population):
        fitness = self.fitness_calculator(self.doc_by_term, population)
        print("avg fitness of initial population: ", sum(fitness)/len(fitness))
        best_idx = nlargest(size, range(len(fitness)), key=lambda idx: fitness[idx])
        print(f"avg fitness of best {size}: ", sum([fitness[i] for i in best_idx])/len(best_idx))
        return [population[i] for i in best_idx]

    def random_population(self, size):
        return [self.random_chromosome() for _ in range(size)]

    def random_chromosome(self):
        k = random.randint(self.k_min, self.k_max)
        return np.random.choice(self.N, size=k, replace=False)

    def run(self, population_size=300, iterations=1000):
        population = self.init_population(population_size)
        fitness = self.fitness_calculator(self.doc_by_term, population)
        for i in range(iterations):
            if i % 10 == 0:
                self.stats(population, fitness)
            # parents selection
            parents, _fitness = self.SUS(population, population_size // 2, fitness)
            # children creation (with point crossover)
            children = self.get_children(parents)
            # mutations
            self.mutate_all(population)
            # survivor selection
            # population = parents + children
            population = population+children
            fitness = self.fitness_calculator(self.doc_by_term, population)
            population, fitness = self.SUS(population, population_size, fitness)

    def stats(self, population, fitness):
        self.timer.stop("iter")
        print("avg fitness:", sum(fitness) / len(fitness))
        max_fit_idx = fitness.index(max(fitness))
        print(f"max_fit: at {max_fit_idx}, fitness {fitness[max_fit_idx]}, genes {population[max_fit_idx]}")
        score_by_labels(population[max_fit_idx], self.doc_by_term, self.labels)

    def SUS(self, population, size, fitness=None):
        """ stochastic universal sampling """
        if fitness is None:
            fitness = self.fitness_calculator(self.doc_by_term, population)
        sum_fitness = sum(fitness)
        pointers_dist = sum_fitness / size
        start_pointer = random.uniform(0, pointers_dist)
        pointers = [start_pointer + i * pointers_dist for i in range(size)]
        chosen = []
        chosen_fitness = []
        curr_sum = fitness[0]
        i = 0
        for point in pointers:
            while curr_sum < point:
                i += 1
                curr_sum += fitness[i]
            chosen.append(population[i])
            chosen_fitness.append(fitness[i])
        return chosen, chosen_fitness

    def get_children(self, all_parents) -> list:
        children = []
        for i in range(0, len(all_parents), 2):
            children.extend(self.point_crossover(all_parents[i], all_parents[i + 1]))
        return children

    def point_crossover(self, parent1, parent2):
        is_valid = False
        lowest = min(min(parent1), min(parent2))
        highest = max(max(parent1), max(parent2))
        child1, child2 = None, None
        while not is_valid:
            crossover_point = random.randrange(lowest, highest + 1)
            child1 = np.concatenate((parent1[parent1 <= crossover_point], parent2[parent2 > crossover_point]))
            child2 = np.concatenate((parent2[parent2 <= crossover_point], parent1[parent1 > crossover_point]))
            is_valid = self.valid(child1) and self.valid(child2)
        return child1, child2

    def mutate_all(self, population):
        for i in range(len(population)):
            if random.random() <= self.mutation_chance:
                self.mutate(population[i])

    def mutate(self, chromosome):
        new_gene = random.randrange(0, self.doc_by_term.shape[0])
        forbidden = set(chromosome)
        while new_gene in forbidden:
            new_gene = random.randrange(0, self.doc_by_term.shape[0])
        new_idx = random.randrange(0, len(chromosome))
        chromosome[new_idx] = new_idx

    def valid(self, chromosome):
        return self.k_min <= len(chromosome) <= self.k_max

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
    corpus, labels = load_corpus_with_labels("../corpusTest.csv", "../labels.csv", n_topics=5)
    print(len(corpus))
    ga = GeneticAlgorithm(corpus, labels, 4, 8)

    ga.run()

