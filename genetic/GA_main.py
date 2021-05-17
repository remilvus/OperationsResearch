import os
import random
from corpus_preparation import load_corpus, load_corpus_with_labels
from genetic.fitness_function import idb_score, idb_score_multi, score_by_labels
import numpy as np
import ray
from genetic.timer import Timer
from heapq import nlargest
import pandas as pd
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, doc_by_term, labels, k_min, k_max, mutation_chance=0.4):
        self.doc_by_term = doc_by_term
        self.N = doc_by_term.shape[0]
        self.k_min = k_min
        self.k_max = k_max
        self.mutation_chance = mutation_chance
        self.labels = labels
        self.timer = Timer()
        self.nr_iter = 0
        self.historic_fitness = []
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

    def choose_best(self, size, population, fitness=None, return_fitness=False):
        if fitness is None:
            fitness = self.fitness_calculator(self.doc_by_term, population)
        print("avg fitness of initial population: ", sum(fitness)/len(fitness))
        best_idx = nlargest(size, range(len(fitness)), key=lambda idx: fitness[idx])
        print(f"avg fitness of best {size}: ", sum([fitness[i] for i in best_idx])/len(best_idx))
        if return_fitness:
            return [population[i] for i in best_idx], [fitness[i] for i in best_idx]
        return [population[i] for i in best_idx]

    def random_population(self, size):
        return [self.random_chromosome() for _ in range(size)]

    def random_chromosome(self):
        k = random.randint(self.k_min, self.k_max)
        return np.random.choice(self.N, size=k, replace=False)

    def run(self, population_size=200, iterations=50, stats_step=10):
        """
        run genetic algorithm.
        :param population_size: size of the populations
        :param iterations: number of iterations to perform
        :param stats_step: every each step stats should be displayed
        :return: best chromosome
        """
        population = self.init_population(population_size)
        fitness = self.fitness_calculator(self.doc_by_term, population)
        for i in range(iterations):
            if i % stats_step == 0:
                self.stats(population, fitness)
            # parents selection
            parents, parents_fitness = self.SUS(population, population_size // 2, fitness)
            # children creation (with point crossover)
            children = self.get_children(parents)
            # mutations
            self.mutate_all(children)
            # survivor selection
            children_fitness = self.fitness_calculator(self.doc_by_term, children)
            # population = parents+children
            # fitness = parents_fitness + children_fitness
            population = population + children
            fitness = fitness + children_fitness
            population, fitness = self.SUS(population, population_size, fitness)
            # population, fitness = self.choose_best(population_size, population, fitness, True)
            self.nr_iter +=1
        return self.choose_best(1, population, fitness)

    def stats(self, population, fitness):
        print(f"iteration nr: {self.nr_iter}")
        self.timer.stop("iter")
        print("avg fitness:", sum(fitness) / len(fitness))
        max_fit_idx = fitness.index(max(fitness))
        self.historic_fitness.append([self.nr_iter, fitness[max_fit_idx], sum(fitness)/len(fitness)])
        print(f"max_fit: at {max_fit_idx}, fitness {fitness[max_fit_idx]}, genes {population[max_fit_idx]}")

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
        max_it = 0
        while not is_valid:
            crossover_point = random.randrange(lowest, highest)
            child1 = np.concatenate((parent1[parent1 <= crossover_point], parent2[parent2 > crossover_point]))
            child2 = np.concatenate((parent2[parent2 <= crossover_point], parent1[parent1 > crossover_point]))
            is_valid = self.valid(child1) and self.valid(child2)
            if max_it >10:
                return parent1, parent2
            max_it +=1
        return child1, child2

    def new_crossover(self, par1, par2):
        set1 = set(par1)
        set2 = set(par2)
        diff1 = list(set1-set2)
        leave1 = random.sample(diff1, len(diff1)//2)
        diff2 = list(set2-set1)
        leave2 = random.sample(diff2, len(diff2)//2)
        child1 = (set1.union(set(leave2)) - set(leave1))
        child2 = (set2.union(set(leave1)) - set(leave2))
        return list(child1), list(child2)

    def mutate_all(self, population):
        for i in range(len(population)):
            if random.random() <= self.mutation_chance:
                self.mutate(population[i])

    def mutate(self, chromosome):
        """ change random gene to new gene, not yet present in chormosome"""
        new_gene = random.randrange(0, self.doc_by_term.shape[0])
        forbidden = set(chromosome)
        while new_gene in forbidden:
            new_gene = random.randrange(0, self.doc_by_term.shape[0])
        new_idx = random.randrange(0, len(chromosome))
        chromosome[new_idx] = new_gene

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

    def plot_stats(self):
        df = pd.DataFrame(self.historic_fitness, columns=["iteration", "max_fitness", "avg_fitness"])
        df.plot(x='iteration', y=['max_fitness', 'avg_fitness'],
                title=f'mut={self.mutation_chance},next=SUS')
        plt.show()

if __name__ == '__main__':
    corpus, labels = load_corpus_with_labels("../example_corpus.csv", "../labels.csv", n_topics=5)

    ga = GeneticAlgorithm(corpus, labels, k_min=4, k_max=8)
    best_chromosome = ga.run(population_size=200, iterations=100, stats_step=5)

    print("best chromosome: ", best_chromosome)
    ga.plot_stats()

