import random
from corpus_preparation import load_corpus
from genetic.fitness_function import davies_bouldin
from heapq import nsmallest
import numpy as np
Kmin = 5
Kmax = 49

def random_chromosome(Kmin, Kmax, nr_docs):
    k = random.randint(Kmin, Kmax)
    return random.sample(range(nr_docs), k=k)

class GeneticAlgorithm:
    def __init__(self, doc_by_term):
        self.doc_by_term = doc_by_term

    def run(self, population_size=100, iterations=1000, Kmin=3, Kmax=10):
        population = self.init_population(population_size, Kmin, Kmax)
        parents = self.choose_best(population_size//2, population)
        print(parents)

    def init_population(self, size, Kmin, Kmax):
        return np.array([random_chromosome(Kmin, Kmax, self.doc_by_term.shape[0]) for _ in range(size)], dtype=object)

    def choose_best(self, size, population):
        dbs = [davies_bouldin(self.doc_by_term, chrom) for chrom in population]
        best_idx = nsmallest(size, range(len(dbs)), key=lambda idx: dbs[idx])
        return population[best_idx]


corpus = load_corpus("..\\corpus.csv")
print(corpus.shape)
ga = GeneticAlgorithm(corpus)
ga.run(100)
