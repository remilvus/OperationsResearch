import random
from LsiPreprocessing import load_corpus

Kmin = 5
Kmax = 50

def random_chromosome(Kmin, Kmax, nr_docs):
    k = random.randint(Kmin, Kmax)
    return random.sample(range(nr_docs), k=k)

def init_population(size, nr_docs, Kmin, Kmax):
    return [random_chromosome(Kmin, Kmax, nr_docs) for _ in range(size)]

corpus = load_corpus("..\\corpus.csv")
print(corpus.shape)
population = init_population(1000, corpus.shape[0], Kmin, Kmax)
print(population[:3])