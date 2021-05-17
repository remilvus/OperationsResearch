from corpus_preparation import load_corpus
import numpy as np
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

EPS = 1e-8


class KrillHerd:
    def __init__(self, krill_count, corpus, num_clusters):
        num_docs = corpus.shape[0]
        self.rng = np.random.default_rng()
        self.corpus = corpus
        self.num_clusters = num_clusters
        self.krill_count = krill_count
        self.best_fitness_history = []
        self.speed_history = []
        self.sensing_history = []

        # initial solution from k-means
        # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(corpus)
        # print("KMEANS fitness ", self._get_fitness(kmeans.labels_[None, :]))
        # self.positions = 2 * (self.rng.random(size=(krill_count, num_docs)) - 0.5) + kmeans.labels_
        # self.positions = np.clip(self.positions, 0, num_clusters - EPS)

        # keeps current krill's positions
        self.positions = self.rng.random(
            size=(krill_count, num_docs)) * num_clusters

        # keeps best solution encountered by each krill
        self.memory = self._position_to_solution(self.positions)

        self.fitness = self._get_fitness(self.memory)
        self.best_fitness = self.fitness
        self.max_fitness = np.max(self.best_fitness)
        # self.min_fitness = np.min(self.best_fitness)
        self.min_fitness = np.min(self.fitness)

        # init KH parameters
        mul = 1.
        self.n_max = 0.1 * mul
        self.f_max = 0.03 * mul
        self.d_max = 0.008 * mul
        self.n_old = self.rng.random((krill_count, num_docs))
        self.f_old = self.rng.random((krill_count, num_docs))
        self.d_old = self.rng.random((krill_count, num_docs))
        self.inertia = 0.5
        self.c_t = 1.  # speed scaling factor

        # init genetic parameters
        self.genes_swaped = 0.2
        self.cross_probability = 0.1
        self.mutation_base = 0.02

        # other
        self.i_max = None
        self.i_curr = None
        self.food_position = None
        self.K_best_X_best = None

    def _get_fitness(self, solution):
        fitness = []
        for j in range(solution.shape[0]):
            f = 0
            for i in range(self.num_clusters):
                s = solution[j, :] == i
                if not np.any(s):
                    continue
                cluster = self.corpus[s] / (np.linalg.norm(self.corpus[s], axis=1)[:, None] + EPS)

                centroid = np.mean(cluster, axis=0)[:, None]
                centroid /= (np.linalg.norm(centroid) + EPS)
                f += np.mean(cluster @ centroid)
            fitness.append(f)
        return np.array(fitness) / self.num_clusters

    def get_K(self):
        """Matrix of normalised fitness difference.
        K[i, j] ~ (fitness_j - fitness_i)
        Shape: (number of krill, number of krill)"""
        f = self.fitness[:, None]
        f_t = self.fitness[None, :]
        if self.max_fitness - self.min_fitness < EPS:
            return np.zeros((self.krill_count, self.krill_count))
        K = (f - f_t) / (self.max_fitness - self.min_fitness)
        return K.T  # transpose instead of -1 multiplication

    def get_X(self):
        """Returns matrix of krill distance-vectors.
        Result has shape (number of krill, number of krill, num of parameters).
        Example: vector from krill `i` to krill `j` is given by X[i,j,:]"""
        shape = (self.positions.shape[0], self.positions.shape[0],
                 self.positions.shape[1])
        X = np.zeros(shape, dtype=np.float32)
        for i in range(X.shape[0]): # for each krill
            x = self.positions[i, :]
            X[i] = self.positions - x
        return X

    def get_X_best(self): # output was manually checked
        """Return arrays of vectors to krills' positions from their respective
        best seen solutions. Shape: (number of krill, number of documents)"""
        diff = self.positions - (self.memory + 0.5)  # moving solution away from the edge (floor function)
        diff_norm = np.linalg.norm(diff)
        X_best = np.multiply(1 / (diff_norm.reshape(-1, 1) + EPS), diff)
        return X_best

    def get_K_best(self): # output was manually checked
        """K_best shape: (number of krill,), values not grater than 0"""
        if self.max_fitness - self.min_fitness < EPS:
            return np.zeros((self.krill_count,))
        K_best = (self.fitness - self.best_fitness) / (self.max_fitness - self.min_fitness)
        return K_best

    def get_K_best_X_best(self):
        """Returns array of scaled by K_best vectors. Vectors are direced from
        krills' positions to their respective best solution.
        Shape: (number of krill, number of documents)"""
        K_best = self.get_K_best()
        X_best = self.get_X_best()
        K_best_X_best = np.multiply(K_best.reshape(-1, 1), X_best)
        return K_best_X_best

    def get_alpha_local(self):
        """Calculate alpha local for each krill. Shape: (num krill, num documents)"""
        K = self.get_K()
        X = self.get_X()

        alpha_l = np.zeros((self.krill_count, self.positions.shape[1]))
        for i in range(self.krill_count):
            distance_vectors = X[i]
            distances = np.linalg.norm(distance_vectors, axis=1)
            sensing_distance = np.mean(distances)
            idx = distances < sensing_distance
            self.sensing_history.append(np.sum(idx))
            alpha_l[i] = K[i, idx] @ (X[i, idx] / (np.linalg.norm(X[i, idx], axis=1)[:, None] + EPS))
        return alpha_l

    def get_alpha_target(self):
        """Computes alpha_target."""
        rand = self.rng.random((self.krill_count, 1))

        # moving towards best seen solution
        C_best = 2*(rand + self.i_curr/self.i_max)
        alpha_t = np.multiply(C_best, self.K_best_X_best)
        return alpha_t

    def get_food_position(self):
        #food_position = (np.sum(np.multiply(1 / (self.fitness.reshape(-1, 1) + EPS), self.positions), axis=0)
        #         / np.sum(1 / (self.fitness + EPS)))
        food_position = (np.sum(np.multiply(self.fitness.reshape(-1, 1), self.positions), axis=0)
                 / np.sum(self.fitness))
        # global labels
        # return labels
        return food_position

    def get_X_food(self):
        diff = self.positions - self.food_position
        diff_norm = np.linalg.norm(diff)
        X_food = np.multiply(1 / (diff_norm.reshape(-1, 1) + EPS), diff)
        return X_food

    def get_K_food(self):
        food_fitness = self._get_fitness(self._position_to_solution(self.food_position.reshape(1, -1)))
        if self.max_fitness - self.min_fitness < EPS:
            return np.zeros((self.krill_count, ))
        K_food = (self.fitness - food_fitness) / (self.max_fitness - self.min_fitness)
        return K_food

    def get_beta_food(self):
        C_food = 2*(1 - self.i_curr/self.i_max)
        K_food = self.get_K_food()
        X_food = self.get_X_food()
        beta_f = C_food * np.multiply(K_food.reshape(-1, 1), X_food)
        return beta_f

    def get_beta_best(self):
        return self.K_best_X_best

    def best_krill(self):
        """Returns position of the most fit krill. Assumes that self.best_fitness
        and self.memory are regularly updated."""
        best_krill = np.argmax(self.best_fitness)
        return self.positions[best_krill]

    def best_clustering(self):
        """Returns clustering obtained by the most fit krill. Assumes that self.best_fitness
        and self.memory are regularly updated."""
        best_krill = np.argmax(self.best_fitness)
        return self.memory[best_krill]

    def scale_positions(self):
        """Scales krill positions."""
        # for i in range(self.positions.shape[1]):
        #     too_big = self.positions[:, i] >= self.num_clusters
        #     too_small = self.positions[:, i] < 0
        #     rand_small = self.rng.random(np.sum(too_small))
        #     rand_big = self.rng.random(np.sum(too_big))
        #     self.positions[too_small, i] = self.rng.random(np.sum(too_small)) * self.memory[too_small, i]
        #     self.positions[too_big, i] = rand_big * self.memory[too_big, i] + (1 - rand_big) * self.num_clusters

        self.positions -= np.min(self.positions, axis=1, keepdims=True)
        self.positions /= np.max(self.positions, axis=1, keepdims=True)
        self.positions *= self.num_clusters

        # self.positions = np.clip(self.positions, 0, self.num_clusters-EPS)

    def move_herd(self):
        # movement induced by other krill
        self.K_best_X_best = self.get_K_best_X_best()
        alpha = self.get_alpha_local() + self.get_alpha_target()
        n = self.n_max * alpha + self.n_old * self.inertia
        self.n_old = n

        self.food_position = self.get_food_position()
        beta = self.get_beta_food() + self.get_beta_best()
        f = self.f_max * beta + self.f_old * self.inertia
        self.f_old = f

        d = self.d_max * (1 - self.i_curr / self.i_max) * self.rng.random(self.positions.shape)
        # log sizes of speed components
        self.positions += self.c_t * self.num_clusters * (n + f + d)

        self.speed_history.append((np.linalg.norm(n),
                                   np.linalg.norm(f),
                                   np.linalg.norm(d)))
        self.scale_positions()

    def start(self, iter=100):
        self.i_max = iter

        bar = tqdm(range(iter))
        for i in bar:
            self.i_curr = i
            self.move_herd()

            # applying KH operators on KH memory

            # crossover and mutation
            if (self.max_fitness - self.min_fitness) < EPS:
                K = np.zeros((self.krill_count, ))
            else:
                K = (self.fitness - self.max_fitness) / (
                            self.max_fitness - self.min_fitness)

            cross_p = 1. + 0.5 * K  # K is negative
            cross_p /= np.sum(cross_p)
            for k in range(self.krill_count):
                if self.rng.random() < self.cross_probability:
                    other_idx = self.rng.choice(np.arange(0, self.krill_count), p=cross_p)
                    cross_idx = self.rng.random((self.positions.shape[1], )) < self.genes_swaped
                    # crossover
                    position_tmp = self.positions[k, cross_idx]
                    self.positions[k, cross_idx] = self.positions[other_idx, cross_idx]
                    self.positions[other_idx, cross_idx] = position_tmp
                    # mutation
                    mu = self.mutation_base / self.fitness[k]
                    mutation_idx = self.rng.random((self.positions.shape[1], )) < mu
                    self.positions[k, mutation_idx] = self.rng.random((np.sum(mutation_idx, ))) * self.num_clusters

            # update krill memory and fitness
            new_memory = self._position_to_solution(self.positions)
            self.fitness = self._get_fitness(new_memory)
            to_update = self.fitness > self.best_fitness
            self.memory[to_update] = new_memory[to_update]
            self.best_fitness[to_update] = self.fitness[to_update]
            self.max_fitness = np.max(self.fitness)
            self.min_fitness = np.min(self.fitness)

            # replace the worst krill with the best solution (assumption:
            #                                       best krill = best solution)
            best_krill = np.argmax(self.fitness)
            worst_krill = np.argmax(-self.fitness)
            # self.positions[worst_krill, :] = self.memory[best_krill]
            # self.n_old[worst_krill, :] = self.n_old[best_krill]
            # self.f_old[worst_krill, :] = self.f_old[best_krill]
            # self.d_old[worst_krill, :] = self.d_old[best_krill]

            self.best_fitness_history.append(self.fitness[best_krill])
            bar.set_description(f"best fitness = {np.max(self.fitness)}")

    @staticmethod
    def _position_to_solution(positions):
        return np.floor(positions)


def visualise_process(herd: KrillHerd):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    fitness = moving_average(herd.best_fitness_history, 50)
    plt.plot(fitness)
    plt.xlabel("numer iteracji")
    plt.ylabel("maksymalny 'fitness'")
    plt.show()

    plt.hist(herd.sensing_history, bins=20)
    plt.xlabel("Number of krill inside sensing distance")
    plt.show()

    plt.plot(list(map(lambda x: x[1], herd.speed_history)), label="f")
    plt.plot(list(map(lambda x: x[0], herd.speed_history)), label="n")
    plt.xlabel("iteration")
    plt.ylabel("speed")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.plot(list(map(lambda x: x[2], herd.speed_history)), label="d")
    plt.xlabel("iteration")
    plt.ylabel("speed")
    plt.legend()
    plt.show()


def get_corpus_subset(DOCUMENTS, REAL_CLUSTERS):
    corpus_path = pathlib.Path("..").joinpath("corpus4k60.csv")
    labels_path = pathlib.Path("..").joinpath("labels.csv")
    corpus = load_corpus(corpus_path)
    labels = np.loadtxt(labels_path, delimiter='\n')[:4000]
    idx = np.logical_and(labels > 0, labels < (REAL_CLUSTERS + 1))

    corpus = corpus[idx, :]
    labels = labels[idx]

    freq = np.zeros(corpus.shape[0])
    for i in range(1, 1 + REAL_CLUSTERS):
        freq[i == labels] = np.sum(i == labels) / len(labels)
    p = 1 / freq
    p /= np.sum(p)
    idx_document = np.random.choice(np.arange(0, len(corpus)), p=p,
                                    size=DOCUMENTS, replace=False)
    corpus = corpus[idx_document, :]
    labels = labels[idx_document]
    return corpus, labels - 1


def get_blobs():
    from sklearn.datasets import make_blobs
    corpus, labels = make_blobs(n_samples=100, n_features=3, centers=[[0, 1, 1], [1, -1, -1], [-1, 0, 0]])
    corpus = corpus / np.linalg.norm(corpus, axis=1)[:, None]
    return corpus, labels


if __name__ == "__main__":
    # Parameters for loading text dataset (not used when using generated blobs)
    DOCUMENTS = 50
    REAL_CLUSTERS = 3
    # Parameters defining krill herd
    KRILL_CLUSTERS = 3
    KRILL_COUNT = 25
    ITERS = 2500

    # corpus, labels = get_corpus_subset(DOCUMENTS, REAL_CLUSTERS)

    corpus, labels = get_blobs()

    idx = np.argsort(labels)
    corpus = corpus[idx]
    labels = labels[idx]
    # print(corpus[:3])
    print("Corpus shape:", corpus.shape)

    herd = KrillHerd(KRILL_COUNT, corpus, KRILL_CLUSTERS)
    # print("memory before\n", herd.memory[0])
    herd.start(iter=ITERS)
    # print("memory after\n", herd.memory[0])
    print("Best clustering\n", herd.best_clustering())

    # for i in range(3):
    #     print(herd.memory[i], herd.best_fitness[i])

    visualise_process(herd)
    print("Correct labels:\n", labels)