from corpus_preparation import load_corpus
import numpy as np
import pathlib


class KrillHerd:
    def __init__(self, krill_count, corpus, num_clusters):
        num_docs = corpus.shape[0]
        self.rng = np.random.default_rng()
        self.corpus = corpus
        self.num_clusters = num_clusters
        self.krill_count = krill_count

        # keeps current krill's positions
        self.positions = self.rng.random(
            size=(krill_count, num_docs)) * num_clusters

        # keeps best solution encountered by each krill
        self.memory = self._position_to_solution(self.positions)

        self.fitness = self._get_fitness(self.memory)
        self.max_fitness = np.max(self.fitness)
        self.min_fitness = np.min(self.fitness)

        # init KH parameters
        self.n_max = 0.01
        self.f_max = 0.03
        self.d_max = 0.005
        self.n_old = np.zeros(shape=(krill_count, num_docs))
        self.f_old = np.zeros(shape=(krill_count, num_docs))
        self.d_old = np.zeros(shape=(krill_count, num_docs))
        self.inertia = 0.5  # todo: set better value
        # init genetic parameters

    def update_positions(self):
        pass

    def calculate_cluster_centroids(self):
        pass

    def _get_fitness(self, solution):
        fitness = []
        for j in range(solution.shape[0]):
            f = 0
            for i in range(self.num_clusters):
                s = solution[j, :] == i

                cluster = self.corpus[s]

                centroid = np.mean(cluster, axis=1)
                f += np.mean(centroid @ cluster)
            fitness.append(f)
        return np.array(fitness) / self.num_clusters

    def best_krill(self):
        """Returns position of the most fit krill"""

    def get_K(self):
        f = self.fitness[:, None]
        f_t = self.fitness[None, :]
        K = (f - f_t) / (self.max_fitness - self.min_fitness)
        return K

    def get_X(self):
        shape = (self.positions.shape[0], self.positions.shape[0],
                 self.positions.shape[1])
        X = np.zeros(shape, dtype=np.float32)
        for i in range(X.shape[0]): # for each krill
            x = self.positions[i, :]
            X[i] = self.positions - x
            X[i] /= (np.linalg.norm(X[i], axis=1)[:, None] + 1e-8)
        return X

    def get_alpha_local(self):
        K = self.get_K()
        X = self.get_X()
        alpha_l = np.zeros((self.krill_count, self.positions.shape[1]))
        for i in range(self.krill_count):
            alpha_l[i] = K[i] @ X[i]

        return alpha_l

    def best_clustering(self):
        """Returns clustering obtained by the most fit krill"""

    def move_herd(self):
        pass
        # movement induced by other krill
        alpha = self.get_alpha_local() # todo add alpha local
        n = self.n_max * alpha + self.n_old * self.inertia
        self.n_old = n

        self.positions += n

        # todo:
        #   second movement
        #   third movement
        #   don't allow krill to leave search area

    def start(self, iter=1000):

        for i in range(iter):
            self.move_herd()
            # motion calculation
            # update krill positions
            # calculate fitness?

            # applying KH operators on KH memory
            # crossover and mutation

            # replace the worst krill

            if i % 100 == 0:
                self.memory = self._position_to_solution(self.positions)
                print(self.positions[:2], "\n")

        self.memory = self._position_to_solution(self.positions)


    @staticmethod
    def _position_to_solution(positions):
        return np.floor(positions)


if __name__ == "__main__":
    corpus_path = pathlib.Path("..").joinpath("corpus.csv")
    corpus = load_corpus(corpus_path)#[:100, :]

    print(corpus.shape)

    herd = KrillHerd(10, corpus, 5)
    print(herd.memory[:3])
    herd.start()
    print(herd.memory[:3])
