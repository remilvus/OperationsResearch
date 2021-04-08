from corpus_preparation import load_corpus
import numpy as np
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt

EPS = 1e-8


class KrillHerd:
    def __init__(self, krill_count, corpus, num_clusters):
        num_docs = corpus.shape[0]
        self.rng = np.random.default_rng()
        self.corpus = corpus
        self.num_clusters = num_clusters
        self.krill_count = krill_count
        self.best_fitness_history = []

        # keeps current krill's positions
        self.positions = self.rng.random(
            size=(krill_count, num_docs)) * num_clusters

        # keeps best solution encountered by each krill
        self.memory = self._position_to_solution(self.positions)

        self.fitness = self._get_fitness(self.memory)
        self.best_fitness = self.fitness
        self.max_fitness = np.max(self.best_fitness)
        self.min_fitness = np.min(self.best_fitness)

        # init KH parameters
        mul = 0.1
        self.n_max = 0.1 * mul
        self.f_max = 0.3 * mul
        self.d_max = 0.5 * 10e-10 * mul
        self.n_old = np.zeros(shape=(krill_count, num_docs))
        self.f_old = np.zeros(shape=(krill_count, num_docs))
        self.d_old = np.zeros(shape=(krill_count, num_docs))
        self.inertia = 0.5  # todo: set better value
        self.c_t = 1.  # todo: set better value
        self.dt = self.num_clusters / 2 # unused for now

        # init genetic parameters

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
                cluster = self.corpus[s]

                # print("-"*5)
                # print(s)
                # print("cluster:\n", cluster)
                centroid = np.mean(cluster, axis=0)[:, None]
                # print("centroid:\n", centroid)
                # print('\n', cluster @ centroid, '\n', np.mean(cluster @ centroid))
                f += np.mean(cluster @ centroid)
            fitness.append(f)
            # print("F", fitness)
        return np.array(fitness) / self.num_clusters

    def get_K(self):
        f = self.fitness[:, None]
        f_t = self.fitness[None, :]
        K = (f - f_t) / (self.max_fitness - self.min_fitness + EPS)
        return K

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
            X[i] /= (np.linalg.norm(X[i], axis=1)[:, None] + EPS)
        print(X.shape)
        return X

    def get_X_best(self):
        diff = self.positions - self.memory
        diff_norm = np.linalg.norm(diff)
        X_best = np.multiply(1 / (diff_norm.reshape(-1, 1) + EPS), diff)
        return X_best

    def get_K_best(self):
        K_best = (self.fitness - self.best_fitness) / (self.max_fitness - self.min_fitness + EPS)
        return K_best

    def get_K_best_X_best(self):
        K_best = self.get_K_best()
        X_best = self.get_X_best()
        K_best_X_best = np.multiply(K_best.reshape(-1, 1), X_best)
        return K_best_X_best

    def get_alpha_local(self):
        """Calculate alpha local for each krill. Shape: (num krill, num documents)"""
        K = self.get_K()
        X = self.get_X()

        # dist_vec = vectors from i_th krill to all other
        # dist = distance from i_th krill to all other
        # sensing dist = mean(dist) / 5
        # # calculated only for krill inside sensing distance
        # alpha_l = - (fitness other - fitness i_th) / (max fitness - worst fitness) / dist other
        # alpha_l *= dist_vec other

        alpha_l = np.zeros((self.krill_count, self.positions.shape[1]))
        for i in range(self.krill_count):
            distances = np.linalg.norm(self.positions[i, :] - self.positions, axis=1)
            sensing_distance = np.sum(distances) / (self.krill_count)
            alpha_l[i] = (K[i] * (distances < sensing_distance)) @ X[i]
        return alpha_l

    def get_alpha_target(self):
        """Computes alpha_target."""
        rand = self.rng.random((self.krill_count, 1))
        C_best = -2 * (1 + rand * self.i_curr / self.i_max)
        # alpha_t = (worst krill - i_th krill) / (max fitness - worst fitness)
        # alpha_t = alpha_t * (worst_krill_pos - ith krill position) / ||(worst_krill_pos - ith krill position)||

        C_best = 2*(self.rng.random((self.krill_count, 1)) + self.i_curr/self.i_max)
        alpha_t = np.multiply(C_best, self.K_best_X_best)
        return alpha_t

    def get_food_position(self):
        return (np.sum(np.multiply(1 / (self.fitness.reshape(-1, 1) + EPS), self.positions), axis=0)
                / np.sum(1 / (self.fitness + EPS)))

    def get_X_food(self):
        diff = self.positions - self.food_position
        diff_norm = np.linalg.norm(diff)
        X_food = np.multiply(1 / (diff_norm.reshape(-1, 1) + EPS), diff)
        return X_food

    def get_K_food(self):
        food_fitness = self._get_fitness(self._position_to_solution(self.food_position.reshape(1, -1)))
        K_food = (self.fitness - food_fitness) / (self.max_fitness - self.min_fitness + EPS)
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
        # TODO: try different approaches (?)
        for i in range(self.positions.shape[1]):
            too_big = self.positions[:, i] >= self.num_clusters
            too_small = self.positions[:, i] < 0
            rand = self.rng.random()
            self.positions[too_small, i] = rand * self.memory[too_small, i]
            self.positions[too_big, i] = rand * self.memory[too_big, i] + (1 - rand) * self.num_clusters

        # self.positions -= np.min(self.positions, axis=1, keepdims=True)
        # self.positions /= np.max(self.positions, axis=1, keepdims=True)
        # self.positions *= self.num_clusters
        # self.positions = np.clip(self.positions, 0, self.num_clusters+1)

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
        # print(np.max(d), np.max(f), np.max(n))
        self.positions += self.c_t * self.num_clusters * self.corpus.shape[0] * (n + f)# + d)
        # self.positions += self.c_t * self.num_clusters * (n + f + d)
        self.scale_positions()

    def start(self, iter=1000):
        self.i_max = iter

        for i in tqdm(range(iter)):
            self.i_curr = i
            self.move_herd()

            # applying KH operators on KH memory
            # crossover and mutation

            # update krill memory and fitness
            new_memory = self._position_to_solution(self.positions)
            self.fitness = self._get_fitness(new_memory)
            to_update = self.fitness > self.best_fitness
            self.memory[to_update] = new_memory[to_update]
            self.best_fitness[to_update] = self.fitness[to_update]
            self.max_fitness = np.max(self.best_fitness)
            self.min_fitness = np.min(self.best_fitness)

            # TODO: Implement genetic operations

            # replace the worst krill with the best solution (assumption:
            #                                       best krill = best solution)
            best_krill = np.argmax(self.fitness)
            worst_krill = np.argmax(-self.fitness)
            # self.positions[worst_krill, :] = self.memory[best_krill]
            # self.n_old[worst_krill, :] = self.n_old[best_krill]
            # self.f_old[worst_krill, :] = self.f_old[best_krill]
            # self.d_old[worst_krill, :] = self.d_old[best_krill]

            if i % 100 == 0:
                print("best fitness:", self.fitness[best_krill])

            self.best_fitness_history.append(self.fitness[best_krill])


    @staticmethod
    def _position_to_solution(positions):
        return np.floor(positions)


if __name__ == "__main__":
    corpus_path = pathlib.Path("..").joinpath("corpus4k60.csv")
    labels_path = pathlib.Path("..").joinpath("labels.csv")
    corpus = load_corpus(corpus_path)#[:100, :]
    # corpus = np.array([[0, 1, 0],
    #                    [-0.1, 1, 0],
    #                    [1., -0.1, 0],
    #                    [1., 0, 0],
    #                    [1., -0.2, 0],
    #                    [-0.5, -0.5, 0],
    #                    [-0.5, -0.7, 0],
    #                    [-0.3, -0.8, 0],
    #                    [1., 0, 4],
    #                    [1., -0.2, 4],
    #                    [-0.5, -0.5, 4],
    #                    [-0.5, -0.7, 4],
    #                    [-0.3, -0.8, 4]
    #                    ])
    from sklearn.datasets import make_blobs
    corpus, labels = make_blobs(n_samples=10, n_features=3, centers=[[1, -1, -1], [-1, 0, 0], [0, 1, 1]])

    corpus = corpus / np.linalg.norm(corpus, axis=1)[:, None]
    idx = np.argsort(labels)
    corpus = corpus[idx]
    labels = labels[idx]
    print(corpus)
    print("corpus shape", corpus.shape)

    herd = KrillHerd(3, corpus, 3) # krill num: 25
    print("positions:\n", herd.positions)
    print("memory before\n", herd.memory[:3])
    herd.start(iter=400)
    print("memory after\n", herd.memory[:3])
    print("best clustering\n", herd.best_clustering())
    real_classes = np.loadtxt(labels_path, delimiter='\n')
    # print(real_classes[:corpus.shape[0]])
    print("positions:\n", herd.positions)
    for i in range(len(herd.fitness)):
        print(herd.memory[i], herd.best_fitness[i])
    plt.plot(herd.best_fitness_history)
    plt.show()
    print(labels)