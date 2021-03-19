from corpus_preparation import load_corpus
import numpy as np
import pathlib


class KrillHerd:
    def __init__(self, krill_count, num_docs, num_clusters):
        self.rng = np.random.default_rng()

        # keeps current krill's positions
        self.positions = self.rng.random(
            size=(krill_count, num_docs)) * num_clusters

        # keeps best solution encountered by each krill
        self.memory = self._position_to_solution(self.positions)

        self.fitness = self._get_fitness()

        # init KH parameters

        # init genetic parameters

    def update_positions(self):
        pass

    def calculate_cluster_centroids(self):
        pass

    def _get_fitness(self):
        pass

    def best_krill(self):
        """Returns position of the most fit krill"""

    def best_clustering(self):
        """Returns clustering obtained by the most fit krill"""

    @staticmethod
    def _position_to_solution(positions):
        return np.floor(positions)


if __name__ == "__main__":
    corpus_path = pathlib.Path("..").joinpath("corpus.csv")
    corpus = load_corpus(corpus_path)
    print(corpus.shape)

    herd = KrillHerd(1000, corpus.shape[1], 5)
    print(herd.memory[:3])
