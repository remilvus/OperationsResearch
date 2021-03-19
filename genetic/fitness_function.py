from sklearn.metrics import davies_bouldin_score
from numpy.linalg import norm
from numpy import zeros
import numpy as np

def cosine_dist(a, b, normalized=False):
    return 1 - cosine_similarity(a, b, normalized)

def cosine_similarity(a, b, normalized: bool):
    if normalized:
        return a @ b
    else:
        return a @ b / (norm(a) * norm(b))

class DB:
    def __init__(self, doc_by_term, dist_func=cosine_dist):
        self.doc_by_term = doc_by_term
        self.N = self.doc_by_term.shape[0]
        self.dist_matrix = self.get_dist_matrix(dist_func)

    def get_dist_matrix(self, dist_func):
        dist_mx = zeros((self.N, self.N))
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                dist_mx[i, j] = dist_func(self.doc_by_term[i], self.doc_by_term[j])
                dist_mx[j, i] = dist_mx[i, j]
        return dist_mx

    def assign_centers(self, chromosome):
        """ list of closest center for each doc
        '''
        chromosome: [1 4 2]
        'doc_ind - center_doc_ind (ordinal in chromosome)' pairs:
        1 - 1 (0),  2 - 2 (2),  3 - 2 (2),  4 - 4 (1),  5 - 1 (0)
        centers_ord: [0 2 2 1 0]
        return:  {1: [1 5], 2: [2, 3], 4: [4]}
        '''
        """
        centers_ord = np.argmin(self.dist_matrix[:, chromosome], axis=1)
        return [chromosome[i] for i in centers_ord]

    def davies_bouldin(self, chromosome):
        centers = self.assign_centers(chromosome)
        return davies_bouldin_score(self.doc_by_term, centers)

