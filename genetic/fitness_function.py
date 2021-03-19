from numpy.linalg import norm
from numpy import mean, zeros
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
        docs_idx_by_center = {}
        for i in range(len(chromosome)):
            center_idx = chromosome[i]
            docs_idx_by_center[center_idx] = np.where(centers_ord == i)
        return docs_idx_by_center

    def Si(self, center_idx, assigned_docs_idx):
        val = mean(self.dist_matrix[center_idx, assigned_docs_idx])
        return val

    def Mij(self, center_i, center_j):
        return self.dist_matrix[center_i, center_j]

    def Rij(self, center_i, center_j, docs_idx_by_center):
        S_sum = self.Si(center_i, docs_idx_by_center[center_i]) + self.Si(center_j, docs_idx_by_center[center_j])
        return S_sum / self.Mij(center_i, center_j)

    def Di(self, ith_gene: int, chromosome, docs_idx_by_center):
        all_Rij = (self.Rij(chromosome[ith_gene], chromosome[jth_gene], docs_idx_by_center)
                      for jth_gene in range(len(chromosome)) if jth_gene != ith_gene)
        return max(all_Rij)

    def DB(self, chromosome):
        docs_idx_by_center = self.assign_centers(chromosome)
        all_Dis = [self.Di(ith_gene, chromosome, docs_idx_by_center) for ith_gene in range(len(chromosome))]
        return mean(all_Dis)

doc_by_term = np.array([[1.2, 3.1, 4],
                        [3.1, 0, 1.1],
                        [1.1, 3.1, 4],
                        [3, 0.5, 1.2]])
chrom = [0, 1]
mydb = DB(doc_by_term)
print(mydb.dist_matrix)
print(mydb.DB(chrom))
chrom = [0, 2]
print(mydb.DB(chrom))
chrom = [0,1,2]
print(mydb.DB(chrom))
chrom = [0,1,2,3]
print(mydb.DB(chrom))
# todo assign centers to each doc - create dict {center: docs}
