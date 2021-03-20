from sklearn.metrics import davies_bouldin_score, pairwise_distances_argmin

def calculate_invDB_fitness(doc_by_term, chromosomes_list):
    return [1/davies_bouldin(doc_by_term, chromosome) for chromosome in chromosomes_list]

def davies_bouldin(doc_by_term, centers):
    centers = assign_centers_dynamic(doc_by_term, centers)
    db = davies_bouldin_score(doc_by_term, centers)
    return db

def assign_centers_dynamic(doc_by_term, chromosome):
    return pairwise_distances_argmin(doc_by_term, doc_by_term[chromosome])

