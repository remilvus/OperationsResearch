from sklearn.metrics import davies_bouldin_score, pairwise_distances_argmin

def davies_bouldin(doc_by_term, chromosome):
    centers = assign_centers_dynamic(doc_by_term, chromosome)
    db = davies_bouldin_score(doc_by_term, centers)
    return db

def assign_centers_dynamic(doc_by_term, chromosome):
    return pairwise_distances_argmin(doc_by_term, doc_by_term[chromosome])

