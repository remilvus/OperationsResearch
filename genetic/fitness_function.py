from sklearn.metrics import davies_bouldin_score, pairwise_distances_argmin
import itertools
import psutil
import ray


def idb_score(doc_by_term, chromosomes):
    """ calculate inverse davies bouldin score for each chromosome"""
    return [1 / davies_bouldin_index(doc_by_term, chromosome) for chromosome in chromosomes]


def davies_bouldin_index(doc_by_term, centers):
    centers = assign_centers_dynamic(doc_by_term, centers)
    db = davies_bouldin_score(doc_by_term, centers)
    return db


def assign_centers_dynamic(doc_by_term, chromosome):
    return pairwise_distances_argmin(doc_by_term, doc_by_term[chromosome])


def idb_score_multi(doc_by_term, chromosomes_list):
    """ idb_score but with parallel computations """
    num_cores = psutil.cpu_count()
    doc_id = ray.put(doc_by_term)  # put in shared memory (read only)
    results = ray.get([idb_score_worker.remote(doc_id, chrom_chunk) for chrom_chunk in chunks(chromosomes_list, num_cores)])
    results_flat = list(itertools.chain(*results))
    return results_flat

@ray.remote
def idb_score_worker(doc_by_term, chromosomes):
    return [1 / davies_bouldin_index(doc_by_term, chromosome) for chromosome in chromosomes]

def chunks(lst, n):
    """ Split list into n chunks. """
    chunk_size = (len(lst)+n-1)//n
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


