from numpy.linalg import norm
from numpy import mean, amax

def cosine_similarity(a, b, normalized=True):
    if normalized:
        return a@b
    else:
        return a@b / (norm(a) * norm(b))

def Si(center_i, assigned_docs, dist_fun=cosine_similarity):
    distances = [dist_fun(center_i, doc) for doc in assigned_docs]
    return mean(distances)

def Mij(center_i, center_j, dist_fun=cosine_similarity):
    return dist_fun(center_i, center_j)

def Rij(center_i, center_j, docs_i, docs_j, dist_fun=cosine_similarity):
    S_sum = Si(center_i, docs_i, dist_fun) + Si(center_j, docs_j, dist_fun)
    return S_sum/Mij(center_i, center_j, dist_fun)

def Di(i: int, centers: list, docs_by_centers: dict, dist_fun = cosine_similarity):
    all_Rs1 = [Rij(centers[i], centers[j], docs_by_centers[i], docs_by_centers[j], dist_fun) for j in range(0,i)]
    all_Rs2 = [Rij(centers[i], centers[j], docs_by_centers[i], docs_by_centers[j], dist_fun) for j in range(i+1,len(centers))]
    if len(all_Rs1) == 0:
        return amax(all_Rs2)
    if len(all_Rs2) == 0:
        return amax(all_Rs1)
    return amax([amax(all_Rs1), amax(all_Rs2)])

def DB(centers, docs_by_centers, dist_fun=cosine_similarity):
    return mean([Di(i, centers, docs_by_centers, dist_fun) for i in range(len(centers))])

# todo assign centers to each doc - create dict {center: docs}