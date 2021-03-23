"""
Convert data loaded from keras into our problem-specific format.
Supports dimension shrinking.
Results saved as csv, to avoid repeated computations.
"""
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse.linalg import svds

# todo: stop words removal
ALL_WORDS = 30980 # how many words to consider
LRA_K = 100  # default number of dimensions to shrink document to

def generate_corpus(keras_dataset, k=LRA_K, max_words=ALL_WORDS, normalized=True, filename='corpus.csv'):
    """ generate and save to file preprocessed doc_by_term matrix """
    doc_by_term = to_document_by_term(keras_dataset, max_words)
    doc_by_term = scale_by_IDF(doc_by_term)
    if k < doc_by_term.shape[1]:
        doc_by_term = shrink_corpus(doc_by_term, k)
    if normalized:
        doc_by_term = normalize(doc_by_term)
    save_corpus(doc_by_term, filename)

def to_document_by_term(keras_dataset, max_words=ALL_WORDS):
    """ convert keras' dataset format: each document is a list of words represented as integers,
     to document_by_term matrix: each row is a document represented as bag-of-words. """
    doc_by_term = np.zeros((len(keras_dataset), max_words + 1))
    for idx, doc in enumerate(keras_dataset):
        for word in doc:
            if word < max_words + 1:
                doc_by_term[idx][word] += 1
    return delete_unused_words(doc_by_term)

def delete_unused_words(doc_by_term):
    """ deleting all-zero columns - unnecessary and problematic for IDF """
    all_zero_col = (doc_by_term == 0).all(0)
    return doc_by_term[:, ~all_zero_col]

def scale_by_IDF(doc_by_term):
    """ represent words by inverse document frequency, to add weight to less common words. """
    N = doc_by_term.shape[0]
    DF = np.count_nonzero(doc_by_term, axis=0)
    IDF = np.log2(N / DF)
    return doc_by_term*IDF

def shrink_corpus(doc_by_term, k):
    """ apply low rank approximation using first k singular values of doc_by_term (converted to sparse matrix) """
    doc_term_s = lil_matrix(doc_by_term)
    u, s, vt = svds(doc_term_s.T, k=k, which='LM')
    new_corpus = doc_term_s @ u
    return new_corpus

def normalize(doc_by_term):
    """ normalize matrix by rows, so that each document is a unit vector """
    norm = np.linalg.norm(doc_by_term, axis=1)
    return doc_by_term / norm[:, None]

def save_corpus(corpus, filename='corpus.csv', **kwargs):
    np.savetxt(filename, corpus, delimiter=',', **kwargs)

def load_corpus(filename='corpus.csv', **kwargs):
    return np.loadtxt(filename, delimiter=',', **kwargs)

