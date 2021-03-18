"""
Convert data loaded from keras into our problem-specific format.
Supports dimension shrinking.
Results saved as csv, to avoid repeated computations.
"""
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse.linalg import svds

# todo: currently no tf_idf
ALL_WORDS = 30980 # how many words to consider
LRA_K = 100  # default number of dimensions to shrink document to

def generate_corpus(keras_dataset, k=LRA_K, max_words=ALL_WORDS):
    corpus = to_document_by_term(keras_dataset, max_words)
    new_corpus = shrink_corpus(corpus, k)
    save_corpus(new_corpus)

def to_document_by_term(keras_dataset, max_words=ALL_WORDS):
    """ convert keras' dataset format: each document is a list of words represented as integers,
     to document_by_term matrix: each row is a document represented as bag-of-words. """
    doc_by_term = np.zeros((len(keras_dataset), max_words + 1))
    for idx, doc in enumerate(keras_dataset):
        for word in doc:
            if word < max_words + 1:
                doc_by_term[idx][word] += 1
    return doc_by_term

def shrink_corpus(doc_by_term, k):
    """ apply lra using first k singular values of doc_by_term (converted to sparse matrix) """
    doc_term_s = lil_matrix(doc_by_term)
    u, s, vt = svds(doc_term_s.T, k=k, which='LM')
    new_corpus = doc_term_s @ u
    return new_corpus

def save_corpus(corpus, filename='corpus.csv'):
    np.savetxt(filename, corpus, delimiter=',')

def load_corpus(filename='corpus.csv'):
    return np.loadtxt(filename, delimiter=',')

