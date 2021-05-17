import tensorflow.keras.datasets.reuters as reuters
from corpus_preparation import *

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    path="op_lab_reuters_dataset.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=2143,
    start_char=1,  # The start of a sequence will be marked with this character.
    oov_char=2,  # The out-of-vocabulary character.
    index_from=3,  # Index actual words with this index and higher.
)
reuters.get_word_index(path="reuters_word_index.json")
corpus = generate_corpus(x_train[:3000], k=100, normalized=False, filename='example_corpus.csv')
print("corpus created")
