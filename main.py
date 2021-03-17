import tensorflow.keras.datasets.reuters as reuters
import pathlib

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

print("Documents:")
print(x_train.shape)
print(x_train[:5])

print("\nLabels:")
print(y_train.shape)
print(y_train[:5])

print("\nWord to index mapping:", pathlib.Path().home())
data_path = pathlib.Path().home().joinpath(".keras/datasets/reuters_word_index.json")
with data_path.open() as f:
    print(f.read(100))
