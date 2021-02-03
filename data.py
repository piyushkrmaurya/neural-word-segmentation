import io
import tensorflow as tf
from sklearn.model_selection import train_test_split

START_TOKEN = "^"
END_TOKEN = "$"


def preprocess_sequence(w):
    w = w.strip()
    w = " ".join([START_TOKEN, w, END_TOKEN])
    return w


class DatasetLoader:
    def __init__(self, data_path, batch_size=1, num_examples=1000, split_ratio=0.1):
        (
            input_tensor,
            target_tensor,
            input_sequence_tokenizer,
            target_sequence_tokenizer,
        ) = self.load_dataset(data_path, num_examples)

        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

        self.input_sequence_tokenizer = input_sequence_tokenizer
        self.target_sequence_tokenizer = target_sequence_tokenizer

        (
            input_tensor_train,
            input_tensor_val,
            target_tensor_train,
            target_tensor_val,
        ) = train_test_split(input_tensor, target_tensor, test_size=split_ratio)

        self.splits = {
            "TRAIN": (input_tensor_train, target_tensor_train),
            "TEST": (input_tensor_val, target_tensor_val),
        }

        self.input_tensor_length = len(input_tensor_train)
        self.input_vocab_size = len(input_sequence_tokenizer.word_index) + 1
        self.target_vocab_size = len(target_sequence_tokenizer.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)
        ).shuffle(self.input_tensor_length)

        self.dataset = dataset.batch(batch_size, drop_remainder=True)

    def create_dataset(self, path, num_examples=None):
        lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
        word_pairs = [
            [preprocess_sequence(w) for w in l.split("\t")]
            for l in lines[:num_examples]
        ]
        input_sequence = [word_pair[0] for word_pair in word_pairs]
        target_sequence = [word_pair[1] for word_pair in word_pairs]
        return input_sequence, target_sequence

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(split=" ", filters="")
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        input_sequence, target_sequence = self.create_dataset(path, num_examples)
        input_tensor, input_sequence_tokenizer = self.tokenize(input_sequence)
        target_tensor, target_sequence_tokenizer = self.tokenize(target_sequence)
        return (
            input_tensor,
            target_tensor,
            input_sequence_tokenizer,
            target_sequence_tokenizer,
        )

    def get_dataset(self):
        return self.dataset
