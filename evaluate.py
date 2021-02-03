import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

from models import Encoder, BahdanauAttention, Decoder
from data import preprocess_sequence, DatasetLoader

tf.get_logger().setLevel("INFO")

DATA_PATH = "data.txt"
NUM_EXAMPLES = 1000
START_TOKEN = "^"
END_TOKEN = "$"

EMBEDDING_DIM = 32
RNN_UNITS = 32

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

BATCH_SIZE = 1

datatset_loader = DatasetLoader(DATA_PATH, BATCH_SIZE, NUM_EXAMPLES)

input_sequence_tokenizer = datatset_loader.input_sequence_tokenizer
target_sequence_tokenizer = datatset_loader.target_sequence_tokenizer

INPUT_VOCAB_SIZE = datatset_loader.input_vocab_size
TARGET_VOCAB_SIZE = datatset_loader.target_vocab_size

input_max_length = datatset_loader.input_tensor.shape[1]
target_max_length = datatset_loader.target_tensor.shape[1]

encoder = Encoder(INPUT_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
decoder = Decoder(TARGET_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()


def evaluate(word):
    word = preprocess_sequence(word)

    inputs = [input_sequence_tokenizer.word_index[i] for i in word.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=input_max_length, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = []

    hidden = [tf.zeros((1, RNN_UNITS))]
    encoder_output, encoder_hidden = encoder(inputs, hidden)

    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([target_sequence_tokenizer.word_index[START_TOKEN]], 0)

    for t in range(target_max_length):
        predictions, decoder_hidden, attention_weights = decoder(
            decoder_input, decoder_hidden, encoder_output
        )

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_sequence_tokenizer.index_word[predicted_id] == END_TOKEN:
            result = " ".join(result)
            return result
        else:
            result.append(target_sequence_tokenizer.index_word[predicted_id])

        decoder_input = tf.expand_dims([predicted_id], 0)

    result = " ".join(result)
    return result


def translate(word):
    result = evaluate(word)

    result = " ".join(result.split(" "))

    print("Input:", word)
    print("Prediction:", result)

if __name__ == "__main__":
    word = sys.argv[1]
    translate(word)
