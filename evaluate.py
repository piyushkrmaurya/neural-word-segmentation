import sys

import tensorflow as tf
from models import Encoder, BahdanauAttention, Decoder

START_TOKEN = "^"
END_TOKEN = "$"

EMBEDDING_DIM = 32
RNN_UNITS = 32

CHECKPOINT_DIR = "training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

BATCH_SIZE = 1

encoder = Encoder(INPUT_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
decoder = Decoder(TARGET_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sequence(sentence)

    inputs = [input_sequence.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, RNN_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_sequence.word_index[START_TOKEN]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_sequence.index_word[predicted_id] + " "

        if target_sequence.index_word[predicted_id] == END_TOKEN:
            return result, sentence

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)

    result = " ".join(result.split(" "))
    sentence = " ".join(sentence.split(" "))

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))


if __name__ == "__main__":
    sentence = sys.argv[1]
    translate(sentence)
