import numpy as np
import os
import time

import tensorflow as tf
from models import Encoder, BahdanauAttention, Decoder
from data import DatasetLoader

DATA_PATH = "data.txt"
NUM_EXAMPLES = 1000
START_TOKEN = "^"
END_TOKEN = "$"

EMBEDDING_DIM = 32
RNN_UNITS = 32

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

BATCH_SIZE = 1
EPOCHS = 40
LERANING_RATE = 0.01

datatset_loader = DatasetLoader(DATA_PATH, BATCH_SIZE, NUM_EXAMPLES)

dataset = datatset_loader.get_dataset()

INPUT_VOCAB_SIZE = datatset_loader.input_vocab_size
TARGET_VOCAB_SIZE = datatset_loader.target_vocab_size

target_sequence_tokenizer = datatset_loader.target_sequence_tokenizer

STEPS_PER_EPOCH = datatset_loader.input_tensor_length // BATCH_SIZE

encoder = Encoder(INPUT_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
decoder = Decoder(TARGET_VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [target_sequence_tokenizer.word_index[START_TOKEN]] * BATCH_SIZE, 1
        )

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == "__main__":

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(STEPS_PER_EPOCH)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 10 == 0:
                print(
                    "Epoch {} Batch {} Loss {:.4f}".format(
                        epoch + 1, batch, batch_loss.numpy()
                    )
                )

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / STEPS_PER_EPOCH))
        print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

