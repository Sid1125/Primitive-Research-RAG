"""
Keras Siamese text embedding models.

Supported encoder architectures:
  - bilstm
  - bigru
  - cnn
"""

import tensorflow as tf
from tensorflow.keras import Model, layers


@tf.keras.utils.register_keras_serializable(package="SIC")
class L2Normalization(layers.Layer):
    """Custom L2 Normalization layer that is serializable."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_encoder(
    vocab_size: int,
    embedding_dim: int,
    lstm_units: int,
    dense_units: int,
    max_seq_length: int,
    architecture: str = "bilstm",
) -> Model:
    """Build the shared encoder branch of the Siamese network."""
    inputs = layers.Input(shape=(max_seq_length,), name="text_input")

    x = layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=max_seq_length,
        name="embedding",
    )(inputs)

    architecture = architecture.lower()
    if architecture == "bilstm":
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, name="lstm"),
            name="bilstm",
        )(x)
    elif architecture == "bigru":
        x = layers.Bidirectional(
            layers.GRU(lstm_units, return_sequences=False, name="gru"),
            name="bigru",
        )(x)
    elif architecture == "cnn":
        x = layers.Conv1D(filters=lstm_units * 2, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    else:
        raise ValueError(
            f"Unsupported encoder architecture '{architecture}'. "
            "Choose from: bilstm, bigru, cnn."
        )

    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = L2Normalization(name="l2_normalize")(x)

    return Model(inputs, x, name=f"text_encoder_{architecture}")


def build_siamese_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dense_units: int = 128,
    max_seq_length: int = 200,
    architecture: str = "bilstm",
) -> tuple:
    """Build the full Siamese network used during training."""
    encoder = build_encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dense_units=dense_units,
        max_seq_length=max_seq_length,
        architecture=architecture,
    )

    input_a = layers.Input(shape=(max_seq_length,), name="input_a")
    input_b = layers.Input(shape=(max_seq_length,), name="input_b")

    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)

    cosine_sim = layers.Dot(axes=1, normalize=False, name="cosine_similarity")(
        [encoded_a, encoded_b]
    )
    siamese = Model([input_a, input_b], cosine_sim, name=f"siamese_{architecture}")
    return siamese, encoder


def contrastive_loss(y_true, y_pred, margin=0.5):
    """Contrastive loss over cosine similarity."""
    y_true = tf.cast(y_true, tf.float32)
    loss = y_true * tf.square(1 - y_pred) + (1 - y_true) * tf.square(tf.maximum(y_pred - margin, 0))
    return tf.reduce_mean(loss)
