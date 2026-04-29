"""
Baseline Model: Conv1D Encoder-Decoder with regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_baseline_model(seq_len: int, feat_dim: int, name: str = "baseline_conv"):
    """
    Encoder-decoder conv baseline with regularization.
    """
    inputs = keras.Input(shape=(seq_len, feat_dim), name="input")

    # Encoder
    x = layers.Conv1D(
        32, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="enc_conv1"
    )(inputs)
    x = layers.Conv1D(
        32, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="enc_conv2"
    )(x)
    x = layers.Conv1D(
        64, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="enc_conv3"
    )(x)
    x = layers.Dropout(0.2, name="enc_drop")(x)

    # Decoder
    x = layers.Conv1D(
        32, kernel_size=1, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dec_conv1"
    )(x)
    x = layers.Conv1D(
        16, kernel_size=1, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="dec_conv2"
    )(x)

    # Regression head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="reg_dense1"
    )(x)
    x = layers.Dropout(0.2, name="reg_drop")(x)
    x = layers.Dense(
        16, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="reg_dense2"
    )(x)
    outputs = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(checkpoint_path: str = "models/baseline_best.keras"):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=0
        ),
    ]