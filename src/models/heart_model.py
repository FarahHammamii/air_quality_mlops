"""
HEART Model - REGULARIZED VERSION for small datasets
Reduced capacity + L2 regularization + gradient clipping
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .attention_preprocessor import HEARTAttentionPreprocessor


def _encoder_decoder_head(x, dropout_rate: float = 0.3):
    """
    Reduced capacity encoder-decoder with regularization.
    Parameters reduced by ~70% to prevent overfitting.
    """
    # Encoder - smaller channels
    x = layers.Conv1D(
        32, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="enc_conv1"
    )(x)
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
    x = layers.Dropout(dropout_rate, name="enc_drop")(x)

    # Decoder - smaller channels
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

    # Regression head - smaller
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="reg_dense1"
    )(x)
    x = layers.Dropout(dropout_rate, name="reg_drop")(x)
    x = layers.Dense(
        16, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="reg_dense2"
    )(x)
    outputs = layers.Dense(1, name="output")(x)
    return outputs


def create_heart_model(seq_len: int,
                       feat_dim: int,
                       num_heads: int = 2,
                       num_layers: int = 1,
                       dropout_rate: float = 0.3,
                       name: str = "heart_model"):
    """
    HEART model with regularization for small datasets.
    
    Args:
        num_heads: Reduced from 4 → 2
        num_layers: Reduced from 2 → 1  
        dropout_rate: Increased from 0.1 → 0.3
    """
    inputs = keras.Input(shape=(seq_len, feat_dim), name="input")

    # HEART attention pre-processor (reduced capacity)
    x = HEARTAttentionPreprocessor(
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_units=min(seq_len // 2, 36),  # Reduced from 72 → 36
        dropout_rate=dropout_rate,
        name="heart_attention",
    )(inputs)

    # Shared encoder-decoder
    outputs = _encoder_decoder_head(x, dropout_rate=dropout_rate)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # ADD gradient clipping for stability
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(checkpoint_path: str = "models/heart_best.keras"):
    """Callbacks with adjusted patience for smaller model"""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,  # Reduced from 20
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,  # Reduced from 8
            min_lr=1e-5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=0
        ),
    ]