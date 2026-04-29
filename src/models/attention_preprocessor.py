"""
HEART Paper: Attention Pre-Processor (O-Att variant)
Faithful implementation with regularization for small datasets
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _make_qkv_network(time_steps: int, n_layers: int, hidden_units: int, name: str):
    """
    Build a Q / K / V sub-network with regularization.
    Reduced capacity for small dataset (147 sequences).
    """
    net_layers = []
    # Use smaller intermediate dimension (cap at 32)
    actual_hidden = min(hidden_units // 2, 32)
    
    for i in range(n_layers):
        if i < n_layers - 1:
            # Hidden layers with ReLU
            net_layers.append(
                layers.Dense(
                    actual_hidden, 
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name=f"{name}_dense_{i}"
                )
            )
        else:
            # Last layer: no activation, maps back to time_steps
            net_layers.append(
                layers.Dense(
                    time_steps, 
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name=f"{name}_dense_{i}"
                )
            )
    return keras.Sequential(net_layers, name=name)


class HEARTAttentionPreprocessor(keras.layers.Layer):
    """
    HEART O-Att attention pre-processor with regularization.
    """

    def __init__(self,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 hidden_units: int = 36,
                 dropout_rate: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self._T = None
        self._F = None

    def build(self, input_shape):
        _, T, F = input_shape
        self._T = T
        self._F = F

        # Create Q, K, V networks per feature and head
        self._Q, self._K, self._V = [], [], []
        for f in range(F):
            Qf, Kf, Vf = [], [], []
            for h in range(self.num_heads):
                Qf.append(_make_qkv_network(T, self.num_layers, T,
                                             name=f"Q_f{f}_h{h}"))
                Kf.append(_make_qkv_network(T, self.num_layers, T,
                                             name=f"K_f{f}_h{h}"))
                Vf.append(_make_qkv_network(T, self.num_layers, T,
                                             name=f"V_f{f}_h{h}"))
            self._Q.append(Qf)
            self._K.append(Kf)
            self._V.append(Vf)

        # Learnable c_f^h scalars with L2 regularization
        self._c = self.add_weight(
            name="c_fh",
            shape=(F, self.num_heads),
            initializer="ones",
            regularizer=tf.keras.regularizers.l2(0.001),
            trainable=True,
        )

        # Learnable residual weight
        self._residual_weight = self.add_weight(
            name="residual_weight",
            shape=(F,),
            initializer="ones",
            regularizer=tf.keras.regularizers.l2(0.001),
            trainable=True,
        )

        self._layernorm = layers.LayerNormalization(epsilon=1e-6)
        self._dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=False):
        feature_outputs = []

        for f in range(self._F):
            x_f = inputs[:, :, f]

            head_outputs = []
            for h in range(self.num_heads):
                q = self._Q[f][h](x_f, training=training)
                k = self._K[f][h](x_f, training=training)
                v = self._V[f][h](x_f, training=training)

                c = self._c[f, h]
                raw = c * tf.math.tanh(q * k)
                A = tf.nn.softmax(raw, axis=-1)
                A = self._dropout(A, training=training)

                head_out = A * v
                head_outputs.append(head_out)

            avg_heads = tf.reduce_mean(tf.stack(head_outputs, axis=0), axis=0)
            r = self._residual_weight[f]
            out = avg_heads + r * x_f
            feature_outputs.append(out)

        x = tf.stack(feature_outputs, axis=-1)
        x = self._layernorm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return config