import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from frog.models.model_tools import argmax_sample


class BlockNADE(tf.keras.layers.Layer):
    """
    The Neural Autoregressive Distribution Estimator with output divided in blocks of
    categorical variables.
    """

    def __init__(self, name, num_visible_units, num_hidden_units, n_timesteps):
        """Initialise various class attributes."""
        super().__init__(name)
        self.n_blocks = len(num_visible_units)
        self.visible_units_per_block = num_visible_units  # a list of units per block
        self.hidden_units = num_hidden_units
        self.bias_visible_dynamic = [Dense(n) for n in self.visible_units_per_block]
        self.bias_hidden_dynamic = Dense(self.hidden_units)
        self.n_timesteps = n_timesteps

    def build(self, input_shape):
        """
        VtoH : Visible-to-hidden weight matrix.
        HtoV : Hidden-to-visible weight matrix.
        b_vis : Visible layer biases, static part
        b_hid : Hidden layer biases, static part
        """
        # TODO: Understand if those are the best initializers
        self.VtoH = [
            self.add_weight(
                f"W_{i}",
                shape=[n, self.hidden_units],
                initializer=tf.keras.initializers.TruncatedNormal(),
                trainable=True,
                # trainable=True if i < self.n_blocks - 1 else False,
            )  # In an ordered NADE, the last line of the VtoH matrix is never trained on
            for i, n in enumerate(self.visible_units_per_block)
        ]
        self.HtoV = [
            self.add_weight(
                f"V_{i}",
                shape=[self.hidden_units, n],
                initializer=tf.keras.initializers.TruncatedNormal(),
                trainable=True,
            )
            for i, n in enumerate(self.visible_units_per_block)
        ]

    def call(self, input_tensor, training=False, targets=None, *args, **kwargs):
        if training:
            return self.train(input_tensor, targets, *args, **kwargs)
        else:
            return self.sample(input_tensor, targets, *args, **kwargs)

    def train(self, input_tensor, targets, *args, **kwargs):
        """
        @return: a list of self.n_blocks outputs, each representing the logits of the probability
         for each class in the output.
        """
        if targets is None:
            raise ValueError("Please specify targets in training mode")
        bias_vis = [b(input_tensor) for b in self.bias_visible_dynamic]  # shapes [NF](B, T, Fi)
        bias_hid = self.bias_hidden_dynamic(input_tensor)  # shape (B, T, H)

        # a list of N equal tensors with shape [NF](B, T, H), NF = number of output heads
        teacher_forcing = [tf.tensordot(t, m, [2, 0]) for t, m in zip(targets, self.VtoH)]

        output = []
        hidden_layer = bias_hid
        # Here, we can't use tf.scan because the y_pred have different shapes
        for i in range(self.n_blocks):
            hidden_layer, y_pred = self._train_step(
                hidden_layer, [self.HtoV[i], teacher_forcing[i], bias_vis[i]]
            )
            output.append(y_pred)
        return output  # shape [NF](B, T, Fi)

    def _train_step(self, hid_prev, current_elements):
        HtoV_i, teacher_forcing_i, bias_vis_i = current_elements

        # Compute next step
        logits_i = tf.tensordot(tf.nn.sigmoid(hid_prev), HtoV_i, axes=[2, 0]) + bias_vis_i
        hid_i = hid_prev + teacher_forcing_i

        return hid_i, logits_i

    def sample(self, input_tensor, targets, *args, **kwargs):
        """
        Build the BlockNADE sampler graph.
        """
        bias_vis = [b(input_tensor) for b in self.bias_visible_dynamic]  # shapes [(B, T, Vi)]
        bias_hid = self.bias_hidden_dynamic(input_tensor)  # shape (B, T, H)
        # TODO: Implement sample_from_probs with a kwarg?
        # sample_from_probs = kwargs["sample_from_probs"]
        sample_from_probs = True
        output = []
        hidden_layer = bias_hid
        for i in range(self.n_blocks):
            hidden_layer, y_samp = self._sample_step(
                hidden_layer, [self.HtoV[i], self.VtoH[i], bias_vis[i], sample_from_probs]
            )
            output.append(y_samp)
        return output

    def _sample_step(self, hid_prev, current_elements):
        HtoV_i, VtoH_i, bias_vis_i, sample_from_probs = current_elements

        logits_i = tf.tensordot(tf.nn.sigmoid(hid_prev), HtoV_i, axes=[2, 0]) + bias_vis_i
        v_i = (
            argmax_sample(logits_i, tf.shape(bias_vis_i)[-1])
            if sample_from_probs
            else tf.nn.softmax(logits_i)
        )
        h_i = hid_prev + tf.tensordot(v_i, VtoH_i, axes=[2, 0])

        return h_i, v_i

