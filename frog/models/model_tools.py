"""
The definition of all the models used.
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import AveragePooling1D, BatchNormalization, Conv1D, Dense
from tensorflow.keras.models import Model

eps = np.finfo(np.float32).eps  # This is the smallest float that can be represented


class TimeOut(Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes

    def on_train_batch_end(self, batch, logs=None):
        minutes = (time.time() - self.t0) / 60
        if minutes > self.timeout:  # 58 minutes
            print(f"\nReached {minutes:.3f} minutes of training, stopping")
            self.model.stop_training = True


class DenseNetBlock1D(Model):
    """
    See Huang G, Liu Z, van der Maaten L, and Weinberger KQ (2016), Densely Connected Convolutional
     Networks. http://arxiv.org/abs/1608.06993
    This implement the Hl layer described in section 3, DenseNets, in the version DenseNet-B
     described in paragraph "Bottleneck layers." There is a difference: ReLU is applied before
     Batch Normalization, as it became customary later on. Also, our convolutions are 1D.
    """

    def __init__(self, name, filters, bottleneck_filters, kernel_size):
        super().__init__(name=name)

        self.bn1a = BatchNormalization()
        self.conv1a = Conv1D(bottleneck_filters, kernel_size=1, padding="same")

        self.bn1b = BatchNormalization()
        self.conv1b = Conv1D(filters, kernel_size=kernel_size, padding="same")

    def call(self, input_tensor, training=False, **kwargs):
        y = tf.nn.relu(input_tensor)  # Apply ReLU before BN, unlike the paper
        y = self.bn1a(y, training=training)
        y = self.conv1a(y)
        y = tf.nn.relu(y)
        y = self.bn1b(y, training=training)
        y = self.conv1b(y)
        return tf.concat([input_tensor, y], axis=-1)


class PoolingBlock1D(Model):
    """
    See Huang G, Liu Z, van der Maaten L, and Weinberger KQ (2016), Densely Connected Convolutional
     Networks. http://arxiv.org/abs/1608.06993
    This implement the transition layer described in section 3, DenseNets, as described in paragraph
     "Pooling layers." The only difference is that our layers are 1D.
    """
    def __init__(self, name, filters, pooling_size):
        super().__init__(name=name)

        self.conv1 = Conv1D(filters, kernel_size=1, padding="same")
        self.bn = BatchNormalization()
        self.ap1 = AveragePooling1D(pooling_size, padding="same")

    def call(self, input_tensor, training=False, **kwargs):
        y = self.bn(input_tensor, training=training)
        y = self.conv1(y)
        y = self.ap1(y)
        return y


class ConvBlock(Model):
    def __init__(self, name, params):
        super().__init__(name=name)
        self.dnb1 = [
            DenseNetBlock1D(
                f"DenseBlock_1-{i}",
                params["filters_dnb_1"],
                params["bottleneck_filters_dnb_1"],
                params["kernel_size_dnb_1"],
            )
            for i in range(params["num_dnb_1"])
        ]
        self.dnb2 = [
            DenseNetBlock1D(
                f"DenseBlock_2-{i}",
                params["filters_dnb_2"],
                params["bottleneck_filters_dnb_2"],
                params["kernel_size_dnb_2"],
            )
            for i in range(params["num_dnb_2"])
        ]
        self.dnb3 = [
            DenseNetBlock1D(
                f"DenseBlock_3-{i}",
                params["filters_dnb_3"],
                params["bottleneck_filters_dnb_3"],
                params["kernel_size_dnb_3"],
            )
            for i in range(params["num_dnb_3"])
        ]
        self.pb1 = PoolingBlock1D(f"PoolingBlock_1", params["filters_pb_1"], 2)
        self.pb2 = PoolingBlock1D(f"PoolingBlock_2", params["filters_pb_2"], 2)

    def call(self, x, training=False, **kwargs):
        for dnl in self.dnb1:
            x = dnl(x, training=training)
        x = self.pb1(x, training=training)
        for dnl in self.dnb2:
            x = dnl(x, training=training)
        x = self.pb2(x, training=training)
        for dnl in self.dnb3:
            x = dnl(x, training=training)
        return x


class DilatedConvBlock(Model):
    def __init__(self, name, params):
        super().__init__(name=name)
        self.dcb = [
            Conv1D(params["filters_dcl"], kernel_size=3, padding="same", dilation_rate=3 ** i)
            for i in range(params["num_dcl"])
        ]

    def call(self, x, training=False, **kwargs):
        for dcl in self.dcb:
            x = dcl(x)
            x = tf.nn.relu(x)
        return x


class MultiTaskOutput(Model):
    def __init__(self, name, output_shapes):
        super().__init__(name=name)
        self.key = Dense(output_shapes["key"].dims[-1].value, name="key")
        self.ton = Dense(output_shapes["degree"].dims[-1].value, name="tonicisation")
        self.deg = Dense(output_shapes["degree"].dims[-1].value, name="degree")
        self.qlt = Dense(output_shapes["quality"].dims[-1].value, name="quality")
        self.inv = Dense(output_shapes["inversion"].dims[-1].value, name="inversion")
        self.roo = Dense(output_shapes["root"].dims[-1].value, name="root")

    def call(self, input_tensor, *args, **kwargs):
        o_qlt = self.qlt(input_tensor)
        o_inv = self.inv(input_tensor)
        o_roo = self.roo(input_tensor)
        o_key = self.key(input_tensor)
        # z = tf.concat([input_tensor, o_key], axis=-1)
        # o_ton = self.ton(z)
        # o_deg = self.deg(z)
        o_ton = self.ton(input_tensor)
        o_deg = self.deg(input_tensor)
        return [o_key, o_ton, o_deg, o_qlt, o_inv, o_roo]


class ProgressionMultiTaskOutput(Model):
    def __init__(self, name, output_shapes):
        super(ProgressionMultiTaskOutput, self).__init__(name=name)
        classes_key = output_shapes["key"].dims[-1].value
        classes_degree = output_shapes["degree"].dims[-1].value

        self.key = Dense(classes_key, name="key")
        self.ton = Dense(classes_degree, name="tonicisation")
        self.deg = Dense(classes_degree, name="degree")

    def call(self, input_tensor, *args, **kwargs):
        o_key = self.key(input_tensor)
        z = tf.concat([input_tensor, o_key], axis=-1)
        o_ton = self.ton(z)
        o_deg = self.deg(z)
        return [o_key, o_ton, o_deg]


class LocalMultiTaskOutput(Model):
    def __init__(self, name, output_shapes):
        super(LocalMultiTaskOutput, self).__init__(name=name)
        classes_root = output_shapes["root"].dims[-1].value
        classes_quality = output_shapes["quality"].dims[-1].value
        classes_inversion = output_shapes["inversion"].dims[-1].value

        self.qlt = Dense(classes_quality, name="quality")
        self.inv = Dense(classes_inversion, name="inversion")
        self.roo = Dense(classes_root, name="root")

    def call(self, input_tensor, *args, **kwargs):
        o_qlt = self.qlt(input_tensor)
        o_inv = self.inv(input_tensor)
        o_roo = self.roo(input_tensor)
        return [o_qlt, o_inv, o_roo]


def binary_cross_entropy(targets, outputs):
    negative = -(1 - targets) * tf.math.log(1 - outputs + eps)
    positive = -targets * tf.math.log(outputs + eps)
    return positive + negative


def softmax_sample_time_distributed(feature_output, n_timesteps, depth):
    feature_output = tf.reshape(feature_output, [-1, depth])
    temp = tf.random.categorical(feature_output, 1)
    temp = tf.one_hot(temp, depth, axis=-1)
    temp = tf.reshape(temp, [-1, n_timesteps, depth])
    return temp


def argmax_sample(feature_output, depth):
    argmax = tf.argmax(feature_output, axis=-1)
    return tf.one_hot(argmax, depth, axis=-1)
