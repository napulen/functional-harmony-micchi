import json
import logging
import os
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional, Dense, GRU
from tensorflow.keras.models import Model

from frog import INPUT_FEATURES
from frog.label_codec import LabelCodec
from frog.models.model_tools import (
    ConvBlock,
    MultiTaskOutput,
    argmax_sample,
    softmax_sample_time_distributed,
)
from frog.models.nade_tools import BlockNADE

logger = logging.getLogger(__name__)

hyper_params = {
    "conv": {
        "num_dnb_1": 3,  # N dense layers in the first block
        "filters_dnb_1": 10,  # each layer adds N feature maps to the input
        "bottleneck_filters_dnb_1": 32,  # in the paper, this is fixed to 4 * filters_dnb_1
        "kernel_size_dnb_1": 7,  # this determines how far you look in the future or past
        "filters_pb_1": 48,  # after the first pooling block, we have 32 total feature maps
        "num_dnb_2": 2,
        "filters_dnb_2": 4,
        "bottleneck_filters_dnb_2": 20,  # in the paper, this is fixed to 4 * filters_dnb_2
        "kernel_size_dnb_2": 3,
        "filters_pb_2": 48,
        "num_dnb_3": 2,
        "filters_dnb_3": 4,
        "bottleneck_filters_dnb_3": 20,  # in the paper, this is fixed to 4 * filters_dnb_2
        "kernel_size_dnb_3": 3,
    },
    "gru": {
        "dropout": 0.2,
        "size": 178,
    },
    "fc": {
        "size": 64,
    },
    "nade": {
        "hidden_size": 350,
        "ensemble_size": 10,
    },
    "optimizer": {
        "learning_rate": 0.003,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "decay": 0.0,
    },
}


hyper_algomus = {  # These are the parameters as used in the Algomus paper
    "conv": {
        "num_dnb_1": 4,  # N dense layers in the first block
        "filters_dnb_1": 8,  # each layer adds N feature maps to the input
        "bottleneck_filters_dnb_1": 32,  # in the paper, this is fixed to 4 * filters_dnb_1
        "kernel_size_dnb_1": 8,  # this determines how far you look in the future or past
        "filters_pb_1": 32,  # after the first pooling block, we have 32 total feature maps
        "num_dnb_2": 4,
        "filters_dnb_2": 5,
        "bottleneck_filters_dnb_2": 20,  # in the paper, this is fixed to 4 * filters_dnb_2
        "kernel_size_dnb_2": 8,
        "filters_pb_2": 48,
        "num_dnb_3": 0,
        "filters_dnb_3": 4,
        "bottleneck_filters_dnb_3": 20,  # in the paper, this is fixed to 4 * filters_dnb_2
        "kernel_size_dnb_3": 3,
    },
    "gru": {
        "dropout": 0.3,
        "size": 64,
    },
    "fc": {
        "size": 64,
    },
    "optimizer": {
        "learning_rate": 0.003,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "decay": 0.0,
    },
}


class CraModel(Model, ABC):
    def __init__(self, name, params):
        super().__init__(name=name)

        self.input_shapes = params["input_shapes"]
        self.output_shapes = params["output_shapes"]
        self.output_features = params["output_features"]
        # whatever feature would work to calculate the timesteps
        self.n_timesteps = params["output_shapes"][params["output_features"][0]][0]
        self.structure = params["structure"]

        self.conv = ConvBlock("ConvBlock", params["conv"])
        gp = params["gru"]
        self.gru = Bidirectional(GRU(gp["size"], return_sequences=True, dropout=gp["dropout"]))
        self.fc = Dense(params["fc"]["size"], activation="tanh")

        # As of TF 2.4.1, there is a bug for which Adam is not correctly loaded when serialised
        # The following lines are a workaround
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(params["optimizer"]["learning_rate"]),
            beta_1=tf.Variable(params["optimizer"]["beta_1"]),
            beta_2=tf.Variable(params["optimizer"]["beta_2"]),
            epsilon=tf.Variable(params["optimizer"]["epsilon"]),
        )
        # this access will invoke optimizer._iterations method and create optimizer.iter attribute
        self.optimizer.iterations
        self.optimizer.decay = tf.Variable(params["optimizer"]["decay"])

    @abstractmethod
    def call(self, inputs, *args, **kwargs):
        pass

    @abstractmethod
    def get_loss(self, targets, outputs):
        pass

    @abstractmethod
    def sample(self, input_tensor, *args, **kwargs):
        pass

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            outputs = self.call(inputs, training=True)
            loss_value = self.get_loss(targets, outputs)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)

    @tf.function
    def network_learn(self, x, y):
        loss_value, grads = self.grad(x + y, y)  # grad already defines training=True
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value

    @tf.function
    def network_valid(self, x, y):
        y_ = self.call(x + y, training=True)  # We evaluate using teacher forcing here
        loss_value = self.get_loss(y, y_)
        return loss_value

    @tf.function
    def call_first_part(self, input_tensor, training):
        x = (
            tf.concat([input_tensor[0], input_tensor[1]], axis=-1)
            if self.structure
            else input_tensor[0]
        )
        x = self.conv(x, training=training)
        x = tf.nn.relu(x)  # Should this be here or not?
        x = self.gru(x, mask=input_tensor[2], training=training)
        x = self.fc(x)
        return x


class ConvGru(CraModel):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.mto = MultiTaskOutput("MultiTaskOutput", params["output_shapes"])
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def call(self, input_tensor, training=False, **kwargs):
        x = self.call_first_part(input_tensor, training)
        return self.mto(x)

    @tf.function
    def get_loss(self, targets, outputs):
        # FIXME: Think of maybe putting different weights (the key should be more important?)
        losses = [self.loss_fn(y_true=t, y_pred=p) for t, p in zip(targets, outputs)]
        return tf.reduce_sum(losses)

    @tf.function
    def sample(self, input_tensor, training=False, mode="argmax"):
        logits = self.call(input_tensor, training=training)
        # FIXME: This sampling is a bit ugly, although it should work correctly
        samples = []
        for feature_name, feature_output in zip(self.output_features, logits):
            n_timesteps, depth = self.output_shapes[feature_name]
            if mode == "sampled":
                temp = softmax_sample_time_distributed(feature_output, n_timesteps, depth)
            elif mode == "argmax":
                temp = argmax_sample(feature_output, depth)
            elif mode == "logits":  # do nothing
                temp = feature_output
            else:
                raise ValueError(f"mode can be logits, sampled, or argmax; you passed {mode}")
            samples.append(temp)
        return samples


class ConvGruBlockNade(CraModel):
    def __init__(self, name, params):
        super().__init__(name, params)

        n_visible = [params["output_shapes"][f][1] for f in self.output_features]
        self.bnade = BlockNADE(
            "BlockNADE", n_visible, params["nade"]["hidden_size"], self.n_timesteps
        )
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def call(self, input_tensor, training=False, targets=None, **kwargs):
        x = self.call_first_part(input_tensor, training)
        y = input_tensor[3:]
        x = self.bnade(x, training=training, targets=y)
        return x

    @tf.function
    def get_loss(self, targets, outputs):
        losses = [self.loss_fn(y_true=t, y_pred=p) for t, p in zip(targets, outputs)]
        # FIXME: Think of maybe putting different weights (the key should be more important?)
        return tf.reduce_mean(tf.reduce_sum(losses, axis=-1))

    @tf.function
    def sample(self, input_tensor, training=False, **kwargs):
        return self.call(input_tensor, training=training, **kwargs)


def create_model(
    model_type, input_shapes, output_shapes, output_features, hyper_params, structure=True
):
    params = {
        "output_shapes": output_shapes,
        "input_shapes": input_shapes,
        "input_features": INPUT_FEATURES,
        "output_features": output_features,
        "structure": structure,
    }
    params = {**params, **hyper_params}
    inputs_1 = [Input(shape=input_shapes[f], name=f) for f in params["input_features"]]
    inputs_2 = [Input(shape=output_shapes[f], name=f) for f in params["output_features"]]
    inputs = inputs_1 + inputs_2

    if model_type == "ConvGru" or model_type == "Algomus":
        model = ConvGru("ConvGru", params)
    elif model_type == "ConvGruBlocknade":
        model = ConvGruBlockNade("ConvGruBlockNade", params)
    else:
        raise ValueError("Model type not recognised")
    model.build(input_shape=[i.shape for i in inputs])
    return model


def load_model_with_info(model_path, verbose=False):
    # Load model
    with open(os.path.join(model_path, "info.json"), "r") as f:
        model_info = json.load(f)
    input_shapes = {k: tf.TensorShape(v) for k, v in model_info["input shapes"].items()}
    output_shapes = {k: tf.TensorShape(v) for k, v in model_info["output shapes"].items()}
    spelling, octave = model_info["input type"].split("_")
    lc = LabelCodec(spelling=spelling == "spelling", mode=model_info["output mode"])
    model = create_model(
        model_info["model type"],
        input_shapes,
        output_shapes,
        lc.output_features,
        model_info["hyper parameters"],
        model_info["use music structure info"],
    )
    model.load_weights(os.path.join(model_path, "model"))
    if verbose:
        model.summary()
    return model, model_info
