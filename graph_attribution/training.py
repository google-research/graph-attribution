# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# pylint: disable=logging-format-interpolation
"""Functions related to a training a GNN using graph_nets."""
from typing import Callable, Tuple

import graph_nets
import numpy as np
import sonnet as snt
import tensorflow as tf

from graph_attribution import graphnet_models as gnn_models
from graph_attribution import graphs as graph_utils
from graph_attribution import templates

# Typing alias.
GraphsTuple = graph_nets.graphs.GraphsTuple


def get_batch_indices(n: int, batch_size: int) -> np.ndarray:
    """Gets shuffled constant size batch indices to train a model."""
    n_batches = n // batch_size
    indices = tf.random.shuffle(tf.range(n))
    indices = indices[:n_batches * batch_size]
    indices = tf.reshape(indices, (n_batches, batch_size))
    return indices


def _get_indices_to_randomize(y: np.ndarray, noise_ratio: float) -> np.ndarray:
    """Selects random indices to randomize data; indices may be returned twice."""
    y = np.squeeze(y)
    num_labels = y.shape[0]
    num_shuffled_labels = int(num_labels * noise_ratio)
    rand_idxs = np.random.permutation(num_labels)[:num_shuffled_labels]
    return rand_idxs


def make_noisy_labels(y: np.ndarray, noise_ratio: np.float) -> np.ndarray:
    """Make noisy target labels by randomly shuffling labels."""
    rand_idxs = _get_indices_to_randomize(y, noise_ratio)
    elts_to_shuffle = y[rand_idxs]
    np.random.shuffle(elts_to_shuffle)
    new_y = np.copy(y)
    new_y[rand_idxs] = elts_to_shuffle
    return new_y


def augment_binary_task(
        x: GraphsTuple,
        y: np.ndarray,
        node_vec: np.ndarray,
        edge_vec: np.ndarray,
        fraction: float = 1.0) -> Tuple[GraphsTuple, np.ndarray]:
    """Augment input graphs and labels with null graphs."""
    if fraction > 1:
        raise ValueError(f'fraction to augment > 1({fraction}).')
    positive_indices = np.argwhere(y.ravel()).ravel()
    n = len(positive_indices)
    n_aug = int(np.floor(fraction * n))
    rand_indices = np.random.permutation(n)[:n_aug]
    indices = positive_indices[rand_indices]

    x_base = graph_utils.get_graphs_tf(x, indices)
    pos_graphs = graph_utils.make_constant_like(x_base, node_vec, edge_vec)
    pos_labels = np.ones((n_aug, 1))
    neg_graphs = graph_utils.make_constant_like(x_base, node_vec, edge_vec)
    neg_labels = np.zeros((n_aug, 1))
    x_aug = graph_nets.utils_tf.concat([x, pos_graphs, neg_graphs], axis=0)
    y_aug = np.concatenate((y, pos_labels, neg_labels))
    return x_aug, y_aug


def make_tf_opt_epoch_fn(
        inputs: GraphsTuple, target: np.ndarray, batch_size: int, model: snt.Module,
        optimizer: snt.Optimizer, loss_fn: templates.LossFunction,
        l2_reg: float = 0.0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Make a tf.function of (inputs, target) for optimization.

    This function is useful for basic inference training of GNN models. Uses all
    variables to create a a function that has a tf.function optimized input
    signature. Function uses pure tf.functions to build batches and aggregate
    losses. The result is a heavily optimized function that is at least 2x
    faster than a basic tf.function with experimental_relax_shapes=True.

    Args:
      inputs: graphs used for training.
      target: values to predict for training.
      batch_size: batch size.
      model: a GNN model.
      optimizer: optimizer, probably Adam or SGD.
      loss_fn: a loss function to optimize.
      l2_reg: l2 regularization weight.

    Returns:
      optimize_one_epoch(intpus, target), a tf.function optimized
      callable.

    """
    # Explicit input signature is faster than experimental relax shapes.
    input_signature = [
        graph_nets.utils_tf.specs_from_graphs_tuple(inputs),
        tf.TensorSpec.from_tensor(tf.convert_to_tensor(target))
    ]
    n = graph_utils.get_num_graphs(inputs)
    n_batches = tf.cast(n // batch_size, tf.float32)

    if l2_reg > 0.0:
        regularizer = snt.regularizers.L2(l2_reg)
        linear_variables = gnn_models.get_linear_variables(model)

    if batch_size == 1 or n == 1:
        def optimize_one_epoch(inputs, target):
            """One epoch single-batch optimization."""
            with tf.GradientTape() as tape:
                loss = loss_fn(target, model(inputs))
                if l2_reg > 0.0:
                    loss += regularizer(linear_variables)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply(grads, model.trainable_variables)
            return loss
    else:
        def optimize_one_epoch(inputs, target):
            """One epoch optimization."""
            loss = tf.constant(0.0, tf.float32)
            for batch in get_batch_indices(n, batch_size):
                x_batch = graph_utils.get_graphs_tf(inputs, batch)
                y_batch = tf.gather(target, batch)
                with tf.GradientTape() as tape:
                    batch_loss = loss_fn(y_batch, model(x_batch))
                    if l2_reg > 0.0:
                        batch_loss += regularizer(linear_variables)

                grads = tape.gradient(batch_loss, model.trainable_variables)
                optimizer.apply(grads, model.trainable_variables)
                loss += batch_loss
            return loss / n_batches

    return tf.function(optimize_one_epoch, input_signature=input_signature)
