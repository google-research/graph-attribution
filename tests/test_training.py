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
"""Tests for training GNN models."""
import numpy as np
import sonnet.v2 as snt
import tensorflow.compat.v2 as tf
from absl.testing import absltest, parameterized
from graph_attribution import experiments, featurization
from graph_attribution import graphnet_models as gnn_models
from graph_attribution import graphs as graph_utils
from graph_attribution import templates, training


class TrainingTests(parameterized.TestCase):
  """Basic tests for training a model."""

  def _setup_graphs_labels(self, n_graphs):
    """Setup graphs and labels for a binary classification learning task."""
    tensorizer = featurization.MolTensorizer()
    smiles_pool = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CCCO']
    smiles = np.random.choice(smiles_pool, n_graphs)
    graphs = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    n_labels = len(graphs.nodes) if n_graphs == 1 else n_graphs
    labels = np.random.choice([0, 1], n_labels).reshape(-1, 1)
    return graphs, labels

  def _setup_model(self, n_graphs):
    target_type = templates.TargetType.globals if n_graphs > 1 else templates.TargetType.nodes
    model = experiments.GNN(10, 10, 10, 1, gnn_models.BlockType('gcn'), 'relu',
                            target_type, 3)
    return model

  @parameterized.named_parameters(('constant', 1024, 256, 4),
                                  ('droplast', 1000, 256, 3))
  def test_get_batch_indices(self, n, batch_size, expected_n_batches):
    batch_indices = training.get_batch_indices(n, batch_size)
    self.assertEqual(batch_indices.shape, (expected_n_batches, batch_size))

  @parameterized.parameters([0.2, 1.0])
  def test_augment_binary_task(self, fraction):
    """Check that data augmention sizes are correct."""
    initial_n = 10
    x, y = self._setup_graphs_labels(initial_n)
    node_vec = np.zeros_like(x.nodes[0])
    edge_vec = np.zeros_like(x.edges[0])
    initial_positive = int(np.sum(y == 1))
    aug_n = int(np.floor(fraction * initial_positive))
    expected_n = initial_n + aug_n * 2
    x_aug, y_aug = training.augment_binary_task(x, y, node_vec, edge_vec,
                                                fraction)
    self.assertEqual(graph_utils.get_num_graphs(x_aug), expected_n)
    self.assertLen(y_aug, expected_n)
    # Make sure half of the augmented examples are positive labels.
    aug_positive = np.sum(y_aug == 1) - initial_positive
    self.assertEqual(aug_positive, aug_n)

  @parameterized.named_parameters(('onegraph', 1),
                                  ('minibatch', 25))
  def test_make_tf_opt_epoch_fn(self, batch_size):
    """Make sure tf-optimized epoch gives a valid loss."""
    x, y = self._setup_graphs_labels(batch_size)
    model = self._setup_model(batch_size)
    opt = snt.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    opt_fn = training.make_tf_opt_epoch_fn(x, y, batch_size, model, opt,
                                           loss_fn)
    loss = opt_fn(x, y).numpy()
    self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
