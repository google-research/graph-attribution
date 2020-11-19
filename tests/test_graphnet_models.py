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
"""Tests for graphnet models."""
import io

from absl.testing import absltest
from absl.testing import parameterized
import graph_nets
import mock
import numpy as np
import sonnet.v2 as snt
import tensorflow as tf

import featurization
import graphnet_models as models
import graphs as graph_utils


class GraphnetModelsTests(parameterized.TestCase):
  """Test GNN implementations."""

  def _setup_graphs(self):
    """Setup graphs and smiles if needed."""
    tensorizer = featurization.MolTensorizer()
    smiles = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    return graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)

  @parameterized.parameters(('relu', tf.nn.relu), ('identity', tf.identity),
                            ('sigmoid', tf.nn.sigmoid),
                            ('softmax', tf.nn.softmax))
  def test_cast_activation(self, act_text, act):
    """Check that we can get a mlp function and output is correct."""
    vecs = self._setup_graphs().nodes
    out_vecs1 = act(vecs)
    out_vecs2 = models.cast_activation(act_text)(vecs)
    out_vecs3 = models.cast_activation(act)(vecs)
    self.assertTrue(np.allclose(out_vecs1, out_vecs2))
    self.assertTrue(np.allclose(out_vecs2, out_vecs3))

  def assertGraphShape(self,
                       graphs,
                       num_graphs=None,
                       node_dim=None,
                       edge_dim=None,
                       globals_dim=None):
    """Check that a graph has the correct shape for several fields."""
    if num_graphs is not None:
      self.assertEqual(graph_utils.get_num_graphs(graphs), num_graphs)
    if node_dim is not None:
      self.assertEqual(graphs.nodes.shape[-1], node_dim)
    if edge_dim is not None:
      self.assertEqual(graphs.edges.shape[-1], edge_dim)
    if globals_dim is not None:
      self.assertEqual(graphs.globals.shape[-1], globals_dim)

  @parameterized.named_parameters(('10_default', 10, None),
                                  ('20_relu', 20, tf.nn.relu),
                                  ('10_relu_text', 10, 'relu'))
  def test_get_mlp_fn(self, out_size, act):
    """Check that we can get a mlp function and output is correct."""
    vecs = self._setup_graphs().nodes
    if act is not None:
      mlp_fn = models.get_mlp_fn([out_size], act)
    else:
      mlp_fn = models.get_mlp_fn([out_size])
    model = mlp_fn()
    new_vecs = model(vecs)
    actual_out_size = new_vecs.numpy().shape[-1]
    self.assertEqual(actual_out_size, out_size)

  @parameterized.named_parameters(('10_relu', 10, tf.nn.relu),
                                  ('20_relu', 20, tf.nn.relu),
                                  ('10_relu_text', 10, 'relu'))
  def test_ReadoutGAP(self, globals_size, act):
    """Check that output global shape, nodes and edges are the same."""
    graphs = self._setup_graphs()
    module = models.ReadoutGAP(globals_size, act)
    out_graphs = module(graphs)
    self.assertGraphShape(
        out_graphs,
        num_graphs=graph_utils.get_num_graphs(graphs),
        globals_dim=globals_size,
        node_dim=graphs.nodes.shape[-1],
        edge_dim=graphs.edges.shape[-1])

  @parameterized.parameters((models.NodeLayer, 10), (models.NodeLayer, 20),
                            (models.GCNLayer, 10), (models.GCNLayer, 20))
  def test_node_layer(self, node_layer, node_size):
    """Check that output only changes node shape."""
    graphs = self._setup_graphs()
    node_fn = models.get_mlp_fn([node_size])
    module = node_layer(node_fn)
    out_graphs = module(graphs)
    self.assertGraphShape(
        out_graphs,
        num_graphs=graph_utils.get_num_graphs(graphs),
        node_dim=node_size,
        edge_dim=graphs.edges.shape[-1])

  def test_NodeAggregatorLayer(self):
    """Check that nodes get aggregated."""
    # Graph has 3 nodes, node #0 and #2 are connected to node #1.
    x_data_dict = [{
        'nodes': np.eye(3),
        'edges': None,
        'receivers': np.array([1, 0, 2, 1]),
        'senders': np.array([0, 1, 1, 2])
    }]
    expected_nodes = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    x = graph_nets.utils_tf.data_dicts_to_graphs_tuple(x_data_dict)
    module = models.NodesAggregator()
    out_nodes = module(x)
    np.testing.assert_allclose(out_nodes, expected_nodes)

  @parameterized.parameters((10, 5), (20, 10))
  def test_NodeEdgeLayer(self, node_size, edge_dim):
    """Check that output only changes node and edge shape."""
    graphs = self._setup_graphs()
    node_fn = models.get_mlp_fn([node_size])
    edge_fn = models.get_mlp_fn([edge_dim])
    module = models.NodeEdgeLayer(node_fn, edge_fn)
    out_graphs = module(graphs)
    self.assertGraphShape(
        out_graphs,
        num_graphs=graph_utils.get_num_graphs(graphs),
        node_dim=node_size,
        edge_dim=edge_dim)

  @parameterized.parameters((1), (5))
  def test_SequentialWithActivations(self, n_layers):
    """Check that activations are correct size, last act should coincide."""
    vecs = self._setup_graphs().nodes
    size_list = [10 * (i + 1) for i in range(n_layers)]
    layers = [snt.Linear(size) for size in size_list]
    module = models.SequentialWithActivations(layers)
    module(vecs)
    new_vecs, acts = module.call_with_activations(vecs)
    act_sizes = [act.numpy().shape[-1] for act in acts]
    self.assertLen(act_sizes, n_layers)
    self.assertEqual(size_list, act_sizes)
    self.assertTrue(np.allclose(new_vecs.numpy(), acts[-1].numpy()))

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  def test_print_model(self, mock_stdout):
    """Check that print function runs."""
    module = snt.nets.MLP([5, 6, 7], with_bias=False)
    module(tf.ones(shape=[1, 10]))
    num_trainable_weights = 10 * 5 + 5 * 6 + 6 * 7
    models.print_model(module)
    self.assertIn(str(num_trainable_weights), mock_stdout.getvalue())


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
