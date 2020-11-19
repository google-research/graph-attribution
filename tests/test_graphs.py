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
"""Test code for functions that are related to graphtuples."""
from absl.testing import absltest
from absl.testing import parameterized
import graph_nets
import numpy as np
import tensorflow as tf

import featurization
import graphs as graph_utils


def _get_graph_field(graph, field):
  """Get a graph field and cast to numpy if a tensor."""
  value = getattr(graph, field)
  return value.numpy() if isinstance(value, tf.Tensor) else value


class GraphsTupleTests(parameterized.TestCase):
  """Tests for GraphTuple manipulations."""

  def _setup_graphs(self):
    """Setup graphs and smiles if needed."""
    tensorizer = featurization.MolTensorizer()
    smiles = [
        'CO', 'Cc1occc1C(=O)Nc2ccccc2',
        'CC(C)=CCCC(C)=CC(=O)', 'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43', 'c1ccsc1',
        'c2ccc1scnc1c2', 'Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl',
    ]

    graphs = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    return graphs, smiles, tensorizer

  def assertEqualGraphsTuple(self, graph_a, graph_b):
    """Check graphs are the same."""
    for field in graph_nets.graphs.ALL_FIELDS:
      a_value = _get_graph_field(graph_a, field)
      b_value = _get_graph_field(graph_b, field)
      if a_value is not None and b_value is not None:
        same = np.allclose(a_value, b_value)
      else:
        same = a_value is None and b_value is None
      if not same:
        raise ValueError(f'graph.{field} is mismatching ({a_value},{b_value})')

  def test_smiles_to_graphs_tuple(self):
    """Check that graphs have same number of nodes as atoms."""
    graphs, smiles_list, _ = self._setup_graphs()
    mol_list = [featurization.smiles_to_mol(smi) for smi in smiles_list]
    n_atoms = [mol.GetNumAtoms() for mol in mol_list]
    self.assertLen(mol_list, graph_utils.get_num_graphs(graphs))
    self.assertEqual(n_atoms, graphs.n_node.numpy().tolist())

  def test_binarize_np_nodes(self):
    """Check that the nodes are indeed binary."""
    graphs, _, _ = self._setup_graphs()
    graphs = graph_utils.cast_to_np(graphs)
    new_graphs = graph_utils.binarize_np_nodes(graphs, 0.5)
    np.testing.assert_allclose(np.unique(new_graphs.nodes), [0., 1.])

  def test_make_constant_like(self):
    """Check constant graph has same shape, and constant nodes/edges."""
    graphs, _, tensorizer = self._setup_graphs()
    node_vec, edge_vec = tensorizer.get_null_vectors()
    const_graphs = graph_utils.make_constant_like(graphs, node_vec, edge_vec)
    self.assertEqual(graphs.nodes.shape, const_graphs.nodes.shape)
    self.assertEqual(graphs.edges.shape, const_graphs.edges.shape)
    self.assertTrue(all(np.allclose(node_vec, x) for x in const_graphs.nodes))
    self.assertTrue(all(np.allclose(edge_vec, x) for x in const_graphs.edges))

  def test_split_graphs_tuple(self):
    """Check that we can split graphtuples into a list of graphs."""
    graphs, _, _ = self._setup_graphs()
    graph_list = list(graph_utils.split_graphs_tuple(graphs))
    self.assertLen(graph_list, graph_utils.get_num_graphs(graphs))
    for index, graph_index in enumerate(graph_list):
      expected_graph = graph_nets.utils_np.get_graph(graphs, index)
      self.assertEqualGraphsTuple(graph_index, expected_graph)

  @parameterized.named_parameters(('ordered', [1, 2, 3, 5, 6]),
                                  ('unsorted', [2, 5, 1, 0]))
  def test_get_graphs_tf(self, indices):
    """Check that we can split graphtuples into a list of graphs."""
    graphs, _, _ = self._setup_graphs()
    sub_graphs = graph_utils.get_graphs_tf(graphs, np.array(indices))
    graph_list = [graph_nets.utils_tf.get_graph(graphs, i) for i in indices]
    expected_graphs = graph_nets.utils_tf.concat(graph_list, axis=0)
    self.assertEqualGraphsTuple(sub_graphs, expected_graphs)

  def test_get_graphs_tf_noedges(self):
    """Check that we can split graphtuples with no edge information."""
    indices = [1, 2, 5, 6]
    graphs, _, _ = self._setup_graphs()
    graphs = graphs.replace(edges=None)
    sub_graphs = graph_utils.get_graphs_tf(graphs, np.array(indices))
    graph_list = [graph_nets.utils_tf.get_graph(graphs, i) for i in indices]
    expected_graphs = graph_nets.utils_tf.concat(graph_list, axis=0)
    self.assertEqualGraphsTuple(sub_graphs, expected_graphs)

  def test_get_graphs_np_noedges(self):
    """Check that we can split graphtuples with no edge information."""
    indices = [1, 2, 5, 6]
    graphs, _, _ = self._setup_graphs()
    graphs = graphs.replace(edges=None)
    graphs = graph_utils.cast_to_np(graphs)
    sub_graphs = graph_utils.get_graphs_np(graphs, np.array(indices))
    graph_list = [graph_nets.utils_tf.get_graph(graphs, i) for i in indices]
    expected_graphs = graph_nets.utils_tf.concat(graph_list, axis=0)
    self.assertEqualGraphsTuple(sub_graphs, expected_graphs)

  @parameterized.named_parameters(('ordered', [1, 2, 3, 5, 6]),
                                  ('unsorted', [2, 5, 1, 0]))
  def test_get_graphs_np(self, indices):
    """Check that we can split graphtuples into a list of graphs."""
    graphs, _, _ = self._setup_graphs()
    graphs = graph_utils.cast_to_np(graphs)
    sub_graphs = graph_utils.get_graphs_np(graphs, indices)
    graph_list = [graph_nets.utils_tf.get_graph(graphs, i) for i in indices]
    expected_graphs = graph_nets.utils_tf.concat(graph_list, axis=0)
    self.assertEqualGraphsTuple(sub_graphs, expected_graphs)

  def _setup_toy_segment_data(self):
    """Toy example for testing segment type of data."""
    data1 = [1., 2., 3.]
    # mu2= 2, std2 ~0.81649658, n2 = 3
    mu1, std1, n1 = np.mean(data1), np.std(data1), len(data1)
    data2 = [4., 5., 6., 7., 8., 9.]
    # mu2= 6.5, std2 ~1.70782513, n2 = 6
    mu2, std2, n2 = np.mean(data2), np.std(data2), len(data2)
    data = tf.reshape(tf.constant(data1 + data2), (-1, 1))
    counts = tf.constant([n1, n2])
    # [[2.],[2.],[2.],[6.5],[6.5],[6.5],[6.5],[6.5], [6.5]]
    mu = tf.reshape(tf.constant([mu1] * n1 + [mu2] * n2), (-1, 1))
    # [[0.81],[0.81,[0.81],[1.70],[1.70],[1.70],[1.70],[1.70], [1.70]]
    std = tf.reshape(tf.constant([std1] * n1 + [std2] * n2), (-1, 1))
    return data, counts, mu, std

  def test_segment_mean_stddev_contents(self):
    """Check that mu/std coincide with expected."""
    data, counts, expected_mu, expected_std = self._setup_toy_segment_data()
    mu, std = graph_utils.segment_mean_stddev(data, counts)
    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(std, expected_std)

  def test_segment_mean_stddev_shapes(self):
    """Check that results have expected shape."""
    graphs, _, _ = self._setup_graphs()
    for data, counts in [(graphs.nodes, graphs.n_node),
                         (graphs.edges, graphs.n_edge)]:
      mu, std = graph_utils.segment_mean_stddev(data, counts)
      self.assertEqual(mu.shape, data.shape)
      self.assertEqual(std.shape, data.shape)

  @parameterized.parameters((10, 0.0), (15, 0.1))
  def test_perturb_graphs_tuple(self, n_samples, sigma):
    graphs, _, _ = self._setup_graphs()
    noisy_graphs = graph_utils.perturb_graphs_tuple(graphs, n_samples, sigma)
    expected_n = n_samples * graph_utils.get_num_graphs(graphs)
    actual_n = graph_utils.get_num_graphs(noisy_graphs)
    self.assertEqual(expected_n, actual_n)
    self.assertEqual(graphs.nodes.shape[-1], noisy_graphs.nodes.shape[-1])
    self.assertEqual(graphs.edges.shape[-1], noisy_graphs.edges.shape[-1])

  @parameterized.parameters((1), (10))
  def test_perturb_graphs_tuple_zero(self, n_samples):
    """When sigma is zero, graphs should be the same."""
    graphs, _, _ = self._setup_graphs()
    sigma = 0.0
    n_graphs = graph_utils.get_num_graphs(graphs)
    noisy_graphs = graph_utils.perturb_graphs_tuple(graphs, n_samples, sigma)
    for index in range(n_samples):
      indices = np.arange(index * n_graphs, (index + 1) * n_graphs)
      sub_graphs = graph_utils.get_graphs_tf(noisy_graphs, indices)
      np.testing.assert_allclose(graphs.nodes, sub_graphs.nodes)
      np.testing.assert_allclose(graphs.edges, sub_graphs.edges)

  def test_perturb_graphs_tuple_noise(self):
    """Check perturb graph information should vary more than original."""
    graphs, _, _ = self._setup_graphs()
    n_samples = 10
    noisy_graphs = graph_utils.perturb_graphs_tuple(graphs, n_samples, 0.5)
    self.assertGreater(np.std(noisy_graphs.nodes), np.std(graphs.nodes))
    self.assertGreater(np.std(noisy_graphs.edges), np.std(graphs.edges))

  def test_reduce_sum_edges(self):
    """Check edges get reduced to nodes."""
    graphs, smiles, _ = self._setup_graphs()
    mol_list = [featurization.smiles_to_mol(smi) for smi in smiles]
    degree_list = []
    for mol in mol_list:
      for atom in mol.GetAtoms():
        degree_list.append(atom.GetDegree())
    edge_graph = graphs.replace(
        nodes=tf.zeros(graphs.nodes.shape[0]),
        edges=tf.ones(graphs.edges.shape[0]))
    degree_graph = graph_utils.reduce_sum_edges(edge_graph)
    np.testing.assert_allclose(degree_list, degree_graph.nodes)
    self.assertIsNone(degree_graph.edges)

  def test_reduce_sum_edges_noedges(self):
    """Check reduce_sum_edges works with no edges."""
    graphs, _, _ = self._setup_graphs()
    expected_graph = graphs.replace(
        nodes=tf.ones(graphs.nodes.shape[0]), edges=None)
    actual_graph = graph_utils.reduce_sum_edges(expected_graph)
    self.assertEqualGraphsTuple(expected_graph, actual_graph)

  @parameterized.parameters((3), (9))
  def test_interp_array(self, n_steps):
    """Check that our interpolation works on arrays."""
    graphs, _, tensorizer = self._setup_graphs()
    ref = graph_utils.make_constant_like(graphs, *tensorizer.get_null_vectors())
    start = ref.nodes
    end = graphs.nodes
    interp = graph_utils._interp_array(start, end, n_steps)  # pylint:disable=protected-access
    mean_arr = np.mean([start, end], axis=0)
    np.testing.assert_allclose(interp[0], start)
    np.testing.assert_allclose(interp[int((n_steps - 1) / 2)], mean_arr)
    np.testing.assert_allclose(interp[-1], end)
    self.assertEqual(interp.shape, (n_steps, start.shape[0], start.shape[1]))

  @parameterized.parameters((3), (9))
  def test_interpolate_graphs_tuple_endpoints(self, n_steps):
    """Check that our interpolation matches at endpoints."""
    several_ends, _, tensorizer = self._setup_graphs()
    for end in graph_utils.split_graphs_tuple(several_ends):
      start = graph_utils.make_constant_like(end,
                                             *tensorizer.get_null_vectors())
      interp, _, _ = graph_utils.interpolate_graphs_tuple(start, end, n_steps)
      start_interp = graph_utils.get_graphs_tf(interp, np.array([0]))
      end_interp = graph_utils.get_graphs_tf(interp, np.array([n_steps - 1]))
      self.assertEqualGraphsTuple(start, start_interp)
      self.assertEqualGraphsTuple(end, end_interp)

  def test_interpolate_graphs_tuple_differences(self):
    """Check that our interpolation has constant differences between steps."""
    n_steps = 8
    several_ends, _, tensorizer = self._setup_graphs()
    for end in graph_utils.split_graphs_tuple(several_ends):
      start = graph_utils.make_constant_like(end,
                                             *tensorizer.get_null_vectors())
      interp, _, _ = graph_utils.interpolate_graphs_tuple(start, end, n_steps)
      steps = list(graph_utils.split_graphs_tuple(interp))
      expected_nodes_diff = tf.divide(end.nodes - start.nodes, n_steps - 1)
      expected_edges_diff = tf.divide(end.edges - start.edges, n_steps - 1)
      for x_cur, x_next in zip(steps[:-1], steps[1:]):
        actual_nodes_diff = x_next.nodes - x_cur.nodes
        actual_edges_diff = x_next.edges - x_cur.edges
        np.testing.assert_allclose(
            expected_nodes_diff, actual_nodes_diff, atol=1e-7)
        np.testing.assert_allclose(
            expected_edges_diff, actual_edges_diff, atol=1e-7)

  @parameterized.parameters((3), (8))
  def test_interpolate_graphs_tuple_batch(self, n_steps):
    """Check that interpolated graphs are same, irrepective if batched."""
    end, _, tensorizer = self._setup_graphs()
    n_graphs = graph_utils.get_num_graphs(end)
    start = graph_utils.make_constant_like(end, *tensorizer.get_null_vectors())
    interp, _, _ = graph_utils.interpolate_graphs_tuple(start, end, n_steps)
    start_iter = graph_utils.split_graphs_tuple(start)
    end_iter = graph_utils.split_graphs_tuple(end)
    for i, (start_i, end_i) in enumerate(zip(start_iter, end_iter)):
      indices = np.arange(0, n_steps * n_graphs, n_graphs) + i
      actual = graph_utils.get_graphs_tf(interp, indices)
      expected, _, _ = graph_utils.interpolate_graphs_tuple(
          start_i, end_i, n_steps)
      self.assertEqualGraphsTuple(expected, actual)


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
