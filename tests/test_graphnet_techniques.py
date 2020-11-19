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
"""Tests for graphnet_techniques."""
from absl.testing import absltest
from absl.testing import parameterized
import graph_nets
import numpy as np
import tensorflow as tf

import experiments
import featurization
import graphnet_models as models
import graphnet_techniques as techniques
import graphs as graph_utils
import templates


class AttributionTechniquesTests(parameterized.TestCase):
  """Test attribution interface correctness."""

  def _setup_graphs_model(self):
    """Setup graphs and smiles if needed."""
    tensorizer = featurization.MolTensorizer()
    smiles = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    graphs = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    # Fix seed so that initialization is deterministic.
    tf.random.set_seed(0)
    model = experiments.GNN(5, 3, 10, 1, models.BlockType('gcn'), 'relu',
                            templates.TargetType.globals, 3)
    model(graphs)
    return graphs, model, tensorizer

  def _setup_technique(self, name, tensorizer):
    """Setup attribution techniques."""
    methods = techniques.get_techniques_dict(*tensorizer.get_null_vectors())
    return methods[name]

  def assertAttribution(self, graphs, atts):
    atts = graph_nets.utils_tf.concat(atts, axis=0)
    self.assertEqual(atts.nodes.ndim, 1)
    self.assertEqual(atts.edges.ndim, 1)
    self.assertEqual(graphs.nodes.shape[0], atts.nodes.shape[0])
    self.assertEqual(graphs.edges.shape[0], atts.edges.shape[0])
    np.testing.assert_allclose(graphs.n_node, atts.n_node)
    np.testing.assert_allclose(graphs.n_edge, atts.n_edge)
    np.testing.assert_allclose(graphs.senders, atts.senders)
    np.testing.assert_allclose(graphs.receivers, atts.receivers)

  @parameterized.parameters([
      'Random', 'CAM', 'GradCAM-last', 'GradCAM-all', 'GradInput',
      'SmoothGrad(GradInput)', 'IG'
  ])
  def test_attribute(self, method_name):
    """Check we can attribute."""
    graphs, model, tensorizer = self._setup_graphs_model()
    method = self._setup_technique(method_name, tensorizer)
    atts = method.attribute(graphs, model)
    self.assertAttribution(graphs, atts)

  @parameterized.parameters(
      ['CAM', 'GradCAM-last', 'GradCAM-all', 'GradInput', 'IG'])
  def test_attribute_independence(self, method_name):
    """Check that atts are the same batched and non-batched."""
    graphs, model, tensorizer = self._setup_graphs_model()
    method = self._setup_technique(method_name, tensorizer)
    atts = method.attribute(graphs, model)
    single_graphs = graph_utils.split_graphs_tuple(graphs)
    for xi, actual in zip(single_graphs, atts):
      expected = method.attribute(xi, model)
      np.testing.assert_allclose(actual.nodes, expected[0].nodes, rtol=1e-2)
      np.testing.assert_allclose(actual.edges, expected[0].edges, rtol=1e-2)
      self.assertAttribution(xi, expected)

  def test_ig_sanity_check(self):
    """Check that IG improves with more integration steps."""
    graphs, model, tensorizer = self._setup_graphs_model()
    ref_fn = techniques.make_reference_fn(*tensorizer.get_null_vectors())
    method_25 = techniques.IntegratedGradients(25, ref_fn)
    method_100 = techniques.IntegratedGradients(100, ref_fn)
    error_25 = method_25.sanity_check(graphs, model)['ig_error'].mean()
    error_100 = method_100.sanity_check(graphs, model)['ig_error'].mean()
    self.assertLessEqual(error_100, error_25)


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
