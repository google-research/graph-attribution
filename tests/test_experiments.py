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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import experiments
import featurization
import graphnet_models as models
import graphnet_techniques as techniques
import graphs as graph_utils
import tasks
import templates


GNN_TESTCASES = [('gcn', 10, 5, 20, 2, models.BlockType('gcn'), 3),
                 ('mpnn', 20, 10, 40, 1, models.BlockType('mpnn'), 4),
                 ('gat', 10, 20, 30, 1, models.BlockType('mpnn'), 4),
                 ('graphnet', 6, 3, 15, 3, models.BlockType('graphnet'), 1)]


class GNNTests(parameterized.TestCase):
  """Test GNN inference/gradient computations."""

  def _setup_graphs(self):
    """Setup graphs and smiles if needed."""
    tensorizer = featurization.MolTensorizer()
    smiles = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    graphs = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    return graphs, graph_utils.get_num_graphs(graphs)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_get_graph_embedding(self, node_size, edge_size, globals_size, y_size,
                               model_type, n_layers):
    """Check that shapes for get_embedding are correct."""
    graphs, n_graphs = self._setup_graphs()
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    emb = model.get_graph_embedding(graphs).numpy()
    self.assertEqual((n_graphs, globals_size), emb.shape)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_call(self, node_size, edge_size, globals_size, y_size, model_type,
                n_layers):
    """Check that shapes for call are correct."""
    graphs, n_graphs = self._setup_graphs()
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    y_out = model(graphs).numpy()
    self.assertEqual((n_graphs, y_size), y_out.shape)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_predict(self, node_size, edge_size, globals_size, y_size, model_type,
                   n_layers):
    """Check that predict is 1-D."""
    graphs, n_graphs = self._setup_graphs()
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    y_out = model.predict(graphs).numpy()
    self.assertEqual((n_graphs,), y_out.shape)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_get_gradient(self, node_size, edge_size, globals_size, y_size,
                        model_type, n_layers):
    """Check that shapes for gradients are correct."""
    graphs, _ = self._setup_graphs()
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    node_grads, edge_grads = model.get_gradient(graphs)
    self.assertEqual(graphs.nodes.shape, node_grads.shape)
    self.assertEqual(graphs.edges.shape, edge_grads.shape)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_get_gap_activations(self, node_size, edge_size, globals_size, y_size,
                               model_type, n_layers):
    """Check that shapes for gap activation are like global vectors."""
    graphs, _ = self._setup_graphs()
    n_nodes, n_edges = graphs.nodes.shape[0], graphs.edges.shape[0]
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    node_act, edge_act = model.get_gap_activations(graphs)
    self.assertEqual((n_nodes, globals_size), node_act.shape)
    self.assertEqual((n_edges, globals_size), edge_act.shape)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_get_prediction_weights(self, node_size, edge_size, globals_size,
                                  y_size, model_type, n_layers):
    """Check that we have enough weights as globals vector dim."""
    graphs, _ = self._setup_graphs()
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    w = model.get_prediction_weights()
    self.assertEqual(w.shape, globals_size)

  @parameterized.named_parameters(*GNN_TESTCASES)
  def test_get_intermediate_activations_gradients(self, node_size, edge_size,
                                                  globals_size, y_size,
                                                  model_type, n_layers):
    """Check shapes for actvations/gradients, check that predictions match."""
    graphs, _ = self._setup_graphs()
    n_nodes, n_edges = graphs.nodes.shape[0], graphs.edges.shape[0]
    model = experiments.GNN(node_size, edge_size, globals_size, y_size,
                            model_type, 'relu', templates.TargetType.globals,
                            n_layers)
    model(graphs)
    acts, grads, y = model.get_intermediate_activations_gradients(graphs)
    y_expected = model.predict(graphs)

    self.assertTrue(np.allclose(y, y_expected))
    self.assertLen(acts, n_layers)
    self.assertLen(grads, n_layers)
    for (act_nodes, act_edges), (grad_nodes, grad_edges) in zip(acts, grads):
      self.assertEqual((n_nodes, node_size), act_nodes.shape)
      self.assertEqual((n_edges, edge_size), act_edges.shape)
      self.assertEqual((n_nodes, node_size), grad_nodes.shape)
      self.assertEqual((n_edges, edge_size), grad_edges.shape)


class ExperimentsDataTests(parameterized.TestCase):
  """Integration test for experiment execution."""

  def _setup_experiment(self):
    """Setup graphs and smiles if needed."""
    smiles = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    n = len(smiles)
    smiles_to_mol = functools.partial(
        featurization.smiles_to_mol, infer_hydrogens=True)
    tensorizer = featurization.MolTensorizer(preprocess_fn=smiles_to_mol)
    train_index, test_index = np.arange(n - 1), np.arange(n - 1, n)
    mol_list = [smiles_to_mol(smi) for smi in smiles]
    x = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    task = tasks.get_task(tasks.Task.crippen)
    y = task.get_true_predictions(mol_list)
    atts = task.get_true_attributions(mol_list)
    exp = experiments.ExperimentData.from_data_and_splits(
        x, y, atts, train_index, test_index)
    model = experiments.GNN(5, 3, 10, 1, models.BlockType.gcn, 'relu',
                            templates.TargetType.globals, 2)
    model(x)
    method = techniques.CAM()

    return exp, model, task, method

  def test_setup_experiment_data(self):
    """Check that our setup function works and ExperimentData is created."""
    exp, _, _, _ = self._setup_experiment()
    self.assertIsInstance(exp, experiments.ExperimentData)

  def test_generate_result_default(self):
    """Test generate result runs."""
    exp, model, task, method = self._setup_experiment()
    _ = experiments.generate_result(model, method, task, exp.x_test, exp.y_test,
                                    exp.att_test)


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
