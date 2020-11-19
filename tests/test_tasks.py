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
"""Tests for tasks."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import rdkit.Chem.Crippen
import tensorflow.compat.v2 as tf

import featurization
import graphs as graph_utils
import tasks as att_tasks
import templates


TASK_TEST_CASES = [(t.name, t) for t in att_tasks.Task]
TASKS_SKIP_DATA_LOAD = [
    att_tasks.Task.bashapes, att_tasks.Task.treegrid, att_tasks.Task.bacommunity
]


class TasksTests(parameterized.TestCase):
  """Test attribution metric evaluation."""

  def _setup_graphs_mols(self):
    """Setup graphs and smiles if needed."""
    smiles_to_mol = functools.partial(
        featurization.smiles_to_mol, infer_hydrogens=True)
    tensorizer = featurization.MolTensorizer(
        preprocess_fn=smiles_to_mol)
    smiles = ['CO', 'CCC', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    mols = [smiles_to_mol(smi) for smi in smiles]
    graphs = graph_utils.smiles_to_graphs_tuple(smiles, tensorizer)
    return graphs, mols

  def _setup_task_data(self, task_enum, mols):
    """Setup target labels and attributions."""
    task = att_tasks.get_task(task_enum)
    if task_enum in TASKS_SKIP_DATA_LOAD:
      proxy_task = att_tasks.get_task(att_tasks.Task.benzene)
      y_true = np.zeros((len(mols), task.n_outputs))
      y_true[:, 0] = 1.0
      att_true = proxy_task.get_true_attributions(mols)
    else:
      y_true = task.get_true_predictions(mols)
      att_true = task.get_true_attributions(mols)
    return task, y_true, att_true

  def assertAttributionShape(self, graphs, att):
    self.assertEqual(
        graph_utils.get_num_graphs(graphs), graph_utils.get_num_graphs(att))
    self.assertEqual(graphs.nodes.shape[0], att.nodes.shape[0])
    if att.edges is not None:
      self.assertEqual(graphs.edges.shape[0], att.edges.shape[0])
    np.testing.assert_allclose(graphs.n_node, att.n_node)
    np.testing.assert_allclose(graphs.n_edge, att.n_edge)
    np.testing.assert_allclose(graphs.senders, att.senders)
    np.testing.assert_allclose(graphs.receivers, att.receivers)

  def test_get_crippen_features(self):
    """Check if crippen features match crippen values."""
    _, mols = self._setup_graphs_mols()
    expected_logp = np.array([rdkit.Chem.Crippen.MolLogP(mol) for mol in mols])
    our_logp = np.array(
        [sum(att_tasks.get_crippen_features(mol)[0]) for mol in mols])
    self.assertTrue(np.allclose(expected_logp, our_logp))

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_instantiation_defaults(self, task_enum):
    """Check that tasks can be initialized."""
    task = att_tasks.get_task(task_enum)
    self.assertIsInstance(task, templates.AttributionTask)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_get_nn_activation_fn(self, task_enum):
    """Check that our activation function returns tensors."""
    task = att_tasks.get_task(task_enum)
    act = task.get_nn_activation_fn()
    out = act(tf.constant([0, 0.1, 0.2, 0.3]))
    self.assertIsInstance(out, tf.Tensor)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_get_true_predictions(self, task_enum):
    """Check that we can retrieve true predictions."""
    _, mols = self._setup_graphs_mols()
    task, y_true, _ = self._setup_task_data(task_enum, mols)
    self.assertLen(mols, y_true.shape[0])
    self.assertEqual(y_true.shape[-1], task.n_outputs)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_get_nn_loss_fn(self, task_enum):
    """Check that our loss on the true prediction is 0.0."""
    _, mols = self._setup_graphs_mols()
    task, y_true, _ = self._setup_task_data(task_enum, mols)
    loss_fn = task.get_nn_loss_fn()
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    loss = loss_fn(y_true, y_true)
    np.testing.assert_allclose(loss, 0.0, atol=1e-6)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_evaluate_predictions(self, task_enum):
    """Check that we can evaluate predictions."""
    _, mols = self._setup_graphs_mols()
    task, y_true, _ = self._setup_task_data(task_enum, mols)
    results = task.evaluate_predictions(y_true, y_true)
    for name, value in results.items():
      self.assertIsInstance(name, str)
      self.assertIsInstance(value, float)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_get_true_attributions(self, task_enum):
    """Check we can retrieve attributions and have consistent shape."""
    graphs, mols = self._setup_graphs_mols()
    _, _, att_true = self._setup_task_data(task_enum, mols)
    for graph, att in zip(graph_utils.split_graphs_tuple(graphs), att_true):
      self.assertAttributionShape(graph, att)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_evaluate_attributions_mean(self, task_enum):
    """Checks that scoring of attributions with mean."""
    _, mols = self._setup_graphs_mols()
    task, _, att_true = self._setup_task_data(task_enum, mols)
    reducer_fn = np.mean
    results = task.evaluate_attributions(att_true, att_true, reducer_fn)
    for name, value in results.items():
      self.assertIsInstance(name, str)
      self.assertIsInstance(value, float)

  @parameterized.named_parameters(*TASK_TEST_CASES)
  def test_evaluate_attributions_identity(self, task_enum):
    """Checks that scoring of attributions with identity function."""
    _, mols = self._setup_graphs_mols()
    task, _, att_true = self._setup_task_data(task_enum, mols)
    reducer_fn = lambda x: x
    results = task.evaluate_attributions(att_true, att_true, reducer_fn)
    for name, values in results.items():
      self.assertIsInstance(name, str)
      self.assertIsInstance(values, (list, np.ndarray))


if __name__ == '__main__':
  tf.config.experimental_run_functions_eagerly(True)
  absltest.main()
