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

# lint as python3
"""Tests for attribution_metrics."""

import warnings

import graph_nets
import numpy as np
import sklearn
import sklearn.metrics
from absl.testing import absltest, parameterized
from graph_attribution import attribution_metrics as att_metrics


class AttributionsMetricsTestCase(parameterized.TestCase):
  """Basic tests for attribution metrics."""

  def _setup_graph_from_nodes(self, nodes):
    return graph_nets.utils_np.data_dicts_to_graphs_tuple([{'nodes': nodes}])

  def test_silent_nan_np_error(self):
    """Check if we get a nan and basic metric raises an error."""
    test_fn = sklearn.metrics.roc_auc_score
    decorated_test_fn = att_metrics.silent_nan_np(test_fn)
    bad_input = ([0, 0], [0, 0])

    # Show that native sklearn version raises an error.
    with self.assertRaises(ValueError):
      test_fn(*bad_input)

    # Show that decorated function returns nan.
    self.assertTrue(np.isnan(decorated_test_fn(*bad_input)))

  @parameterized.parameters([np.nanmax, np.nanmean])
  def test_silent_nan_np_runtime_warning(self, test_fn):
    """Check if metric is truly silent."""
    decorated_test_fn = att_metrics.silent_nan_np(test_fn)
    bad_input = [np.nan]

    # Show that native sklearn version raises an warning.
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      test_fn(bad_input)
      self.assertLen(w, 1)

    # Show that decorated function has no warnings.
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      decorated_test_fn(bad_input)
      self.assertEmpty(w)

  @parameterized.named_parameters([('pearson_r', att_metrics.pearson_r_score),
                                   ('kendall_tau',
                                    att_metrics.kendall_tau_score),
                                   ('f1_score', att_metrics.nan_f1_score)])
  def test_nodewise_metric(self, metric_fn):
    """Check that metric is applied node-wise in a graph."""
    y_true = [0, 1]
    expected_result = 1.0  # Assumes all metrics are scoring functions.
    att_true = [self._setup_graph_from_nodes(y_true)]
    node_metric_fn = att_metrics.nodewise_metric(metric_fn)
    actual_result = node_metric_fn(att_true, att_true)
    self.assertEqual(expected_result, actual_result)

  @parameterized.named_parameters([('att_auroc', att_metrics.nan_auroc_score),
                                   ('att_acc', att_metrics.accuracy_score),
                                   ('att_f1', att_metrics.nan_f1_score)])
  def test_attribution_metric(self, metric_fn):
    """Check that metric is applied node-wise in a graph."""
    y_true = [[0, 1], [1, 0]]  # Two possible ground truths.
    y_pred = [0, 1]
    expected_result = 1.0  # Assumes all metrics are scoring functions.
    att_true = [self._setup_graph_from_nodes(y_true)]
    att_pred = [self._setup_graph_from_nodes(y_pred)]
    node_metric_fn = att_metrics.attribution_metric(metric_fn)
    actual_result = node_metric_fn(att_true, att_pred)
    self.assertEqual(expected_result, actual_result)

  @parameterized.parameters([(0.2, 0.4), (0.3, 0.8)])
  def test_optimal_threshold(self, neg_prob, pos_prob):
    """Check that returned probability is between both class probabilities."""
    y_true = [0, 1]
    y_prob = [neg_prob, pos_prob]
    p_threshold = att_metrics.get_optimal_threshold(y_true, y_prob)
    self.assertGreater(p_threshold, neg_prob)
    self.assertLess(p_threshold, pos_prob)

  def test_get_opt_binary_attributions(self):
    """Check that opt binary attribution are actually binary."""
    y_true = [0, 1]
    y_prob = [0.2, 0.8]
    att_true = [self._setup_graph_from_nodes(y_true)]
    att_prob = [self._setup_graph_from_nodes(y_prob)]
    att_pred = att_metrics.get_opt_binary_attributions(att_true, att_prob)
    node_pred = [att.nodes for att in att_pred]
    # Check nodes are binary.
    np.testing.assert_array_equal(np.unique(node_pred), np.array([0, 1]))


if __name__ == '__main__':
  absltest.main()
