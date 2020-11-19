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
"""Attribution tasks."""
import collections
import enum
from typing import (Any, Callable, List, MutableMapping, Optional, Text, Tuple,
                    Union)

import graph_nets
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

from graph_attribution import attribution_metrics as att_metrics
from graph_attribution import graphs as graph_utils
from graph_attribution import templates

try:
    from rdkit import Chem
    from graph_attribution import mol_tasks
    from graph_attribution import fragment_identifier
    HAS_RDKIT = True
except ModuleNotFoundError:
    HAS_RDKIT = False

print(f'rdkit detected? {str(HAS_RDKIT)}')


# Typing aliases.
GraphsTuple = graph_nets.graphs.GraphsTuple
TargetType = templates.TargetType
AttributionDataset = templates.AttributionDataset
AttributionTaskType = templates.AttributionTaskType
AttributionTask = templates.AttributionTask


def normalize_attributions(att_list: List[GraphsTuple],
                           positive: bool = False) -> List[GraphsTuple]:
    """Normalize all nodes to 0 to 1 range via quantiles."""
    all_values = np.concatenate([att.nodes for att in att_list])
    all_values = all_values[all_values > 0] if positive else all_values

    normalizer = sklearn.preprocessing.QuantileTransformer()
    normalizer.fit(all_values.reshape(-1, 1))
    new_att = []
    for att in att_list:
        normed_nodes = normalizer.transform(att.nodes.reshape(-1, 1)).ravel()
        new_att.append(att.replace(nodes=normed_nodes))
    return new_att


class DummyDataset(AttributionDataset):
    """A dummy dataset for when there is no generator (i.g. data was loaded)."""

    def __init__(self, name: Text):
        self._name = name

    @property
    def name(self) -> Text:
        return self._name

    def get_true_predictions(self, x: List[Any]) -> np.ndarray:
        raise NotImplementedError(f'{self.name} dataset is a DummyDataset')

    def get_true_attributions(self, x: List[Any]) -> List[GraphsTuple]:
        raise NotImplementedError(f'{self.name} dataset is a DummyDataset')


class RegresionTaskType(AttributionTaskType):
    """Regresion att_metrics for an attribution task."""

    def __init__(self):
        self.n_outputs = 1

    def get_nn_activation_fn(self) -> templates.Activation:
        """Activation useful for NN model building."""
        return tf.identity

    def get_nn_loss_fn(self) -> templates.LossFunction:
        """Loss function useful for NN model training."""

        def loss_fn(y_true, y_pred):
            return tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred))

        return loss_fn

    def evaluate_predictions(self, y_true: np.ndarray,
                             y_pred: np.ndarray) -> MutableMapping[Text, Any]:
        """Scores predictions and return a dict of (metric_name, value)."""
        values = [('R2', sklearn.metrics.r2_score(y_true, y_pred)),
                  ('RMSE', att_metrics.rmse(y_true, y_pred)),
                  ('tau', att_metrics.kendall_tau_score(y_true, y_pred)),
                  ('r', att_metrics.pearson_r_score(y_true, y_pred))]
        return collections.OrderedDict(values)

    def preprocess_attributions(self,
                                att_list: List[GraphsTuple]) -> List[GraphsTuple]:
        """Prepare attribtuions for visualization or evaluation."""
        new_att = []
        for att in att_list:
            att = graph_utils.cast_to_np(graph_utils.reduce_sum_edges(att))
            new_att.append(att)
        return new_att

    def evaluate_attributions(
        self,
        true_att: List[GraphsTuple],
        pred_att: List[GraphsTuple],
        reducer_fn: Optional[Callable[[np.ndarray], Any]] = None
    ) -> MutableMapping[Text, Any]:
        """Scores attributions, return dict of (metric, reduce_fn(values))."""
        reducer_fn = reducer_fn or np.nanmean
        pred_att = self.preprocess_attributions(pred_att)
        stats = collections.OrderedDict()
        stats['ATT tau'] = reducer_fn(
            att_metrics.nodewise_kendall_tau_score(true_att, pred_att))
        stats['ATT r'] = reducer_fn(
            att_metrics.nodewise_pearson_r_score(true_att, pred_att))
        return stats


class BinaryClassificationTaskType(AttributionTaskType):
    """Binary classification metrics for an attribution task."""

    def __init__(self, use_magnitude: bool = False):
        self.n_outputs = 1
        self.use_magnitude = use_magnitude

    def get_nn_activation_fn(self) -> templates.Activation:
        return tf.nn.sigmoid

    def get_nn_loss_fn(self) -> templates.LossFunction:

        def loss_fn(y_true, y_pred):
            return tf.reduce_mean(
                tf.losses.binary_crossentropy(y_true, y_pred))

        return loss_fn

    def evaluate_predictions(self, y_true: np.ndarray,
                             y_prob: np.ndarray) -> MutableMapping[Text, Any]:
        p_tol = att_metrics.get_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= p_tol).astype(np.float32)
        return collections.OrderedDict([
            ('AUROC', att_metrics.nan_auroc_score(y_true, y_prob)),
            ('F1', att_metrics.nan_f1_score(y_true, y_pred)),
            ('ACC', att_metrics.accuracy_score(y_true, y_pred))
        ])

    def preprocess_attributions(self,
                                many_att: List[GraphsTuple],
                                positive: bool = False,
                                normalize: bool = False) -> List[GraphsTuple]:
        """Prepare attributions for visualization or evaluation."""
        new_att = []
        for att in many_att:
            # If the attribution is 2D, then we pick the last truth.
            if att.nodes.ndim > 1:
                att = att.replace(nodes=att.nodes[:, -1])
            if self.use_magnitude:
                att = att.replace(nodes=np.abs(att.nodes))
                if att.edges is not None:
                    att = att.replace(edges=np.abs(att.edges))

            att = graph_utils.cast_to_np(graph_utils.reduce_sum_edges(att))
            new_att.append(att)

        if normalize:
            new_att = normalize_attributions(new_att, positive)

        return new_att

    def evaluate_attributions(
        self,
        att_true: List[GraphsTuple],
        att_pred: List[GraphsTuple],
        reducer_fn: Optional[Callable[[np.ndarray], Any]] = None
    ) -> MutableMapping[Text, Any]:
        reducer_fn = reducer_fn or np.nanmean
        att_probs = self.preprocess_attributions(att_pred, normalize=True)
        att_true_last = self.preprocess_attributions(att_true)
        att_binary = att_metrics.get_opt_binary_attributions(
            att_true_last, att_probs)
        stats = collections.OrderedDict()
        stats['ATT AUROC'] = reducer_fn(
            att_metrics.attribution_auroc(att_true, att_probs))
        stats['ATT F1'] = reducer_fn(
            att_metrics.attribution_f1(att_true, att_binary))
        stats['ATT ACC'] = reducer_fn(
            att_metrics.attribution_accuracy(att_true, att_binary))
        return stats


class MultiClassificationTaskType(templates.AttributionTaskType):
    """Multi classification metrics for an attribution task."""

    def __init__(self, n_classes, use_magnitude: bool = False):
        self.n_outputs = n_classes
        self.use_magnitude = use_magnitude

    def get_nn_activation_fn(self) -> templates.Activation:
        return tf.nn.softmax

    def get_nn_loss_fn(self) -> templates.LossFunction:

        def loss_fn(y_true, y_pred):
            return tf.reduce_mean(
                tf.losses.categorical_crossentropy(y_true, y_pred))

        return loss_fn

    def evaluate_predictions(self, y_true: np.ndarray,
                             y_prob: np.ndarray) -> MutableMapping[Text, Any]:
        label_true = np.argmax(y_true, axis=1)
        label_pred = np.argmax(y_prob, axis=1)
        return collections.OrderedDict([
            ('ACC', att_metrics.accuracy_score(label_true, label_pred))
        ])

    def preprocess_attributions(self,
                                att_list: List[GraphsTuple],
                                normalize: bool = False) -> List[GraphsTuple]:
        """Prepare attributions for visualization or evaluation."""
        new_att = []
        for att in att_list:
            att = graph_utils.cast_to_np(att)
            if att.nodes.ndim == 2:
                att = att.replace(nodes=np.squeeze(att.nodes))
            if self.use_magnitude:
                att = att.replace(nodes=np.abs(att.nodes))
                if att.edges is not None:
                    att = att.replace(edges=np.abs(att.edges))

            att = graph_utils.cast_to_np(graph_utils.reduce_sum_edges(att))
            new_att.append(att)

        if normalize:
            new_att = normalize_attributions(new_att, positive=False)

        return new_att

    def evaluate_attributions(
        self,
        att_true: List[GraphsTuple],
        att_pred: List[GraphsTuple],
        reducer_fn: Optional[Callable[[np.ndarray], Any]] = None
    ) -> MutableMapping[Text, Any]:
        reducer_fn = reducer_fn or np.nanmean
        att_true = [graph_utils.cast_to_np(att) for att in att_true]
        att_probs = self.preprocess_attributions(att_pred, normalize=True)
        att_binary = att_metrics.get_opt_binary_attributions(
            att_true, att_probs)

        stats = collections.OrderedDict()
        stats['ATT AUROC'] = reducer_fn(
            att_metrics.attribution_auroc(att_true, att_probs))
        stats['ATT ACC'] = reducer_fn(
            att_metrics.attribution_auroc(att_true, att_binary))
        return stats


# Aliases for shorter task definitions.
if HAS_RDKIT:
    _frag_rule = fragment_identifier.BasicFragmentRule
    def _and_rules(rules): return fragment_identifier.CompositeRule(
        'AND', rules)

    benzene_dataset = mol_tasks.FragmentLogicDataset(
        _frag_rule('benzene', 'c1ccccc1'))
    logic7_dataset = mol_tasks.FragmentLogicDataset(
        _and_rules(
            [_frag_rule('flouride', '[FX1]'),
             _frag_rule('carbonyl', '[CX3]=O')]))
    logic8_dataset = mol_tasks.FragmentLogicDataset(
        _and_rules([
            _frag_rule('unbranched alkane', '[R0;D2,D1][R0;D2][R0;D2,D1]'),
            _frag_rule('carbonyl', '[CX3]=O')
        ]))
    logic10_dataset = mol_tasks.FragmentLogicDataset(
        _and_rules([
            _frag_rule('amine', '[NX3;H2]'),
            _frag_rule('ether2', '[OD2](C)C'),
            _frag_rule('benzene', '[cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1')
        ]))
    crippen_dataset = mol_tasks.CrippenLogPDataset()
else:
    benzene_dataset = DummyDataset('benzene')
    logic7_dataset = DummyDataset('logic7')
    logic8_dataset = DummyDataset('logic8')
    logic10_dataset = DummyDataset('logic10')
    crippen_dataset = DummyDataset('crippen')


class Task(enum.Enum):
    """Types of tasks readily implemented."""
    benzene = 'benzene'
    logic7 = 'logic7'
    logic8 = 'logic8'
    logic10 = 'logic10'
    crippen = 'crippen'
    bashapes = 'bashapes'
    treegrid = 'treegrid'
    bacommunity = 'bacommunity'


NODE_TASKS = [Task.bashapes]
GLOBALS_TASKS = [
    Task.crippen, Task.benzene, Task.logic7, Task.logic8, Task.logic10
]
MOL_TASKS = [
    Task.crippen,
    Task.benzene,
    Task.logic7,
    Task.logic8,
    Task.logic10]


def get_task(task: Union[Task, Text]) -> templates.AttributionTask:
    """Retrieve a task by it's enum/name."""
    task = Task(task)
    task_map = {
        Task.crippen:
            AttributionTask(crippen_dataset, RegresionTaskType(),
                            TargetType.globals),
        Task.benzene:
            AttributionTask(benzene_dataset, BinaryClassificationTaskType(),
                            TargetType.globals),
        Task.logic7:
            AttributionTask(logic7_dataset, BinaryClassificationTaskType(),
                            TargetType.globals),
        Task.logic8:
            AttributionTask(logic8_dataset, BinaryClassificationTaskType(),
                            TargetType.globals),
        Task.logic10:
            AttributionTask(logic10_dataset, BinaryClassificationTaskType(),
                            TargetType.globals),
        Task.bashapes:
            AttributionTask(
                DummyDataset('BA-Shapes'),
                MultiClassificationTaskType(n_classes=4, use_magnitude=True),
                TargetType.nodes),
        Task.treegrid:
            AttributionTask(
                DummyDataset('Tree-Grid'),
                BinaryClassificationTaskType(use_magnitude=True),
                TargetType.nodes),
        Task.bacommunity:
            AttributionTask(
                DummyDataset('BA-community'),
                MultiClassificationTaskType(n_classes=8, use_magnitude=False),
                TargetType.nodes),
    }
    return task_map[task]
