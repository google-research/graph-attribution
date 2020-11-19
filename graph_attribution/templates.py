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
"""Templates for models, tasks and attribution techniques."""
import abc
import enum
from typing import Any, Callable, List, MutableMapping, Optional, Text, Tuple

import graph_nets
import numpy as np
import tensorflow as tf

NodeEdgeTensors = Tuple[tf.Tensor, tf.Tensor]
OrderedDict = MutableMapping
GraphsTuple = graph_nets.graphs.GraphsTuple
LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
Activation = Callable[[tf.Tensor], tf.Tensor]


class TargetType(enum.Enum):
    """Types of targets for prediction."""
    nodes = 'nodes'
    globals = 'globals'
    edges = 'edges'


class TransparentModel(abc.ABC):
    """Abstract class for a Model that can be probed with AttributionTechnique."""

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __call__(self, inputs: GraphsTuple) -> tf.Tensor:
        """Typical forward pass for the model."""

    @abc.abstractmethod
    def predict(self,
                inputs: GraphsTuple,
                task_index: Optional[int] = None,
                batch_index: Optional[int] = None) -> tf.Tensor:
        """Forward pass with output set on the task of interest, returns 1D tensor.

        Often models will have a multi-dimensional output, or several outputs.
        Output of predict is the subset of the model output
        that is relevant for attribution. This is used in many TransparentModel
        methods.

        Args:
          inputs: input for model.
          task_index: output of prediction tensor to focus on. Defaults to 0.
          batch_index: example of prediction tensor to focus on. Default to all.
        """

    @abc.abstractmethod
    def get_gradient(self,
                     inputs: GraphsTuple,
                     task_index: Optional[int] = None,
                     batch_index: Optional[int] = None) -> NodeEdgeTensors:
        """Gets gradient of target w.r.t. to the input."""

    @abc.abstractmethod
    def get_gap_activations(self,
                            inputs: GraphsTuple,
                            task_index: Optional[int] = None) -> NodeEdgeTensors:
        """Gets node-wise and edge-wise contributions to graph embedding.

        Asummes there is a global average pooling (GAP) layer to produce the
        graph embedding (u). This returns the pre-pooled activations.
        With this layer the graph embedding is of the form
        u = sum_i nodes_i + sum_j edges_j , nodes_i and edges_j have been
        transformed to the same dim as u (i.g. via a MLP). Useful for CAM.

        Args:
          inputs: Model inputs.
          task_index: index for task to predict.
        """

    @abc.abstractmethod
    def get_prediction_weights(self,
                               task_index: Optional[int] = None) -> tf.Tensor:
        """Gets last layer prediction weights.

        Assumes layer is of type Linear with_bias=False, useful for CAM.

        Args:
          task_index: dim of linear weights (task) to focus on. Defaults to 0.
        """

    @abc.abstractmethod
    def get_intermediate_activations_gradients(
        self,
        x: GraphsTuple,
        task_index: Optional[int] = None,
        batch_index: Optional[int] = None
    ) -> Tuple[List[NodeEdgeTensors], List[NodeEdgeTensors], np.ndarray]:
        """Gets gradients and activations for intermediate layers."""

    @abc.abstractmethod
    def get_attention_weights(self, inputs: GraphsTuple) -> tf.Tensor:
        """Gets attention weights for a GAT-like block."""


class AttributionTechnique(abc.ABC):
    """Abstract class for an attribution technique."""

    name: Text
    sample_size: int  # Number of graphs to hold in memory per input.

    @abc.abstractmethod
    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Compute GraphTuple with node and edges importances.

        Assumes that x (GraphTuple) has node and edge information as 2D arrays
        and the returned attribution will be a list of GraphsTuple, for each
        graph inside of x, with the same shape but with 1D node and edge arrays.

        Args:
          x: Input to get attributions for.
          model: model that gives gradients, predictions, activations, etc.
          task_index: index for task to focus attribution.
          batch_index: index for example to focus attribution.
        """


class AttributionDataset(abc.ABC):
    """Abstract class for an attribution dataset constructor.

    Can be thought as a class to build an attribution dataset,
    generates predictions and attributions.
    """

    @property
    @abc.abstractmethod
    def name(self) -> Text:
        pass

    @abc.abstractmethod
    def get_true_attributions(self, x: List[Any]) -> List[GraphsTuple]:
        """Computes ground truth attribution for some list of inputs x.

        If there are k datapoints, the GraphsTuple will have k graphs.

        Args:
          x: List of datapoints.
        """

    @abc.abstractmethod
    def get_true_predictions(self, x: List[Any]) -> np.ndarray:
        """Get true prediction values, useful for training a model."""


class AttributionTaskType(abc.ABC):
    """Abstract class for an attribution task type.

    Can be thought as setting a task evaluator. Assumes
    there is a predictive and scorable task for which there is an underlying
    attribution that 'explains' the preditive task. It also has functions to
    aid neural network model building(get_nn_activation_fn) and optimizing
    get_nn_loss_fn) based on the predictive task.
    """
    n_outputs: int

    @abc.abstractmethod
    def evaluate_predictions(self, y_true: np.ndarray,
                             y_pred: np.ndarray) -> OrderedDict[Text, Any]:
        """Evaluate metrics on predictions, return results as (metric, value)."""

    @abc.abstractmethod
    def evaluate_attributions(
        self,
        y_true: List[GraphsTuple],
        y_pred: List[GraphsTuple],
        reducer_fn: Optional[Callable[[np.ndarray], Any]] = None
    ) -> OrderedDict[Text, Any]:
        """Evaluate attribution metrics on predicted attributions.

        Assumes attributions are stored as many graphs with 1D node/edge
        information. reducer_fn will take the metrics on each graph and apply
        a transformation on it (i.g. np.mean, lambda x: x). Results are a dict
        of the form (metric, reducer_fn(values)).

        Args:
          y_true: True attributions.
          y_pred: Predicted attributions.
          reducer_fn: Function that takes numpy arrays.

        Returns:
          results_dict, pairs of metrics and values.
        """

    @abc.abstractmethod
    def get_nn_activation_fn(self) -> Activation:
        """Get activation function for building a NN a predictive task."""

    @abc.abstractmethod
    def get_nn_loss_fn(self) -> LossFunction:
        """Get a loss function for training a NN in a predictive task."""


class AttributionTask():
    """Encapsulation class for (dataset, task_type, target_type).

    Can be thought as setting a problem specification. Assumes there is a way
    of generating ground truth attributions and predictions, as well as a
    framework for evaluating. Has additional information like the target_type for
    a task.
    """

    def __init__(self, dataset: AttributionDataset,
                 task_type: AttributionTaskType, target_type: TargetType):
        self.dataset = dataset
        self.task_type = task_type
        self.target_type = target_type

    @property
    def name(self) -> Text:
        return self.dataset.name

    @property
    def n_outputs(self) -> int:
        return self.task_type.n_outputs

    def evaluate_predictions(self, y_true: np.ndarray,
                             y_pred: np.ndarray) -> OrderedDict[Text, Any]:
        """Evaluate metrics on predictions, return results as (metric, value)."""
        return self.task_type.evaluate_predictions(y_true, y_pred)

    def evaluate_attributions(
        self,
        y_true: List[GraphsTuple],
        y_pred: List[GraphsTuple],
        reducer_fn: Optional[Callable[[np.ndarray], Any]] = None
    ) -> OrderedDict[Text, Any]:
        """Evaluate attribution metrics on predicted attributions.

        Assumes attributions are stored as many graphs with 1D node/edge
        information. reducer_fn will take the metrics on each graph and apply
        a transformation on it (i.g. np.mean, lambda x: x). Results are a dict
        of the form (metric, reducer_fn(values)).

        Args:
          y_true: True attributions.
          y_pred: Predicted attributions.
          reducer_fn: Function that takes numpy arrays.

        Returns:
          results_dict, pairs of metrics and values.
        """
        return self.task_type.evaluate_attributions(y_true, y_pred, reducer_fn)

    def get_nn_activation_fn(self) -> Callable[[Any], Any]:
        """Get activation function for building a NN a predictive task."""
        return self.task_type.get_nn_activation_fn()

    def get_nn_loss_fn(self) -> Callable[[Any, Any], Any]:
        """Get a loss function for training a NN in a predictive task."""
        return self.task_type.get_nn_loss_fn()

    def get_true_attributions(self, x: List[Any]) -> List[GraphsTuple]:
        """Computes ground truth attribution for some list of inputs x."""
        return self.dataset.get_true_attributions(x)

    def get_true_predictions(self, x: List[Any]) -> np.ndarray:
        """Get true prediction values, useful for training a model."""
        return self.dataset.get_true_predictions(x)
