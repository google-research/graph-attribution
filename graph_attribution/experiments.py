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
"""Functions for running experiments (colabs, xm) and storing/saving data."""
import dataclasses
import os
from typing import (Any, Callable, List, MutableMapping, Optional, Text, Tuple,
                    Union)

import graph_nets
import more_itertools
import numpy as np
import sonnet as snt
import tensorflow as tf
from graph_attribution import datasets
from graph_attribution import graphnet_models as models
from graph_attribution import graphnet_techniques as techniques
from graph_attribution import graphs as graph_utils
from graph_attribution import tasks, templates

# Typing alias.
GraphsTuple = graph_nets.graphs.GraphsTuple
TransparentModel = templates.TransparentModel
AttributionTechnique = templates.AttributionTechnique
AttributionTask = templates.AttributionTask
NodeEdgeTensors = templates.NodeEdgeTensors
MethodDict = MutableMapping[Text, AttributionTechnique]
OrderedDict = MutableMapping


def set_seed(random_seed: int):
    """Sets initial seed for random numbers."""
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)


def get_graph_block(block_type: models.BlockType, node_size: int,
                    edge_size: int, global_size: int, index: int) -> snt.Module:
    """Gets a GNN block based on enum and sizes."""
    name = f'{block_type.name}_{index+1}'
    if block_type == models.BlockType.gcn:
        return models.GCNLayer(models.get_mlp_fn([node_size] * 2), name=name)
    elif block_type == models.BlockType.gat:
        return models.SelfAttention(
            node_size, models.get_mlp_fn([node_size] * 2))
    elif block_type == models.BlockType.mpnn:
        return models.NodeEdgeLayer(
            models.get_mlp_fn([node_size] * 2),
            models.get_mlp_fn([edge_size] * 2),
            name=name)
    elif block_type == models.BlockType.graphnet:
        use_globals = index != 0
        return graph_nets.modules.GraphNetwork(
            node_model_fn=models.get_mlp_fn([node_size] * 2),
            edge_model_fn=models.get_mlp_fn([edge_size] * 2),
            global_model_fn=models.get_mlp_fn([global_size] * 2),
            edge_block_opt={'use_globals': use_globals},
            node_block_opt={'use_globals': use_globals},
            global_block_opt={'use_globals': use_globals},
            name=name)
    else:
        raise ValueError(f'block_type={block_type} not implemented')


class GNN(snt.Module, templates.TransparentModel):
    """A general graph neural network for graph property prediction."""

    def __init__(self,
                 node_size: int,
                 edge_size: int,
                 global_size: int,
                 y_output_size: int,
                 block_type: models.BlockType,
                 activation: models.Activation,
                 target_type: templates.TargetType,
                 n_layers: int = 3):
        super(GNN, self).__init__(name=block_type.name)

        # Graph encoding step, basic linear mapping.
        self.encode = graph_nets.modules.GraphIndependent(
            node_model_fn=lambda: snt.Linear(node_size),
            edge_model_fn=lambda: snt.Linear(edge_size))
        # Message passing steps or GNN blocks.
        gnn_layers = [
            get_graph_block(
                block_type,
                node_size,
                edge_size,
                global_size,
                index)
            for index in range(0, n_layers)
        ]
        self.gnn = models.SequentialWithActivations(gnn_layers)
        if target_type == templates.TargetType.globals:
            readout = models.ReadoutGAP(global_size, tf.nn.softmax)
        else:
            readout = graph_nets.modules.GraphIndependent()

        self.readout = readout
        self.linear = snt.Linear(y_output_size, with_bias=False)
        self.activation = models.cast_activation(activation)
        self.pred_layer = snt.Sequential([self.linear, self.activation])
        self.block_type = block_type
        self.target_type = target_type

    def cast_task_batch_index(
            self,
            task_index: Optional[int] = None,
            batch_index: Optional[int] = None) -> Tuple[int, Union[int, slice]]:
        """Provide defaults for task and batch indices when not present."""
        task_index = 0 if task_index is None else task_index
        batch_index = slice(None) if batch_index is None else batch_index
        return task_index, batch_index

    @tf.function(experimental_relax_shapes=True)
    def get_graph_embedding(self, x: GraphsTuple) -> tf.Tensor:
        """Build a graph embedding."""
        out_graph = self.readout(self.gnn(self.encode(x)))
        return models.get_graph_attribute(out_graph, self.target_type)

    def __call__(self, x: GraphsTuple) -> tf.Tensor:
        """Typical forward pass for the model."""
        graph_emb = self.get_graph_embedding(x)
        y = self.pred_layer(graph_emb)
        return y

    def predict(self,
                x: GraphsTuple,
                task_index: Optional[int] = None,
                batch_index: Optional[int] = None) -> tf.Tensor:
        """Forward pass with output set on the task of interest (y[batch_index, task_index])."""
        task_index, batch_index = self.cast_task_batch_index(
            task_index, batch_index)
        return self(x)[batch_index, task_index]

    @tf.function(experimental_relax_shapes=True)
    def get_gradient(self,
                     x: GraphsTuple,
                     task_index: Optional[int] = None,
                     batch_index: Optional[int] = None) -> NodeEdgeTensors:
        """Gets gradient of inputs wrt to the target."""
        with tf.GradientTape(watch_accessed_variables=False) as gtape:
            gtape.watch([x.nodes, x.edges])
            y = self.predict(x, task_index, batch_index)
        nodes_grad, edges_grad = gtape.gradient(y, [x.nodes, x.edges])
        return nodes_grad, edges_grad

    @tf.function(experimental_relax_shapes=True)
    def get_gap_activations(self, x: GraphsTuple) -> NodeEdgeTensors:
        """Gets node-wise and edge-wise contributions to graph embedding."""
        return self.readout.get_activations(self.gnn(self.encode(x)))

    @tf.function(experimental_relax_shapes=True)
    def get_prediction_weights(self,
                               task_index: Optional[int] = None) -> tf.Tensor:
        """Gets last layer prediction weights."""
        task_index, _ = self.cast_task_batch_index(task_index, None)
        w = self.linear.w[:, task_index]
        return w

    @tf.function(experimental_relax_shapes=True)
    def get_intermediate_activations_gradients(
        self,
        x: GraphsTuple,
        task_index: Optional[int] = None,
        batch_index: Optional[int] = None
    ) -> Tuple[List[NodeEdgeTensors], List[NodeEdgeTensors], tf.Tensor]:
        """Gets intermediate layer activations and gradients."""
        task_index, batch_index = self.cast_task_batch_index(
            task_index, batch_index)
        acts = []
        grads = []
        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as gtape:
            gtape.watch([x.nodes, x.edges])
            x = self.encode(x)
            outputs, acts = self.gnn.call_with_activations(x)
            outputs = self.readout(outputs)
            embs = models.get_graph_attribute(outputs, self.target_type)
            y = self.pred_layer(embs)[batch_index, task_index]
        acts = [(act.nodes, act.edges) for act in acts]
        grads = gtape.gradient(y, acts)
        return acts, grads, y

    @tf.function(experimental_relax_shapes=True)
    def get_attention_weights(self, inputs: GraphsTuple) -> List[tf.Tensor]:
        if self.block_type != models.BlockType.gat:
            raise ValueError(
                f'block_type={self.block_type.name}, attention only works with "gat" blocks'
            )

        outs = self.encode(inputs)
        weights = []
        for block in self.gnn._layers:  # pylint: disable=protected-access
            outs, w = block.apply_attention(outs)
            weights.append(w)

        return weights

    @classmethod
    def from_hparams(cls, hp, task:AttributionTask) -> 'GNN':
      return cls(node_size = hp.node_size,
               edge_size = hp.edge_size,
               global_size = hp.global_size,
               y_output_size = task.n_outputs,
               block_type = models.BlockType(hp.block_type),
               activation = task.get_nn_activation_fn(),
               target_type = task.target_type,
               n_layers = hp.n_layers)

def get_batched_attributions(method: AttributionTechnique,
                             model: TransparentModel,
                             inputs: GraphsTuple,
                             batch_size: int = 2500) -> List[GraphsTuple]:
    """Batched attribution since memory (e.g. IG) can be an issue."""
    n = graph_utils.get_num_graphs(inputs)
    att_pred = []
    actual_batch_size = int(np.ceil(batch_size / method.sample_size))
    for chunk in more_itertools.chunked(range(n), actual_batch_size):
        x_chunk = graph_utils.get_graphs_tf(inputs, np.array(chunk))
        att = method.attribute(x_chunk, model)
        att_pred.extend(att)
    return att_pred


def generate_result(model: templates.TransparentModel,
                    method: templates.AttributionTechnique,
                    task: templates.AttributionTask,
                    inputs: GraphsTuple,
                    y_true: np.ndarray,
                    true_atts: List[GraphsTuple],
                    pred_atts: Optional[List[GraphsTuple]] = None,
                    reducer_fn: Optional[Callable[[np.ndarray],
                                                  Any]] = np.nanmean,
                    batch_size: int = 1000) -> MutableMapping[Text, Any]:
    """For a given model, method and task, generate metrics."""
    if pred_atts is None:
        pred_atts = get_batched_attributions(method, model, inputs, batch_size)
    result = task.evaluate_attributions(
        true_atts, pred_atts, reducer_fn=reducer_fn)
    # Need to reshape since predict returns a 1D array.
    y_pred = model.predict(inputs).numpy().reshape(-1, 1)
    result.update(task.evaluate_predictions(y_true, y_pred))
    result['Task'] = task.name
    result['Technique'] = method.name
    result['Model'] = model.name
    return result


@dataclasses.dataclass(frozen=True)
class ExperimentData:
    """Helper class to hold all data relating to an experiment."""
    x_train: GraphsTuple
    x_test: GraphsTuple
    y_train: np.ndarray
    y_test: np.ndarray
    att_test: List[GraphsTuple]
    x_aug: Optional[GraphsTuple] = None
    y_aug: Optional[np.ndarray] = None

    @classmethod
    def from_data_and_splits(cls, x, y, att, train_index,
                             test_index, x_aug=None, y_aug=None):
        """Build class from data and split indices."""
        if np.intersect1d(train_index, test_index).shape[0]:
            raise ValueError('train/test indices have overlap!.')
        return cls(
            x_train=graph_utils.get_graphs_tf(x, np.array(train_index)),
            x_test=graph_utils.get_graphs_tf(x, np.array(test_index)),
            y_train=y[train_index],
            y_test=y[test_index],
            att_test=[att[i] for i in test_index],
            x_aug=x_aug, y_aug=y_aug)


def get_experiment_setup(
    task_type: Union[tasks.Task, Text], block_type: Union[models.BlockType,
                                                          Text]
) -> Tuple[ExperimentData, AttributionTask, MethodDict]:
    """Get experiment data based on task_name."""
    task_type = tasks.Task(task_type)
    task = tasks.get_task(task_type)
    fnames = datasets.get_default_experiment_filenames(task_type)
    has_null_features = os.path.exists(fnames['null'])
    expects_aug = isinstance(task.task_type, tasks.BinaryClassificationTaskType
                             ) and has_null_features

    # Load inputs, labels and attributions.
    x = datasets.load_graphstuples(fnames['x'])
    if len(x) > 1:
        raise ValueError(f'Expected a single graph for x={fnames["x"]}')
    x = x[0]

    y = datasets.load_npz(fnames['y'])['y']
    att = datasets.load_graphstuples(fnames['att'])
    train_index, test_index = datasets.load_train_test_indices(
        fnames['splits'])

    x_aug = datasets.load_graphstuples(
        fnames['x_aug']) if expects_aug else None
    y_aug = datasets.load_npz(fnames['y_aug'])[
        'y_aug'] if expects_aug else None

    # Setup experiment data
    exp = ExperimentData.from_data_and_splits(x, y, att, train_index, test_index,
                                              x_aug, y_aug)
    # Null vectors for IG are possible when a graph has node/edge features.
    if has_null_features:
        null_data = datasets.load_npz(fnames['null'])
        node_null, edge_null = null_data['node'], null_data['edge']
    else:
        node_null, edge_null = None, None

    # Get available methods for the problem.
    use_attention = models.BlockType(block_type) == models.BlockType.gat
    use_gap_readout = task.target_type == templates.TargetType.globals
    methods = techniques.get_techniques_dict(node_null, edge_null,
                                             use_gap_readout, use_attention)

    return exp, task, methods
