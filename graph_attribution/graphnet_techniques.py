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
"""Attribution techniques."""
import collections
import functools
from typing import Callable, List, MutableMapping, Optional, Text

import graph_nets
import numpy as np
import pandas as pd
import tensorflow as tf

from graph_attribution import graphs as graph_utils
from graph_attribution import templates

# Typing aliases.
GraphsTuple = graph_nets.graphs.GraphsTuple
TransparentModel = templates.TransparentModel
AttributionTechnique = templates.AttributionTechnique


def make_reference_fn(node_vec: np.ndarray, edge_vec: np.ndarray):
    """Make reference function."""
    ref_fn = functools.partial(
        graph_utils.make_constant_like, node_vec=node_vec, edge_vec=edge_vec)
    return ref_fn


class RandomBaseline(AttributionTechnique):
    """Random baseline: random node and edge attributions from uniform(0,1)."""

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        rand_nodes = np.random.uniform(size=(x.nodes.shape[0]))
        rand_edges = np.random.uniform(size=(x.edges.shape[0]))
        graphs = x.replace(nodes=rand_nodes, edges=rand_edges, globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class AttentionWeights(AttributionTechnique):
    """Use attention weights as importance features.

      AttentionWeights uses the attention weights from multi-headead
      self-attention GNN blocks. The weights are on edges and are normalized via
      softmax. We reduce attention on all heads and all blocks to arrive to a
      single value for each edge. These value can be interpreted as importance
      values on the connectivity of a graph.


      Based on "Graph Attention Networks" (https://arxiv.org/abs/1710.10903) and
      "GNNExplainer: Generating Explanations for Graph Neural Networks"
      (https://arxiv.org/pdf/1903.03894.pdf).

    """

    def __init__(self,
                 head_reducer: Callable[...,
                                        tf.Tensor] = tf.math.reduce_mean,
                 block_reducer: Callable[...,
                                         tf.Tensor] = tf.math.reduce_mean,
                 name: Optional[Text] = None):
        """Init.

        Args:
          head_reducer: function used to combine attention weights from each
            attention head in a block.
          block_reducer: function used to combine attention weights across blocks.
          name: name for module.
        """
        self.name = name or self.__class__.__name__
        self.sample_size = 1
        self.head_reducer = head_reducer
        self.block_reducer = block_reducer

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        weights = model.get_attention_weights(x)
        weights = tf.stack(weights)  # [n_blocks, n_edges, n_heads]
        weights = self.head_reducer(weights, axis=2)  # [n_blocks, n_edges]
        weights = self.block_reducer(weights, axis=0)  # [n_edges]
        empty_nodes = tf.zeros(len(x.nodes))
        graphs = x.replace(nodes=empty_nodes, edges=weights, globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class GradInput(AttributionTechnique):
    """GradInput: Gradient times input.

    GradInput uses the gradient of a target y w.r.t its input and multiplies it
    by its input. The magnitud of the derivitate at a particular
    atom can be interpreted as a measure of how much the atom needs to be changed
    to least affect the target. Same for edges. The sign gives indication if
    this change is positive or negative. In this sense the gradient is interpreted
    as a measure of importance of each component in the input. An equation for
    this method is:

      GradInput(x) = w^T * x, where w = gradient(y w.r.t x)

    Based on "Deep Inside Convolutional Networks: Visualising Image
    Classification Models and Saliency Maps"
    (https://arxiv.org/pdf/1312.6034.pdf).
    """

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        node_grad, edge_grad = model.get_gradient(x, task_index, batch_index)
        node_weights = tf.einsum('ij,ij->i', x.nodes, node_grad)
        edge_weights = tf.einsum('ij,ij->i', x.edges, edge_grad)
        graphs = x.replace(
            nodes=node_weights,
            edges=edge_weights,
            globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class CAM(AttributionTechnique):
    """CAM: Decompose output as a linear sum of nodes and edges.

    CAM (Class Activation Maps) assumes the model has a global average pooling
    layer (GAP-layer) right before prediction. This means the prediction can be
    written as weighted sum of the pooled elements plus an final activation.
    In the case of graphs, a GAP layer should take nodes and edges activations
    and will sum them to create a graph embedding layer. The CAM model follows
    the equation:

      CAM(x) = (node_activations + edge_activations)*w

    Based on "Learning Deep Features for Discriminative Localization"
    (https://arxiv.org/abs/1512.04150).
    """

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def attribute(self,
                  x,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        node_act, edge_act = model.get_gap_activations(x)
        weights = model.get_prediction_weights()
        node_weights = tf.einsum('ij,j', node_act, weights)
        edge_weights = tf.einsum('ij,j', edge_act, weights)
        graphs = x.replace(
            nodes=node_weights,
            edges=edge_weights,
            globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class GradCAM(AttributionTechnique):
    """GradCAM: intermediate activations and gradients as input importance.

    GradCAM is the gradient version of CAM using ideas from Gradient times Input,
    removing the necessity of a GAP layer.
    For each convolution layer, in the case of graphs a GNN block, the
    activations can be retrieved and interpreted as a transformed version of the
    input. In a GNN intermediate activations are graphs with updated information.
    The gradient of a target y w.r.t these activations can be seen as measure of
    importance. The equation for gradCAM are:

      GradCAM(x) = reduce_i mean(w_i^T G_i(x), axis=-1)

    G_i(x) is the intermediate layer activations.
    reduce_i is an reduction operation over intermediate layers (e.g. mean, sum).

    Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization" (https://arxiv.org/abs/1610.02391).
    """

    def __init__(self,
                 last_layer_only: bool = True,
                 reduce_fn=tf.reduce_mean,
                 name: Optional[Text] = None):
        """GradCAM constructor.

        Args:
          last_layer_only: If to use only the last layer activations, if not will
            use all last activations.
          reduce_fn: Reduction operation for layers, should have the same call
            signature as tf.reduce_mean (e.g. tf.reduce_sum).
          name: identifying label for method.
        """
        self.name = name or self.__class__.__name__
        self.sample_size = 1
        self.last_layer_only = last_layer_only
        self.reduce_fn = reduce_fn
        try:
            reduce_fn([[0], [1]], axis=0)
        except BaseException:
            raise ValueError(
                'reduce_fn should have a signature like tf.reduce_mean!')

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        acts, grads, _ = model.get_intermediate_activations_gradients(
            x, task_index, batch_index)
        node_w, edge_w = [], []
        layer_indices = [-1] if self.last_layer_only else list(
            range(len(acts)))
        for index in layer_indices:
            node_act, edge_act = acts[index]
            node_grad, edge_grad = grads[index]
            node_w.append(tf.einsum('ij,ij->i', node_act, node_grad))
            edge_w.append(tf.einsum('ij,ij->i', edge_act, edge_grad))

        node_weights = self.reduce_fn(node_w, axis=0)
        edge_weights = self.reduce_fn(edge_w, axis=0)
        graphs = x.replace(
            nodes=node_weights,
            edges=edge_weights,
            globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class SmoothGrad(AttributionTechnique):
    """SmoothGrad: Mean attributions on noisy graphs.

    Smoothgrad takes an attribution technique (m), and copies of an initial input
    x that have additinal noise and averages the attributions. Initially used
    to sharpen saliency maps in images, we re-implement this idea for graphs.
    Noisy graphs are graphs with additive gaussian noise that depends on the
    variance of the node and edge information.

      Smoothgrad(x,m) = 1/n sum_i m(x + noise_i)

    From the paper "SmoothGrad: removing noise by adding noise"
    (https://arxiv.org/abs/1706.03825).
    """

    def __init__(self,
                 method: AttributionTechnique,
                 num_samples: int = 100,
                 sigma: float = 0.15,
                 name: Optional[Text] = None):
        """Constructor for SmoothGrad.

        Args:
          method: Base attribution method, should implement AttributionTechnique.
          num_samples: Number of noisy graphs to create.
          sigma: Float that controls noise level.
          name: identifying label for method.
        """
        self.num_samples = num_samples
        self.sigma = sigma
        self.method = method
        self.name = name or f'{self.__class__.__name__}({method.name})'
        self.sample_size = num_samples * method.sample_size

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        n = self.num_samples
        n_nodes = int(tf.reduce_sum(x.n_node))
        n_edges = int(tf.reduce_sum(x.n_edge))
        noisy_x = graph_utils.perturb_graphs_tuple(x, n, self.sigma)
        atts = self.method.attribute(noisy_x, model, task_index, batch_index)
        atts = graph_nets.utils_tf.concat(atts, axis=0)
        many_nodes = tf.reshape(atts.nodes, (n, n_nodes))
        node_weights = tf.reduce_mean(many_nodes, axis=0)
        many_edges = tf.reshape(atts.edges, (n, n_edges))
        edge_weights = tf.reduce_mean(many_edges, axis=0)
        graphs = x.replace(
            nodes=node_weights,
            edges=edge_weights,
            globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))


class IntegratedGradients(AttributionTechnique):
    r"""IG: path intergral between a graph and a counterfactual.

    Because IntegratedGradients is based on path integrals, it has nice
    properties associated to integrals, namely IG(x+y) = IG(x)+ IG(y) and
    if y is a target to predict, then y(x) - y(ref) = sum(IG(x,ref)). This last
    property is useful for sanity checks and also indicates that the difference
    in predictions can be retrieved from the attribution.

      IG(x,ref) = \integral_{1}^{0} grad(y w.r.t. interp(ref,x,t))*stepsize   t

      where stepsize = (x-ref)/n_steps.

    From the paper "Axiomatic Attribution for Deep Networks"
    (https://arxiv.org/abs/1703.01365) and "Using attribution to decode binding
    mechanism in neural network models for chemistry"
    (https://www.pnas.org/content/116/24/11624).
    """

    def __init__(self,
                 num_steps: int,
                 reference_fn: Callable[[GraphsTuple], GraphsTuple],
                 name: Optional[Text] = None):
        """Constructor for IntegratedGradients.

        Args:
          num_steps: Number of steps for integration, more steps is more accurate.
          reference_fn: function that will take a graph and return a reference or
            conterfactual graph.
          name: identifying label for method.
        """
        self.name = name or self.__class__.__name__
        self.make_reference = reference_fn
        self.num_steps = num_steps
        self.sample_size = num_steps

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        n = self.num_steps
        ref = self.make_reference(x)
        n_nodes = tf.reduce_sum(x.n_node)
        n_edges = tf.reduce_sum(x.n_edge)
        interp, node_steps, edge_steps = graph_utils.interpolate_graphs_tuple(
            ref, x, n)
        nodes_grad, edges_grad = model.get_gradient(
            interp, task_index, batch_index)
        # Node shapes: [n_nodes * n, nodes.shape[-1]] -> [n_nodes*n].
        node_values = tf.einsum('ij,ij->i', nodes_grad, node_steps)
        edge_values = tf.einsum('ij,ij->i', edges_grad, edge_steps)
        # Node shapes: [n_nodes * n] -> [n_nodes, n].
        node_values = tf.transpose(tf.reshape(node_values, (n, n_nodes)))
        edge_values = tf.transpose(tf.reshape(edge_values, (n, n_edges)))
        # Node shapes: [n_nodes, n] -> [n_nodes].
        node_ig = tf.reduce_sum(node_values, axis=1)
        edge_ig = tf.reduce_sum(edge_values, axis=1)
        graphs = x.replace(nodes=node_ig, edges=edge_ig, globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))

    def sanity_check(self,
                     x: GraphsTuple,
                     model: TransparentModel,
                     task_index: Optional[int] = None,
                     batch_index: Optional[int] = None) -> pd.DataFrame:
        """IG score should be the difference between x and reference scores."""
        results = []
        ref = self.make_reference(x)
        atts = self.attribute(x, model, task_index, batch_index)
        pred_scores = model.predict(x).numpy()
        ref_scores = model.predict(ref).numpy()
        for index, att in enumerate(atts):
            stats = collections.OrderedDict()
            stats['pred_score'] = pred_scores[index]
            stats['ref_score'] = ref_scores[index]
            stats['score_diff'] = stats['pred_score'] - stats['ref_score']
            stats['node_contrib'] = np.sum(att.nodes)
            stats['edge_contrib'] = np.sum(att.edges)
            stats['ig_score'] = np.sum(att.nodes) + np.sum(att.edges)
            stats['ig_error'] = np.abs(stats['ig_score'] - stats['score_diff'])
            results.append(stats)
        return pd.DataFrame(results)


def get_techniques_dict(
        node_arr: Optional[np.ndarray] = None,
        edge_arr: Optional[np.ndarray] = None,
        use_gap_readout: bool = True,
        use_attention: bool = None) -> MutableMapping[Text, AttributionTechnique]:
    """Dictionary of default methods."""

    methods = [
        RandomBaseline(name='Random'),
        GradInput(name='GradInput'),
        SmoothGrad(GradInput(), name='SmoothGrad(GradInput)'),
        GradCAM(last_layer_only=True, name='GradCAM-last'),
        GradCAM(last_layer_only=False, name='GradCAM-all'),
    ]
    if node_arr is not None and edge_arr is not None:
        ref_fn = make_reference_fn(node_arr, edge_arr)
        methods.append(IntegratedGradients(200, ref_fn, name='IG'))
    if use_gap_readout:
        methods.append(CAM(name='CAM'))
    if use_attention:
        methods.append(AttentionWeights())
    methods = collections.OrderedDict([(m.name, m) for m in methods])
    return methods
