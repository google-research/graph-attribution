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
"""Implementations of graph neural networks, based on graph_nets."""
import enum
from typing import Callable, List, Text, Tuple, Union

import graph_nets
import numpy as np
import sonnet as snt
import tensorflow as tf

from graph_attribution import templates

GraphsTuple = graph_nets.graphs.GraphsTuple
Activation = templates.Activation


class BlockType(enum.Enum):
    """Prediction types for odor tasks."""
    gcn = 'gcn'
    gat = 'gat'
    mpnn = 'mpnn'
    graphnet = 'graphnet'


def print_model(model: snt.Module):
    print(f'{model.__class__.__name__} : {model.name}\n')
    print(snt.format_variables(model.variables))
    n_params = np.sum([np.prod(v.shape) for v in model.variables])
    trainable_params = np.sum(
        [np.prod(v.shape) for v in model.trainable_variables])
    print(f'\nParams: {trainable_params} trainable out of {n_params}')


def get_linear_variables(model: snt.Module) -> List[tf.Variable]:
    """Gets all linear weight variables, useful for regularization."""
    weight_vars = []
    for v in model.trainable_variables:
        layer_name, _ = v.name.split('/')[-2:]
        if layer_name.startswith('linear_'):
            weight_vars.append(v)
    return weight_vars


def cast_activation(act: Union[Text, Activation]) -> Activation:
    """Map string to activation, or just pass the activation function."""
    activations = {
        'selu': tf.nn.selu,
        'softplus': tf.nn.softplus,
        'relu': tf.nn.relu,
        'leaky_relu': tf.nn.leaky_relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax,
        'identity': tf.identity
    }
    if callable(act):
        return act
    else:
        return activations[act]


def get_mlp_fn(
        layer_sizes: List[int],
        act: Union[Text, Activation] = 'relu') -> Callable[[], snt.Module]:
    """Instantiates a new MLP, followed by LayerNorm."""

    def make_mlp():
        return snt.Sequential([
            snt.nets.MLP(
                layer_sizes, activate_final=True, activation=cast_activation(act)),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])

    return make_mlp


def get_graph_attribute(x: GraphsTuple, target_type: templates.TargetType):
    """Like getattr(x, target_type.name)."""
    if target_type == templates.TargetType.nodes:
        return x.nodes
    elif target_type == templates.TargetType.globals:
        return x.globals
    elif target_type == templates.TargetType.edges:
        return x.edges
    else:
        raise NotImplementedError(f'target_type={target_type.name}.')


class ReadoutGAP(snt.Module):
    """Global Average pooling style-layer."""

    def __init__(self, global_size, activation, name='ReadoutGAP'):
        super(ReadoutGAP, self).__init__(name=name)
        reducer = tf.math.unsorted_segment_sum
        self.node_reducer = graph_nets.blocks.NodesToGlobalsAggregator(reducer)
        self.edge_reducer = graph_nets.blocks.EdgesToGlobalsAggregator(reducer)
        self.node_emb = snt.Sequential(
            [snt.Linear(global_size),
             cast_activation(activation)])
        self.edge_emb = snt.Sequential(
            [snt.Linear(global_size),
             cast_activation(activation)])

    def get_activations(
            self, graph: GraphsTuple) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get pre-pooling activations for nodes and edges."""
        return self.node_emb(graph.nodes), self.edge_emb(graph.edges)

    def __call__(self, inputs: GraphsTuple) -> GraphsTuple:
        new_nodes, new_edges = self.get_activations(inputs)
        graph = inputs.replace(nodes=new_nodes, edges=new_edges)
        new_globals = self.node_reducer(graph) + self.edge_reducer(graph)
        return graph.replace(
            nodes=inputs.nodes, edges=inputs.edges, globals=new_globals)


class NodesAggregator(snt.Module):
    """Agregates neighboring nodes based on sent and received nodes."""

    def __init__(self,
                 reducer=tf.math.unsorted_segment_sum,
                 name='nodes_aggregator'):
        super(NodesAggregator, self).__init__(name=name)
        self.reducer = reducer

    def __call__(self, graph):
        num_nodes = tf.reduce_sum(graph.n_node)
        adjacent_nodes = tf.gather(graph.nodes, graph.senders)
        return self.reducer(adjacent_nodes, graph.receivers, num_nodes)


class NodeLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes, but uses edges."""

    def __init__(self, *args, **kwargs):
        super(NodeLayer, self).__init__(*args, use_globals=False, **kwargs)


class GCNLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes using neighboring nodes and edges."""

    def __init__(self, *args, **kwargs):
        super(GCNLayer, self).__init__(*args, use_globals=False, **kwargs)
        self.gather_nodes = NodesAggregator()

    def __call__(self, graph):
        """Collect nodes, adjacent nodes, edges and update to get new nodes.

        Args:
          graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
            features (if `use_received_edges` or `use_sent_edges` is `True`),
            individual nodes features (if `use_nodes` is True) and per graph globals
            (if `use_globals` is `True`) should be concatenable on the last axis.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes.
        """

        nodes_to_collect = []

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_received_edges:
            edge2node = self._received_edges_aggregator(graph)
            nodes_to_collect.append(edge2node)

        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        adjacent_nodes = self.gather_nodes(graph)
        nodes_to_collect.append(adjacent_nodes)

        if self._use_globals:
            # The hint will be an integer if the graph has node features and the total
            # number of nodes is known at tensorflow graph definition time, or None
            # otherwise.
            num_nodes_hint = graph_nets.blocks._get_static_num_nodes(graph)
            nodes_to_collect.append(
                graph_nets.blocks.broadcast_globals_to_nodes(
                    graph, num_nodes_hint=num_nodes_hint))

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        return graph.replace(nodes=updated_nodes)


class NodeEdgeLayer(snt.Module):
    """GNN layer that only updates nodes and edges."""

    def __init__(self, node_model_fn, edge_model_fn, name='NodeEdgeLayer'):
        super(NodeEdgeLayer, self).__init__(name=name)
        self.edge_block = graph_nets.blocks.EdgeBlock(
            edge_model_fn=edge_model_fn, use_globals=False)
        self.node_block = graph_nets.blocks.NodeBlock(
            node_model_fn=node_model_fn, use_globals=False)

    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        return self.node_block(self.edge_block(graph))


# We disable protected access for this module since it uses several
# hidden functions from graph_nets.modules.
# pylint: disable=protected-access


class SelfAttention(snt.Module):
    """Self-attention module.

    Module is the same as graph_nets.modules.SelfAttention but exposing functions
    for attention weights.
    The module is based on the following three papers:
     * A simple neural network module for relational reasoning (RNs):
         https://arxiv.org/abs/1706.01427
     * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
     * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.
    The input to the modules consists of a graph containing values for each node
    and connectivity between them, a tensor containing keys for each node
    and a tensor containing queries for each node.
    The self-attention step consist of updating the node values, with each new
    node value computed in a two step process:
    - Computing the attention weights between each node and all of its senders
     nodes, by calculating sum(sender_key*receiver_query) and using the softmax
     operation on all attention weights for each node.
    - For each receiver node, compute the new node value as the weighted average
     of the values of the sender nodes, according to the attention weights.
    - Nodes with no received edges, get an updated value of 0.
    Values, keys and queries contain a "head" axis to compute independent
    self-attention for each of the heads.
    """

    def __init__(self,
                 node_size: int,
                 node_model_fn: Callable[[], snt.Module],
                 num_heads: int = 1,
                 name='self_attention'):
        """Inits the module.

        Args:
          node_size: Node dimension, will be used for keys and queries also.
          node_model_fn: Function to create a update module for nodes.
          num_heads: Number of heads for multi-headed attention.
          name: The module name.
        """
        super(SelfAttention, self).__init__(name=name)
        self._normalizer = graph_nets.modules._unsorted_segment_softmax
        self.num_heads = num_heads
        self.value_size = node_size // num_heads
        self.key_size = node_size // num_heads
        self.keys = snt.Linear(node_size * num_heads)
        self.queries = snt.Linear(node_size * num_heads)
        self.node_model = node_model_fn()
        self.edge_reducer = graph_nets.blocks.ReceivedEdgesToNodesAggregator(
            tf.math.unsorted_segment_sum)

    def split_heads(self, x: tf.Tensor) -> tf.Tensor:
        x_heads = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads))
        return tf.transpose(x_heads, perm=[0, 2, 1])

    def fold_heads(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reshape(x, (tf.shape(x)[0], -1))

    def attention(self, node_values, node_keys, node_queries, attention_graph):
        """Connects the multi-head self-attention module.

        The self-attention is only computed according to the connectivity of the
        input graphs, with receiver nodes attending to sender nodes.
        Args:
          node_values: Tensor containing the values associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, key_size].
          node_keys: Tensor containing the key associated to each of the nodes. The
            expected shape is [total_num_nodes, num_heads, key_size].
          node_queries: Tensor containing the query associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, query_size]. The
            query size must be equal to the key size.
          attention_graph: Graph containing connectivity information between nodes
            via the senders and receivers fields. Node A will only attempt to attend
            to Node B if `attention_graph` contains an edge sent by Node A and
            received by Node B.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes containing the
          aggregated attended value for each of the nodes with shape
          [total_num_nodes, num_heads, value_size].
        Raises:
          ValueError: if the input graph does not have edges.
        """

        # Sender nodes put their keys and values in the edges.
        # [total_num_edges, num_heads, query_size]
        sender_keys = graph_nets.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_keys))
        # [total_num_edges, num_heads, value_size]
        sender_values = graph_nets.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_values))

        # Receiver nodes put their queries in the edges.
        # [total_num_edges, num_heads, key_size]
        receiver_queries = graph_nets.blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_queries))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        attention_weights_logits = tf.reduce_sum(
            sender_keys * receiver_queries, axis=-1)
        normalized_attention_weights = graph_nets.modules._received_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attented_edges = sender_values * \
            normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = graph_nets.blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.math.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attented_edges))

        outputs = attention_graph.replace(nodes=aggregated_attended_values)
        return outputs, normalized_attention_weights

    def apply_attention(self,
                        inputs: GraphsTuple) -> Tuple[GraphsTuple, tf.Tensor]:
        """Essentially the call, but we return also weights."""
        edge_nodes = tf.concat(
            [inputs.nodes, self.edge_reducer(inputs)], axis=-1)
        v = self.split_heads(tf.tile(edge_nodes, [1, self.num_heads]))
        q = self.split_heads(self.queries(edge_nodes))
        k = self.split_heads(self.keys(edge_nodes))
        out, weights = self.attention(v, q, k, inputs)
        out = out.replace(nodes=self.fold_heads(out.nodes))
        out = out.replace(nodes=self.node_model(out.nodes))
        return out, weights

    def __call__(self, inputs: GraphsTuple) -> GraphsTuple:
        out, _ = self.apply_attention(inputs)
        return out


# pylint: enable=protected-access


class SequentialWithActivations(snt.Sequential):
    """Extend snt.Sequential with function for intermediate activations."""

    def call_with_activations(self, inputs, *args, **kwargs):
        """Same code as snt.call but also stores intermediate activations."""
        outputs = inputs
        acts = []
        for i, mod in enumerate(self._layers):
            if i == 0:
                # Pass additional arguments to the first layer.
                outputs = mod(outputs, *args, **kwargs)
            else:
                outputs = mod(outputs)
            acts.append(outputs)
        return outputs, acts
