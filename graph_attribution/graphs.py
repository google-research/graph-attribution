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
"""Extra functions for manipulating GraphsTuple objects."""
from typing import Iterator, List, Tuple

import graph_nets
import numpy as np
import tensorflow as tf
import tree

GraphsTuple = graph_nets.graphs.GraphsTuple

# Alias to mirror the tf version.
cast_to_np = graph_nets.utils_tf.nest_to_numpy

# Numpy and tf compatible version of graph_nets.utils_tf.get_num_graphs


def get_num_graphs(graph): return graph.n_node.shape[0]


def get_input_spec(x: GraphsTuple) -> tf.TensorSpec:
    """Gets input signature for a graphstuple, useful for tf.function."""
    return graph_nets.utils_tf.specs_from_graphs_tuple(
        x, dynamic_num_graphs=True)


def print_graphs_tuple(graphs: GraphsTuple):
    """Print a graph tuple's shapes and contents."""
    print("Shapes of GraphsTuple's fields:")
    print(
        graphs.map(
            lambda x: x if x is None else x.shape,
            fields=graph_nets.graphs.ALL_FIELDS))


def cast_to_tf(graphs: GraphsTuple) -> GraphsTuple:
    """Convert GraphsTuple numpy arrays to tf.Tensor."""

    def cast_fn(x):
        return tf.convert_to_tensor(x) if isinstance(x, np.ndarray) else x

    return tree.map_structure(cast_fn, graphs)


def reduce_sum_edges(graphs: GraphsTuple) -> GraphsTuple:
    """Adds edge information into nodes and sets edges to None."""
    if graphs.nodes.ndim > 1:
        raise ValueError('Can only deal with 1D node information.')
    if graphs.edges is not None and graphs.edges.ndim > 1:
        raise ValueError('Can only deal with 1D edge information.')

    if graphs.edges is None:
        return graphs

    num_nodes = tf.reduce_sum(graphs.n_node)
    edge_contribution = tf.math.unsorted_segment_sum(graphs.edges,
                                                     graphs.receivers, num_nodes)
    new_nodes = graphs.nodes + edge_contribution
    return graphs.replace(nodes=new_nodes, edges=None)


def binarize_np_nodes(graph: GraphsTuple, tol: float) -> GraphsTuple:
    """Binarize node values based on a threshold, useful for classification."""
    return graph.replace(nodes=(graph.nodes >= tol).astype(np.float32))


def make_constant_like(graphs: GraphsTuple, node_vec: np.ndarray,
                       edge_vec: np.ndarray) -> GraphsTuple:
    """Make a similar graph but with constant nodes and edges."""
    using_tensors = isinstance(graphs.nodes, tf.Tensor)
    nodes = np.tile(node_vec, (sum(graphs.n_node), 1))
    edges = np.tile(edge_vec, (sum(graphs.n_edge), 1))
    if using_tensors:
        nodes = tf.convert_to_tensor(nodes, graphs.nodes.dtype)
        edges = tf.convert_to_tensor(edges, graphs.edges.dtype)
    return graphs.replace(nodes=nodes, edges=edges)


def segment_mean_stddev(
        data: tf.Tensor, segment_counts: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculate mean and stddev for segmented tensor (e.g.

    ragged-like).
    Expects a 2D tensor for data and will return mean and std in the same shape,
    with repeats acoording to segment_counts.

    Args:
      data: 2D tensor.
      segment_counts: 1D int tensor with counts for each segment. Should satisfy
        sum(segment_counts) = data.shape[0].

    Returns:
      Segment-wise mean and std, replicated to same shape as data.
    """
    segment_ids = tf.repeat(
        tf.range(segment_counts.shape[0]), segment_counts, axis=0)
    mean_per_segment = tf.math.segment_mean(data, segment_ids)
    mean = tf.repeat(mean_per_segment, segment_counts, axis=0)
    diff_squared_sum = tf.math.segment_sum(tf.square(data - mean), segment_ids)
    counts = tf.reshape(tf.cast(segment_counts, tf.float32), (-1, 1))
    std_per_segment = tf.sqrt(diff_squared_sum / counts)
    std = tf.repeat(std_per_segment, segment_counts, axis=0)
    return mean, std


def perturb_graphs_tuple(graphs: GraphsTuple, num_samples: int,
                         sigma: float) -> GraphsTuple:
    """Sample graphs with additive gaussian noise.

    For a given collection of graphs we create noisey versions of the initial
    graphs by summing random normal noise scaled by a constant factor (sigma)
    and per-graph variance on node and edge information. Connectivity is the
    same.

    Args:
      graphs: input graphs on which to add noise.
      num_samples: number of times to create noisy graphs.
      sigma: scaling factor for noise.

    Returns:
      GraphsTuple with num_samples times more graphs.
    """

    _, node_stddev = segment_mean_stddev(graphs.nodes, graphs.n_node)
    _, edge_stddev = segment_mean_stddev(graphs.edges, graphs.n_edge)

    def add_noise(x, stddev):
        return x + tf.random.normal(x.shape,
                                    stddev=sigma * stddev, dtype=x.dtype)

    graph_list = []
    for _ in tf.range(num_samples):
        graph = graphs.replace(
            nodes=add_noise(graphs.nodes, node_stddev),
            edges=add_noise(graphs.edges, edge_stddev))
        graph_list.append(graph)

    return graph_nets.utils_tf.concat(graph_list, axis=0)


def split_graphs_tuple(graphs: GraphsTuple) -> Iterator[GraphsTuple]:
    """Converts several grouped graphs into a list of single graphs."""
    n = get_num_graphs(graphs)
    nodes = []
    node_offsets = [0] + np.cumsum(graphs.n_node).tolist()
    for i, j in zip(node_offsets[:-1], node_offsets[1:]):
        nodes.append(graphs.nodes[i:j])
    edges = []
    has_edges = graphs.edges is not None
    receivers, senders = [], []
    edge_offsets = [0] + np.cumsum(graphs.n_edge).tolist()
    for node_offset, i, j in zip(node_offsets[:-1], edge_offsets[:-1],
                                 edge_offsets[1:]):
        if has_edges:
            edges.append(graphs.edges[i:j])
        else:
            edges.append(None)

        receivers.append(graphs.receivers[i:j] - node_offset)
        senders.append(graphs.senders[i:j] - node_offset)

    if graphs.globals is None:
        g_globals = [None for i in range(n)]
    else:
        g_globals = [graphs.globals[i] for i in range(n)]

    graph_list = map(GraphsTuple, nodes, edges, receivers, senders, g_globals,
                     graphs.n_node[:, np.newaxis], graphs.n_edge[:, np.newaxis])

    return graph_list


def get_graphs_np(graphs: GraphsTuple, indices=List[int]) -> GraphsTuple:
    """Gets a new graphstuple (numpy) based on a list of indices."""
    node_indices = np.insert(np.cumsum(graphs.n_node), 0, 0)
    node_slice = np.concatenate(
        [np.arange(node_indices[i], node_indices[i + 1]) for i in indices])
    nodes = graphs.nodes[node_slice]

    edge_indices = np.insert(np.cumsum(graphs.n_edge), 0, 0)
    edge_slice = np.concatenate(
        [np.arange(edge_indices[i], edge_indices[i + 1]) for i in indices])

    edges = graphs.edges[edge_slice] if graphs.edges is not None else None

    n_edge = graphs.n_edge[indices]
    n_node = graphs.n_node[indices]

    offsets = np.repeat(node_indices[indices], graphs.n_edge[indices])
    new_offsets = np.insert(np.cumsum(n_node), 0, 0)
    senders = graphs.senders[edge_slice] - offsets
    receivers = graphs.receivers[edge_slice] - offsets
    senders = senders + np.repeat(new_offsets[:-1], n_edge)
    receivers = receivers + np.repeat(new_offsets[:-1], n_edge)
    g_globals = graphs.globals[indices] if graphs.globals is not None else None
    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=g_globals,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge)


def get_graphs_tf(graphs: GraphsTuple, indices: np.ndarray) -> GraphsTuple:
    """Gets a new graphstuple (tf) based on a list of indices."""
    node_indices = tf.concat(
        [tf.constant([0]), tf.cumsum(graphs.n_node)], axis=0)
    node_starts = tf.gather(node_indices, indices)
    node_ends = tf.gather(node_indices, indices + 1)
    node_slice = tf.ragged.range(node_starts, node_ends).values
    nodes = tf.gather(graphs.nodes, node_slice)

    edge_indices = tf.concat(
        [tf.constant([0]), tf.cumsum(graphs.n_edge)], axis=0)
    edge_starts = tf.gather(edge_indices, indices)
    edge_ends = tf.gather(edge_indices, indices + 1)
    edge_slice = tf.ragged.range(edge_starts, edge_ends).values

    edges = tf.gather(graphs.edges,
                      edge_slice) if graphs.edges is not None else None

    n_edge = tf.gather(graphs.n_edge, indices)
    n_node = tf.gather(graphs.n_node, indices)

    offsets = tf.repeat(node_starts, tf.gather(graphs.n_edge, indices))
    senders = tf.gather(graphs.senders, edge_slice) - offsets
    receivers = tf.gather(graphs.receivers, edge_slice) - offsets
    new_offsets = tf.concat([tf.constant([0]), tf.cumsum(n_node)], axis=0)
    senders = senders + tf.repeat(new_offsets[:-1], n_edge)
    receivers = receivers + tf.repeat(new_offsets[:-1], n_edge)

    g_globals = tf.gather(graphs.globals,
                          indices) if graphs.globals is not None else None

    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=g_globals,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge)


def _interp_array(start: tf.Tensor, end: tf.Tensor,
                  num_steps: int) -> tf.Tensor:
    """Linearly interpolate 2D tensors, returns 3D tensors.

    Args:
      start: 2D tensor for start point of interpolation of shape [x,y].
      end: 2D tensor as end point of interpolation of shape [x,y] (same as start).
      num_steps: number of steps to interpolate.

    Returns:
      New tensor of shape [num_steps, x, y]
    """
    alpha = tf.linspace(0., 1., num_steps)
    beta = 1 - alpha
    return tf.einsum('a,bc->abc', alpha, end) + tf.einsum('a,bc->abc', beta,
                                                          start)


def interpolate_graphs_tuple(
        start: GraphsTuple, end: GraphsTuple,
        num_steps: int) -> Tuple[GraphsTuple, tf.Tensor, tf.Tensor]:
    """Interpolate two graphs of same shape."""
    nodes_interp = _interp_array(start.nodes, end.nodes, num_steps)
    edges_interp = _interp_array(start.edges, end.edges, num_steps)
    node_steps = tf.tile(nodes_interp[1] - nodes_interp[0], (num_steps, 1))
    edge_steps = tf.tile(edges_interp[1] - edges_interp[0], (num_steps, 1))
    graphs = []
    for nodes, edges in zip(nodes_interp, edges_interp):
        graphs.append(end.replace(nodes=nodes, edges=edges))

    interp_graph = graph_nets.utils_tf.concat(graphs, axis=0)
    return interp_graph, node_steps, edge_steps
