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

"""Functionality related to visualizing attributions."""
from typing import Dict, List, Optional, Text, Tuple

import graph_nets
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Typing aliases.
GraphsTuple = graph_nets.graphs.GraphsTuple
ColorMap = matplotlib.cm.ScalarMappable

PLOT_NAME_MAPS = {
    'ATT AUROC': 'Attribution auROC',
    'ATT tau': r'Attribution Kendall $\tau$',
    'ATT r': r'Attribution Pearson $r$',
    'graphnet': 'GraphNets',
    'gat': 'GAT',
    'gcn': 'GCN',
    'mpnn': 'MPNN',
    'block_type': '',
    'Technique': '',
    'SmoothGrad(GradInput)': 'SmoothGrad(GI)',
    'l2_reg': 'L2 Reg. Coeff.',
    '(amine & ether2 & benzene)': 'Amine AND Ether AND Benzene',
    '(flouride & carbonyl)': 'Fluoride AND Carbonyl',
    '(unbranched alkane & carbonyl)': 'Unbranched Alkane AND Carbonyl',
    'benzene': 'Benzene',
    'logic7': 'Fluoride AND Carbonyl',
    'logic8': 'Unbranched Alkane AND Carbonyl',
    'logic10': 'Amine AND Ether AND Benzene',
    'crippen': 'CrippenLogP'
}


def get_nx_subgraph(g: nx.Graph, node_index: int,
                    radius: int) -> Tuple[nx.Graph, np.ndarray]:
    """Extract a subgraph centered on a node and with a given neighborhood.

    Gets a subgraph using networkx's single_source_shortest_path_length.
    This function retrieves all nodes that are at a maxmium distance (radius)
    for a given node (node_index) and constructs a subgraph with these nodes.

    Args:
      g: a networkx graph.
      node_index: the node we want to center the distance lookup.
      radius: cutoff distance for the lookup, for example radius=2 will look up
        nodes that have at most a distance of 2 edges from node_index.

    Returns:
      sub_graph, sub_indices
    """
    length = nx.single_source_shortest_path_length(g, node_index, radius)
    sub_graph = g.subgraph(list(length.keys()))
    sub_indices = np.array(list(sub_graph.nodes))
    return sub_graph, sub_indices


def draw_nx_graph(g: nx.Graph,
                  labels: np.ndarray,
                  cmap: Text = 'PiYG',
                  pos: Optional[Dict[int, np.ndarray]] = None):
    """Draw a networkx plot of a graph.

    Args:
      g: a networkx graph.
      labels: values on nodes, used for coloring the nodes.
      cmap: matplotlib colormap used to convert values into rgb.
      pos: dictionary that maps node indices to 2D coordinates. By default will
        use nx.spring_layout.
    """
    plt.figure(figsize=(12, 12))
    pos = pos or nx.spring_layout(g)
    nx.draw_networkx(g, pos=pos, node_color=labels, cmap=cmap)
    plt.axis('off')


def get_regression_colormaps(atts: List[GraphsTuple]) -> List[ColorMap]:
    """Gets colormaps based on a list of attributions."""
    cmap_list = []
    for att in atts:
        vmin, vmax = np.min(att.nodes), np.max(att.nodes)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        pal = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
        cmap_list.append(pal)
    return cmap_list


def get_binaryclass_colormaps(atts: List[GraphsTuple]) -> List[ColorMap]:
    """Gets colormaps based on a list of attributions."""
    cmap_list = []
    for att in atts:
        vmax = np.max(att.nodes)
        colors = sns.cubehelix_palette(
            start=2.0, rot=0, light=1.0, dark=0.5, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
        pal = matplotlib.cm.ScalarMappable(norm=norm, cmap=colors)
        cmap_list.append(pal)
    return cmap_list
