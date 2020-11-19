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
from typing import (Any, Callable, List, MutableMapping, Optional, Text, Tuple,
                    Union)

import graph_nets
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import \
    AllChem  # pylint: disable=unused-import  Required for _CalcCrippenContribs

from graph_attribution import attribution_metrics as att_metrics
from graph_attribution import graphs as graph_utils
from graph_attribution import templates

Mol = Chem.Mol
GraphsTuple = graph_nets.graphs.GraphsTuple

CRIPPEN_PATH = 'data/crippen/crippen_subgraph_contributions.csv'


def _get_mol_sender_receivers(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """Get connectivity (messages) info for a data_dict."""
    senders, receivers = [], []
    for bond in mol.GetBonds():
        id1 = bond.GetBeginAtom().GetIdx()
        id2 = bond.GetEndAtom().GetIdx()
        senders.extend([id1, id2])
        receivers.extend([id2, id1])
    return np.array(senders), np.array(receivers)


def _make_attribution_from_nodes(mol: Mol, nodes: np.ndarray,
                                 global_vec: np.ndarray) -> GraphsTuple:
    """Makes an attribution from node information."""
    senders, receivers = _get_mol_sender_receivers(mol)
    data_dict = {
        'nodes': nodes.astype(np.float32),
        'senders': senders,
        'receivers': receivers,
        'globals': global_vec.astype(np.float32)
    }
    return graph_nets.utils_np.data_dicts_to_graphs_tuple([data_dict])


def get_crippen_features(
        mol: Chem.Mol) -> Tuple[np.ndarray, List[Any], List[Any]]:
    """Calculate crippen features."""
    n_atoms = mol.GetNumAtoms()
    n_atoms_with_h = Chem.AddHs(mol).GetNumAtoms()
    if n_atoms != n_atoms_with_h:
        raise ValueError('Your molecule might not have explicit hydrogens!')
    atom_types = [None] * n_atoms
    atom_labels = [None] * n_atoms
    contribs = Chem.rdMolDescriptors._CalcCrippenContribs(  # pylint: disable=protected-access
        mol,
        force=True,
        atomTypes=atom_types,
        atomTypeLabels=atom_labels)
    logp, _ = zip(*contribs)  # Second component is molecular reflecivity.
    return np.array(logp), atom_types, atom_labels


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


class CrippenLogPDataset(templates.AttributionDataset):
    """CrippenLogP dataset.

    Crippen's model for LogP (Octanol/Water Partition Coefficient) considers
    LogP as a weighted sum of atom's contributions. Atom's contributions are
    assigned based on their local graph neighborhood, they are classified and
    based on their label given a score. For actual LogP, there are better models,
    this is only used as a real-world example for a synthetic task.
    The equation for the score is:

      CrippenLogP = sum_i  w|sub_graph(atom_i)

    The attributions are w.

    Based on "Prediction of Physicochemical Parameters by
    Atomic Contributions" (https://pubs.acs.org/doi/10.1021/ci990307l).
    """

    def __init__(self, load_data: bool = False,
                 data_path: Optional[Text] = None):
        """CrippenLogPTask.

        Args:
          load_data: Bool if to add additional data related to atom labels.
          data_path: If loading data, file path for the file. If None, will load
            froma default path.
        """
        self.data = None
        if load_data:
            if data_path is None:
                self.data = pd.read_csv(CRIPPEN_PATH)

    @property
    def name(self) -> Text:
        return 'CrippenLogP'

    def get_true_predictions(self, mols: List[Chem.Mol]) -> np.ndarray:
        """Gets Crippen values."""
        values = [sum(get_crippen_features(mol)[0]) for mol in mols]
        return np.array(values).reshape(-1, 1)

    def get_true_attributions(self, mols: List[Chem.Mol]) -> List[GraphsTuple]:
        """Gets crippen values for each molecule as a GraphsTuple."""
        graph_list = []
        for mol in mols:
            logp = np.array(get_crippen_features(mol)[0])
            senders, receivers = _get_mol_sender_receivers(mol)
            data_dict = {
                'nodes': logp,
                'edges': None,
                'senders': senders,
                'receivers': receivers
            }
            graph_list.append(
                graph_nets.utils_np.data_dicts_to_graphs_tuple([data_dict]))
        return graph_list

    def get_true_predictions_attributions(
            self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, List[GraphsTuple]]:
        """Gets true predictions and attributions for a list of molecules."""
        return self.get_true_predictions(
            mols), self.get_true_attributions(mols)


class FragmentLogicDataset(templates.AttributionDataset):
    """Base Fragment logic generator.

    General purpose fragment identification dataset builder.
    Given a fragment identification rule it will setup an attribution task for
    binary classification if a graph obeys a rule or not.
    Supports complex logics like &, | and multi-fragments.
    """

    def __init__(self, rule: 'fragment_identifier.AbstractFragmentRule'):
        """Constructor for base FragmentLogicTask."""
        self.rule = rule

    @property
    def name(self) -> Text:
        return self.rule.label

    def get_true_predictions(self, x: List[Mol]) -> np.ndarray:
        binary_matches = [bool(self.rule.match(m)) for m in x]
        return np.array(binary_matches, dtype=np.float32).reshape(-1, 1)

    def get_true_attributions(self, mols: List[Mol]) -> List[GraphsTuple]:
        """Gets fragments matches and converts them to multi-truth attributions."""
        att = []
        for mol in mols:
            n_atoms = mol.GetNumAtoms()
            matches = self.rule.match(mol)
            if matches:
                nodes = np.array(matches).T.astype(np.float32)
            else:
                nodes = np.zeros((n_atoms, 1))
            global_vec = np.array(sum(nodes) > 0)
            att.append(_make_attribution_from_nodes(mol, nodes, global_vec))

        return att
