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
"""Methods for turning molecules into GraphsTuples."""

from typing import Any, Dict, List, Text, Tuple

import graph_nets
import more_itertools
import numpy as np
import rdkit.Chem
import sonnet as snt
import tensorflow as tf
from rdkit import Chem

from graph_attribution import datasets
from graph_attribution import graphnet_models as models
from graph_attribution import graphnet_techniques as techniques
from graph_attribution import graphs as graph_utils
from graph_attribution import tasks, templates

# Typing alias.
GraphsTuple = graph_nets.graphs.GraphsTuple

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
]
NULL_MOLECULE_SMILES = '[*]~[*]'
BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'UNSPECIFIED']


def smiles_to_mol(smiles, infer_hydrogens=False):
    """Basic smiles to RDkit mol."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, 'Could not parse smiles {}'.format(smiles)
    if infer_hydrogens:
        mol = Chem.AddHs(mol)
    return mol


def _onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class MolTensorizer(object):
    """MolTensorizer: Convert data into molecular tensors.

    Utility object to preprocess a dataset, compute feature values, learn
    their range and convert molecules into tensors.
    """

    def __init__(self, preprocess_fn=smiles_to_mol):
        self.preprocess_fn = preprocess_fn

    def get_null_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets atom and bond featurized vectors for unspecified molecule."""
        null_mol = smiles_to_mol(NULL_MOLECULE_SMILES)
        null_atom = null_mol.GetAtomWithIdx(0)
        null_bond = null_mol.GetBondWithIdx(0)
        null_atomvec = self.atom_features(null_atom)
        null_bondvec = self.bond_features(null_bond)
        return null_atomvec, null_bondvec

    def atom_features(self, atom):
        return np.array(_onehot_encoding_unk(atom.GetSymbol(), ATOM_TYPES))

    def bond_features(self, bond):
        return np.array(_onehot_encoding_unk(
            str(bond.GetBondType()), BOND_TYPES))

    def mol_to_data_dict(self, mol: Chem.Mol) -> Dict[Text, np.ndarray]:
        """Gets data dict from a single mol."""
        nodes = np.array([self.atom_features(atom) for atom in mol.GetAtoms()])
        edges = np.zeros((mol.GetNumBonds() * 2, len(BOND_TYPES)))
        senders = []
        receivers = []
        for index, bond in enumerate(mol.GetBonds()):
            id1 = bond.GetBeginAtom().GetIdx()
            id2 = bond.GetEndAtom().GetIdx()
            bond_arr = self.bond_features(bond)
            edges[index * 2, :] = bond_arr
            edges[index * 2 + 1, :] = bond_arr
            senders.extend([id1, id2])
            receivers.extend([id2, id1])
        data_dict = {
            'nodes': nodes.astype(np.float32),
            'edges': edges.astype(np.float32),
            'globals': np.array([0.], dtype=np.float32),
            'senders': np.array(senders, np.int32),
            'receivers': np.array(receivers, np.int32)
        }
        return data_dict

    def transform_data_dict(self,
                            data: List[Any]) -> List[Dict[Text, np.ndarray]]:
        """Transform to data dicts, useful with graph_nets library."""
        mol_list = [self.preprocess_fn(item) for item in data]
        data_dicts = list(map(self.mol_to_data_dict, mol_list))
        return data_dicts


def smiles_to_graphs_tuple(
        smiles_list: List[Text],
        tensorizer: MolTensorizer) -> GraphsTuple:
    """Converts smiles to graphs tuple."""
    graph_list = tensorizer.transform_data_dict(smiles_list)
    return graph_nets.utils_tf.data_dicts_to_graphs_tuple(graph_list)
