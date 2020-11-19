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
"""Code to identify fragments in molecules."""
import abc
import functools
import itertools
from typing import Any, Iterator, List, Optional, Sequence, Text, Tuple

import numpy as np
from rdkit import Chem


def _sparse_to_dense(sparse_indices: List[int], max_value) -> np.ndarray:
    dense_array = np.zeros([max_value], dtype=np.bool)
    dense_array[sparse_indices] = 1
    return dense_array


def _nonnull_powerset(iterable) -> Iterator[Tuple[Any]]:
    """Returns powerset of iterable, minus the empty set."""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1))


def _reduce_logical_or(arrays: Sequence[np.ndarray], array_size) -> np.ndarray:
    initial_value = np.zeros(array_size, dtype=np.bool)
    return functools.reduce(np.logical_or, arrays, initial_value)


def _reduce_logical_and(
        arrays: Sequence[np.ndarray], array_size) -> np.ndarray:
    initial_value = np.ones(array_size, dtype=np.bool)
    return functools.reduce(np.logical_and, arrays, initial_value)


class AbstractFragmentRule(metaclass=abc.ABCMeta):
    """Class encapsulating fragment identification logic."""

    @abc.abstractmethod
    def match(self, mol: Chem.Mol) -> List[np.ndarray]:
        """Return a list of all possible masking arrays.

        For example, if the matching rule is "Nitrogen", and the molecule is NCCN:
        >>> BasicFragmentRule('nitrogen', -'N').match(Chem.MolFromSmiles('NCCN')
        ...
        [np.ndarray([1, 0, 0, 0]),
         np.ndarray([1, 0, 0, 1]),
         np.ndarray([0, 0, 0, 1])]

        This is because there are many possible correct answers when trying to
        answer the question, "Why did you think there was a N in this molecule?".

        Args:
          mol: Molecule to test for fragments.

        Returns: List of legal fragment masks.
        """


class BasicFragmentRule(AbstractFragmentRule):
    """Single fragment identifier."""

    def __init__(self, label: Text, smarts: Text,
                 description: Optional[Text] = None):
        self.label = label
        self.smarts = smarts
        self.pattern = Chem.MolFromSmarts(smarts)
        self.description = description
        # Update properties, sometimes rings or valence info is not set.
        self.pattern.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(self.pattern)

    def __len__(self):
        """Gets size of matched fragment."""
        return self.pattern.GetNumAtoms()

    def substruct_matches(self, mol: Chem.Mol) -> List[List[int]]:
        """Gets all matching substructs as a list of sparse indice lists."""
        return list(map(list, mol.GetSubstructMatches(self.pattern)))

    def match(self, mol: Chem.Mol) -> List[np.ndarray]:
        matches = self.substruct_matches(mol)
        mol_size = mol.GetNumAtoms()
        dense_matches = [_sparse_to_dense(index_list, mol_size)
                         for index_list in matches]
        all_matches = [_reduce_logical_or(match_set, mol_size)
                       for match_set in _nonnull_powerset(dense_matches)]
        return all_matches


class CompositeRule(AbstractFragmentRule):
    """Multiple fragment identifier."""

    def __init__(self, rule_type: Text,
                 fragment_logics: List[AbstractFragmentRule]):
        self.rule_type = rule_type
        self.fragment_logics = fragment_logics

    @classmethod
    def or_(cls, fragment_logics):
        return cls('OR', fragment_logics)

    @classmethod
    def and_(cls, fragment_logics):
        return cls('AND', fragment_logics)

    @property
    def label(self):
        rule_str = {'OR': ' | ', 'AND': ' & '}[self.rule_type]
        labels = [rule.label for rule in self.fragment_logics]
        return f'({rule_str.join(labels)})'

    def match(self, mol: Chem.Mol) -> List[np.ndarray]:
        subrule_matches = [logic.match(mol) for logic in self.fragment_logics]
        mol_size = mol.GetNumAtoms()
        composite_matches = []
        for combination in itertools.product(*subrule_matches):
            if self.rule_type == 'OR':
                for match_subset in _nonnull_powerset(combination):
                    composite_matches.append(
                        _reduce_logical_or(
                            match_subset, mol_size))
            elif self.rule_type == 'AND':
                composite_matches.append(
                    _reduce_logical_or(
                        combination, mol_size))
        return composite_matches
