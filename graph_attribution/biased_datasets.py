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
"""Code for biased datasets."""
import os
from typing import Dict, Text, Tuple

import graph_nets
import numpy as np
import pandas as pd

from graph_attibution import featurization, fragment_identifier
from graph_attribution import datasets, tasks

# Typing alias.
GraphsTuple = graph_nets.graphs.GraphsTuple

BIAS_DIR = os.path.join(datasets.DATA_DIR, 'dataset_bias')


def load_molgraph_dataset(exp_dir: Text,
                          dataset_name: Text,
                          use_h: bool = False):
    """loads relevant data for a molecule graph problem."""
    data_file = get_output_filename('smiles', exp_dir, dataset_name)
    splits_file = get_output_filename('splits', exp_dir, dataset_name)

    df = load_df_from_csv(data_file)
    train_index, test_index = load_train_test_indices(splits_file)

    def smi_to_mol(s): return featurization.smiles_to_mol(
        s, infer_hydrogens=use_h)
    df['mol'] = df['smiles'].apply(smi_to_mol)
    mols = df['mol'].tolist()
    tensorizer = featurization.MolTensorizer(preprocess_fn=smi_to_mol)
    x = smiles_to_graphs_tuple(df['smiles'].tolist(), tensorizer)
    return df, x, mols, tensorizer, train_index, test_index


def get_bias_experiment_setup(
    dataset_name: Text, ab_mix: float
) -> Tuple[OrderedDict[Text, ExperimentData], OrderedDict[
        Text, AttributionTask], MethodDict]:
    """Setup experiment data for a bias task."""
    exp_dir = os.path.join(datasets.BIAS_DIR, dataset_name)
    df, x, mols, tensorizer, train_index, test_index = load_molgraph_dataset(
        exp_dir, dataset_name)

    train_index = datasets.interpolate_bias_train_indices(
        df, train_index, ab_mix)
    rules = datasets.setup_bias_rules(exp_dir, dataset_name)
    rule_indices = datasets.setup_bias_test_indices(df, test_index)

    task_dict = collections.OrderedDict()
    exp_dict = collections.OrderedDict()
    for rule_name, rule in rules.items():
        task = templates.AttributionTask(
            tasks.FragmentLogicDataset(rule=rule),
            tasks.BinaryClassificationTaskType(), templates.TargetType.globals)
        task_dict[rule_name] = task
        y = task.get_true_predictions(mols)
        atts = task.get_true_attributions(mols)
        test_indices = rule_indices[rule_name]
        exp_dict[rule_name] = ExperimentData.from_data_and_splits(
            x, y, atts, train_index, test_indices)

    methods = techniques.get_techniques_dict(*tensorizer.get_null_vectors())
    return exp_dict, task_dict, methods


def interpolate_bias_train_indices(df: pd.DataFrame, train_indices: np.ndarray,
                                   ab_mix: float) -> np.ndarray:
    """Interpolate between two sets of positive labels for training."""
    train_df = df.iloc[train_indices]
    neg_indices = train_df[train_df['~(A&B)']].index.tolist()
    a_and_b_indices = train_df[train_df['A&B']].index.tolist()
    a_not_b_indices = train_df[train_df['A&~B']].index.tolist()
    a_and_b_n = int(np.round(sum(train_df['A&B']) * ab_mix))
    a_not_b_n = int(np.round(sum(train_df['A&~B']) * (1.0 - ab_mix)))
    pos_indices = a_and_b_indices[:a_and_b_n] + a_not_b_indices[:a_not_b_n]
    pos_indices = pos_indices[:min(len(pos_indices), len(neg_indices))]
    new_indices = pos_indices + neg_indices
    return new_indices


def setup_bias_rules(
        exp_dir: Text,
        dataset_name: Text) -> Dict[Text, fragment_identifier.AbstractFragmentRule]:
    """Setup fragment identification rules for a biased dataset experiment."""
    frag_file = get_output_filename('smarts', exp_dir, dataset_name)
    frag_df = load_df_from_csv(frag_file).set_index('role')

    rules = {}
    rules['A'] = fragment_identifier.BasicFragmentRule('A',
                                                       frag_df.loc['A'].smarts,
                                                       frag_df.loc['A'].label)
    rules['B'] = fragment_identifier.BasicFragmentRule('B',
                                                       frag_df.loc['B'].smarts,
                                                       frag_df.loc['B'].label)
    rules['(A & B)'] = fragment_identifier.CompositeRule('AND',
                                                         [rules['A'], rules['B']])
    return rules


def setup_bias_test_indices(df: pd.DataFrame,
                            test_index: List[int]) -> Dict[Text, np.ndarray]:
    """Gets test indices for each subgraph task."""
    test_df = df.iloc[test_index]
    negative_indices = test_df[test_df['~(A&B)']].index.tolist()
    only_a_indices = test_df[test_df['A&~B']].index.tolist()
    only_b_indices = test_df[test_df['~A&B']].index.tolist()
    a_and_b_indices = test_df[test_df['A&B']].index.tolist()

    test_indices = {}
    test_indices['A'] = np.array(only_a_indices + negative_indices)
    test_indices['B'] = np.array(only_b_indices + negative_indices)
    test_indices['(A & B)'] = np.array(a_and_b_indices + negative_indices)
    return test_indices
