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

"""Functionality for loading, setting up and saving datasets."""
import os
from typing import Dict, List, Optional, Text, Tuple

import graph_nets
import numpy as np
import pandas as pd

from graph_attribution import graphs as graph_utils
from graph_attribution import tasks

# Typing alias.
GraphsTuple = graph_nets.graphs.GraphsTuple

# Data loading aliases
load_npz = np.load
load_df_from_csv = pd.read_csv
save_npz = np.savez_compressed


DATA_DIR = 'data'

DEFAULT_HPARAM_FLAGFILE = os.path.join(DATA_DIR,
                                       'default_hparams.flagfile')

DEFAULT_FILES = dict([('hparams', 'hparams.flagfile'),
                      ('init_state', 'init_state/checkpoint'),
                      ('splits', '{}_traintest_indices.npz'),
                      ('smiles', '{}_smiles.csv'), ('smarts', '{}_smarts.csv'),
                      ('nx_graph', '{}_nxgraph.pkl'),
                      ('checkpoint', 'checkpoint'),
                      ('saved_model', 'saved_model'), ('losses', 'losses.npz'),
                      ('x', 'x_{}.npz'), ('y', 'y_{}.npz'),
                      ('null_vectors', 'null_vectors.npz'),
                      ('predictions', 'predictions.npz'),
                      ('results', 'aggregate_results.csv'),
                      ('attribution_metrics', '{}_attribution_metrics.npz'),
                      ('attributions', '{}_raw_attribution_datadicts.npz')])


def get_task_dir(t): return os.path.join(DATA_DIR, tasks.Task(t).name)


def get_output_filename(fname_kwarg: Text,
                        work_dir: Text,
                        name: Optional[Text] = None):
    """Get output file name using the default file naming scheme."""
    # Separate attributions because these results will be written per method.
    fname = DEFAULT_FILES[fname_kwarg]
    is_formatable = '{' in fname and '}' in fname
    if is_formatable and name is None:
        raise ValueError(
            f'{fname_kwarg} is formattable, missing name argument.')
    if is_formatable:
        return os.path.join(work_dir, fname.format(name))
    else:
        return os.path.join(work_dir, fname)


def get_default_experiment_filenames(
        task_type: tasks.Task,
        data_label: Optional[Text] = None) -> Dict[Text, Text]:
    """Construct filenames for an experiment."""
    label = task_type.name
    exp_dir = get_task_dir(task_type.name)
    fnames = {
        'splits': get_output_filename('splits', exp_dir, label),
        'x': get_output_filename('x', exp_dir, 'true'),
        'y': get_output_filename('y', exp_dir, 'true'),
        'att': get_output_filename('attributions', exp_dir, 'true'),
        'null': get_output_filename('null_vectors', exp_dir),
        'x_aug': get_output_filename('x', exp_dir, 'aug'),
        'y_aug': get_output_filename('y', exp_dir, 'aug')
    }
    if data_label:
        fnames.update(
            {'data': get_output_filename(data_label, exp_dir, label)})

    return fnames


def load_train_test_indices(filename: Text) -> Tuple[np.ndarray, np.ndarray]:
    """Read a numpy file with train/test indices."""
    data = load_npz(filename)
    return data['train_index'], data['test_index']


def save_graphtuples(filename: Text, graphs: List[GraphsTuple]):
    """Save a list of graphstuples with np.savez_compressed."""
    np_graphs = list(map(graph_utils.cast_to_np, graphs))
    data_dicts = list(
        map(graph_nets.utils_np.graphs_tuple_to_data_dicts, np_graphs))
    np.savez_compressed(filename, datadict_list=data_dicts)


def load_graphstuples(filename: Text) -> List[GraphsTuple]:
    """Load a list of graphstuples with np.load."""
    data = load_npz(filename, allow_pickle=True)['datadict_list']
    return list(map(graph_nets.utils_np.data_dicts_to_graphs_tuple, data))
