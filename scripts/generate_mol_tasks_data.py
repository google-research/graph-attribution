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
"""Generate files for molecular graph attribution tasks."""

import tqdm
from absl import app

from graph_attribution import datasets, experiments, featurization
from graph_attribution import graphs as graph_utils
from graph_attribution import tasks, training

AUG_FRACTION = 0.2
RANDOM_SEED = 42


def generate_and_write_mol_dataset(task_type: tasks.Task):
    """Generate all variables and optionally files needed for a molecular task."""
    task = tasks.get_task(task_type)
    use_h = task_type == tasks.Task.crippen
    use_data_aug = isinstance(
        task.task_type,
        tasks.BinaryClassificationTaskType)
    # Generate filenames
    fnames = datasets.get_default_experiment_filenames(task_type, 'smiles')

    # Load data and prepare featurizers.
    df = datasets.load_df_from_csv(fnames['data'])
    def smi_to_mol(s): return featurization.smiles_to_mol(
        s, infer_hydrogens=use_h)
    df['mol'] = df['smiles'].apply(smi_to_mol)
    tensorizer = featurization.MolTensorizer(preprocess_fn=smi_to_mol)
    mols = df['mol'].tolist()

    # Inputs, labels and attributions.
    x = featurization.smiles_to_graphs_tuple(df['smiles'].tolist(), tensorizer)
    y = task.get_true_predictions(mols)
    att = task.get_true_attributions(mols)

    # Null vectors for Integrated Gradients and data augmentation.
    node_null, edge_null = tensorizer.get_null_vectors()

    train_index, _ = datasets.load_train_test_indices(fnames['splits'])

    if use_data_aug:
        experiments.set_seed(RANDOM_SEED)
        x_aug, y_aug = training.augment_binary_task(
            graph_utils.get_graphs_tf(x, train_index),
            y[train_index],
            node_null,
            edge_null,
            fraction=AUG_FRACTION)

    # Save files
    datasets.save_graphtuples(fnames['att'], att)
    datasets.save_npz(fnames['y'], **{'y': y})
    datasets.save_graphtuples(fnames['x'], [x])
    datasets.save_npz(fnames['null'], **{'node': node_null, 'edge': edge_null})
    if use_data_aug:
        datasets.save_graphtuples(fnames['x_aug'], [x_aug])
        datasets.save_npz(fnames['y_aug'], **{'y_aug': y_aug})


def main(_):

    for task_type in tqdm.tqdm(tasks.MOL_TASKS):
        print(f'Generating {task_type.name}')
        generate_and_write_mol_dataset(task_type)

        # Attempt to load data as a check that this will work later.
        _ = experiments.get_experiment_setup(task_type, 'gcn')


if __name__ == '__main__':
    app.run(main)
