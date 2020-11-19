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
"""Default hparams for a model."""
from typing import Any, Dict, Optional, Text

import ml_collections

ConfigDict = ml_collections.ConfigDict


def get_hparams(override_dict: Optional[Dict[Text, Any]] = None) -> ConfigDict:
    hp = ConfigDict()
    hp.node_size = 50  # [20, 30, 40, 50]
    hp.edge_size = 20  # [8, 20, 32]
    hp.global_size = 100  # [50, 75, 100, 125]
    hp.n_layers = 3  # [1, 2, 3, 4, 5]
    hp.task_type = 'benzene'
    hp.block_type = 'gcn'
    hp.learning_rate = 3e-4
    hp.epochs = 300
    hp.batch_size = 256
    hp.random_seed = 42  # list(range(42, 67))
    if override_dict:
        hp.update(override_dict)
    return hp
