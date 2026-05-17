# SPDX-License-Identifier: Apache-2.0
"""Training/val dataloader registration stub (inference-only release).

The public OSCAR inference release does not ship training dataloaders or mock
data builders. We still register placeholder ``data_train=mock``/``data_val=mock``
nodes so that Hydra's defaults-list resolution can complete -- the inference
entry point bypasses ``dataloader_train`` entirely.
"""

from hydra.core.config_store import ConfigStore


def register_training_and_val_data():
    cs = ConfigStore.instance()
    cs.store(group="data_train", package="dataloader_train", name="mock", node={})
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node={})
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node={})
    cs.store(group="data_val", package="dataloader_val", name="mock", node={})


def register_training_and_val_data_no_cosmos():
    register_training_and_val_data()
