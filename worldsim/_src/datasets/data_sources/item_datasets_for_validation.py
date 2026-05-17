# SPDX-License-Identifier: Apache-2.0
"""Minimal stub of ``worldsim._src.datasets.data_sources.item_datasets_for_validation``.

The public inference release computes text embeddings online via the Cosmos-Reason1
encoder; it does NOT rely on the pre-baked ``neg_string_umt5.pt`` / ``empty_string_umt5.pt``
files. These symbols remain importable so that the conditioner module still
loads, but their use should always go through online text encoding instead.
"""

import os

from worldsim._src.datasets.item_dataset import ItemDatasetConfig

_EMBEDDINGS_DIR = os.environ.get(
    "WORLDSIM_EMBEDDINGS_DIR",
    "/nonexistent-public-release",
)


def get_itemdataset_option_local(name: str) -> ItemDatasetConfig:
    return ITEMDATASET_OPTIONS_LOCAL[name]


ITEMDATASET_OPTIONS_LOCAL = {
    "empty_string_umt5": ItemDatasetConfig(
        path=f"{_EMBEDDINGS_DIR}/empty_string_umt5.pt",
        length=1,
    ),
    "neg_string_umt5": ItemDatasetConfig(
        path=f"{_EMBEDDINGS_DIR}/neg_string_umt5.pt",
        length=1,
    ),
}
