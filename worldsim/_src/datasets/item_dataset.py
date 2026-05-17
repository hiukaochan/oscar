# SPDX-License-Identifier: Apache-2.0
"""Minimal stub of ``worldsim._src.datasets.item_dataset``.

The public inference release ships only the symbols needed by inference code
paths (specifically: :class:`ItemDatasetConfig`, which is referenced by the
conditioner module to look up cached negative-prompt embeddings).
"""

import dataclasses


@dataclasses.dataclass
class ItemDatasetConfig:
    path: str
    length: int
