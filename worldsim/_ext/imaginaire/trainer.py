# SPDX-License-Identifier: Apache-2.0
"""Inference-only stub.

The public OSCAR release does not ship training. ``ImaginaireTrainer`` is
referenced by the config tree (``c.trainer.type``) for attrs validation; the
inference path (``inference/inference_oscar.py``, ``oscar_diffusers``,
``oscar_diffsynth``) never instantiates it.

For training, port the upstream NVIDIA Cosmos-Predict2.5 trainer at
https://github.com/nvidia-cosmos/cosmos-predict2.5.
"""


class ImaginaireTrainer:
    """Placeholder. See module docstring."""
    pass
