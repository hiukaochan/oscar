# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Inference-only config builder for the OSCAR public release.

Pares down the training-side config tree to the minimum required to
instantiate ``CosmosI2VControlModel`` and run inference. Training-side
register functions (optimizer/scheduler/ema/callbacks) are deleted from
this release; their defaults-list entries are removed here.
"""

from typing import Any, List

import attrs

from worldsim._ext.imaginaire import config
from worldsim._ext.imaginaire.trainer import ImaginaireTrainer as Trainer
from worldsim._ext.imaginaire.utils.config_helper import import_all_modules_from_package

from worldsim._src.configs.common.defaults.checkpoint import register_checkpoint
from worldsim._src.configs.common.defaults.ckpt_type import register_ckpt_type
from worldsim._src.configs.common.defaults.dataloader import register_training_and_val_data
from worldsim._src.configs.common.defaults.tokenizer import register_tokenizer
from worldsim._src.configs.common.defaults.conditioner_i2v import register_conditioner

from worldsim._src.configs.agibot_control.defaults.model import register_model
from worldsim._src.configs.agibot_control.defaults.net import register_net
from worldsim._src.configs.agibot_control.defaults.dataloader import agibot_register_dataloader
from worldsim._src.configs.agibot_control.defaults.conditioner import register_conditioner as register_custom_conditioner


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"model": "ddp"},
            {"net": None},
            {"conditioner": "i2v_conditioner"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "local"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "oscar_public"
    c.job.group = "inference"
    c.job.name = "inference_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.callbacks = None

    register_training_and_val_data()
    agibot_register_dataloader()
    register_custom_conditioner()
    register_model()
    register_net()
    register_conditioner()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    import_all_modules_from_package(
        "worldsim._src.configs.agibot_control.experiment", reload=True,
    )
    return c
