# SPDX-License-Identifier: Apache-2.0
"""Action-condition flavor of Cosmos V2V conditioner.

Adds a single field `action: Optional[Tensor]` (shape (B, 80, 14), float32) on
top of Video2WorldCondition. No broadcast() override is needed — the inherited
Video2WorldCondition.broadcast delegates to T2VCondition.broadcast →
broadcast_condition, which iterates every non-None dataclass field and broadcasts
them as whole tensors. `action` is small and not context-parallel-split, so
auto-forwarding is correct.

The conditioner class itself is a thin subclass that overrides forward() to
return the new dataclass, mirroring Video2WorldConditionerV2's pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from worldsim._src.conditioner.cosmos2_v2v_conditioner import (
    Video2WorldCondition,
    Video2WorldConditioner,
)


@dataclass(frozen=True)
class ActionConditionedVideo2WorldCondition(Video2WorldCondition):
    """Video2WorldCondition + EE-trajectory action.

    action shape: (B, num_frames - 1, 14), float32.
    Layout per frame: [left_arm(6) + gripper_left(1) + right_arm(6) + gripper_right(1)].
    Single-arm episodes have all right-half slots == 0.
    """

    action: Optional[torch.Tensor] = None


class ActionConditionedVideo2WorldConditioner(Video2WorldConditioner):
    """Returns ActionConditionedVideo2WorldCondition; otherwise identical to parent."""

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ActionConditionedVideo2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return ActionConditionedVideo2WorldCondition(**output)
