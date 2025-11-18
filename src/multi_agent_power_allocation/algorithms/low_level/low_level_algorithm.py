from abc import ABC
from typing import Tuple

import attrs

import torch
import torch.nn as nn

import numpy as np

from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)


@attrs.define
class DummyActor:
    def train(self, mode: bool = False):
        pass

    def __call__(self, obs, **kwds):
        pass


@attrs.define
class LowLevelAlgorithm(ABC):
    actor: DummyActor | nn.Module

    def inference(self, obs: np.ndarray | torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def learn(
        self, data: ReplayBufferSamples
    ) -> Tuple[float, float, float, float, float]:
        """
        Return
            actor_losses
            critic_losses
            critic2_losses
            alpha_losses
            alphas
        """
        raise NotImplementedError()
