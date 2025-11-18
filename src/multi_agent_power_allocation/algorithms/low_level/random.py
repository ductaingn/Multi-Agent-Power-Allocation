import attrs

import torch

import gymnasium as gym

from multi_agent_power_allocation.algorithms.low_level.low_level_algorithm import (
    LowLevelAlgorithm,
    DummyActor,
)
from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)


@attrs.define
class Random(LowLevelAlgorithm):
    action_space: gym.spaces.Space
    actor: DummyActor = attrs.field(init=False, default=DummyActor())

    def inference(self, obs, **kwargs):
        batch_size = obs.shape[0]
        return torch.rand((batch_size,) + self.action_space.shape)

    def learn(self, data: ReplayBufferSamples):
        return [0.0] * 5
