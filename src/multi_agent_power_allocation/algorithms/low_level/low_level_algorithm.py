from abc import ABC
from typing import Any, Dict, Tuple

import attrs
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

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

    def state_dict(self) -> Dict[str, Any]:
        """
        Collect everything needed to restore the learner: network/optimizer state dicts,
        tensors (e.g. SAC `log_alpha`) and plain scalars (e.g. DQN `exploration_rate`).
        Algorithms with other kinds of state (e.g. RAQL tables) extend this.
        """
        state = {}
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if isinstance(value, (nn.Module, Optimizer)):
                state[field.name] = value.state_dict()
            elif isinstance(value, torch.Tensor):
                state[field.name] = value.detach().cpu().clone()
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                state[field.name] = value
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name, saved in state.items():
            value = getattr(self, name)
            if isinstance(value, (nn.Module, Optimizer)):
                value.load_state_dict(saved)
            elif isinstance(value, torch.Tensor):
                with torch.no_grad():
                    value.copy_(saved.to(value.device))
            else:
                setattr(self, name, saved)

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
