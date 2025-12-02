from copy import deepcopy
from typing import Callable

import attrs

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from gymnasium.spaces import Discrete

from multi_agent_power_allocation.algorithms.low_level.low_level_algorithm import (
    LowLevelAlgorithm,
    DummyActor,
)
from multi_agent_power_allocation.nn.module import DQNQNetwork as QNetwork
from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)


LAMBDA = 0.995  # Similar to RAQL


def log3(x):
    return np.log(x) / np.log(3)


def get_exploration_schedule(_lambda: float = LAMBDA):
    def linear_decay(value: float):
        return value * _lambda

    return linear_decay


@attrs.define
class DQN(LowLevelAlgorithm):
    q_net: QNetwork
    q_net_optim: optim.Optimizer
    action_space: Discrete
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 1.0
    max_grad_norm: float = 10
    exploration_schedule: Callable[[float], float] = attrs.field(
        init=False, factory=get_exploration_schedule
    )
    actor: DummyActor = attrs.field(init=False, factory=DummyActor)
    exploration_rate: float = attrs.field(init=False, default=1.0)
    q_net_target: QNetwork = attrs.field(init=False)

    @q_net_target.default
    def _q_net_target_factory(self):
        q_net_target = deepcopy(self.q_net)
        q_net_target.eval()
        return q_net_target

    def train(self, mode: bool):
        self.q_net.train(mode)
        self.q_net_target.train(mode)

    def inference(self, obs, deterministic: bool = False, **kwargs):
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = torch.tensor(
                np.array([self.action_space.sample() for _ in range(obs.shape[0])])
            )
        else:
            q_values = self.q_net(obs)
            actions = q_values.argmax(dim=1).reshape(-1)

        return actions

    def soft_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def sync_target_weights(self):
        self.soft_update(self.q_net.parameters(), self.q_net_target.parameters())

    def learn(self, data: ReplayBufferSamples):
        """
        Adapt from stablebaselines3
        """
        self.q_net.train(True)

        losses = []

        for gradient_step in range(self.gradient_steps):
            with torch.no_grad():
                next_q_values = self.q_net_target(data.next_observations)

                next_q_values, _ = next_q_values.max(dim=1)

                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    data.rewards + (1 - data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(data.observations)

            current_q_values = torch.gather(
                current_q_values, dim=1, index=data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            self.q_net_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_net_optim.step()

            losses.append(loss.cpu().item())

            self.sync_target_weights()

        self.q_net.train(False)

        return np.mean(losses), 0.0, 0.0, 0.0, 0.0
