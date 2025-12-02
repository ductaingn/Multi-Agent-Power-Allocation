from copy import deepcopy
from typing import Tuple

import attrs

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Independent, Normal

import numpy as np

from multi_agent_power_allocation.algorithms.low_level.low_level_algorithm import (
    LowLevelAlgorithm,
)
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic
from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)


@attrs.define
class SAC(LowLevelAlgorithm):
    actor: SACPAACtor
    actor_optim: optim.Optimizer
    critic: SACPACritic
    critic_optim: optim.Optimizer
    critic2: SACPACritic
    critic2_optim: optim.Optimizer
    target_entropy: float
    log_alpha: torch.Tensor
    alpha_optim: optim.Optimizer
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    critic_target: SACPACritic = attrs.field(init=False)
    critic2_target: SACPACritic = attrs.field(init=False)

    @critic_target.default
    def _critic_target_factory(self):
        critic_target = deepcopy(self.critic)
        critic_target.eval()
        return critic_target

    @critic2_target.default
    def _critic2_target_factory(self):
        critic2_target = deepcopy(self.critic2)
        critic2_target.eval()
        return critic2_target

    def train(self, mode: bool):
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic2.train(mode)

    def _inference_impl(
        self, obs, deterministic: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.actor(obs)
        action_dist = Independent(
            Normal(
                mu,
                std,
            ),
            1,
        )
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
        log_prob = action_dist.log_prob(action).unsqueeze(-1)
        action = torch.tanh(action)
        log_prob = log_prob - torch.log(
            (1 - action.pow(2)) + np.finfo(np.float32).eps.item()
        ).sum(-1, keepdim=True)

        return action, log_prob

    def inference(self, obs, deterministic: bool = False, **kwargs) -> torch.Tensor:
        action, _ = self._inference_impl(obs, deterministic)
        return action

    def _inference_log_prob(
        self, obs, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action, log_prob = self._inference_impl(obs, deterministic)

        return action, log_prob

    def optimize_critic(
        self,
        data: ReplayBufferSamples,
        alpha: torch.Tensor,
        critic: SACPACritic,
        optimizer: optim.Optimizer,
    ):
        with torch.no_grad():
            next_actions, next_log_prob = self._inference_log_prob(
                data.next_observations, False
            )
            next_q_values = torch.cat(
                [
                    self.critic_target(data.next_observations, next_actions),
                    self.critic2_target(data.next_observations, next_actions),
                ],
                dim=1,
            )
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            next_q_values = next_q_values - alpha * next_log_prob
            target_q_values = (
                data.rewards + (1 - data.dones) * self.gamma * next_q_values
            )

        current_q_values = critic(data.observations, data.actions)

        # Compute loss
        critic_loss = 0.5 * F.mse_loss(current_q_values, target_q_values).sum()

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        return critic_loss.cpu().item()

    def soft_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def sync_target_weights(self):
        self.soft_update(self.critic.parameters(), self.critic_target.parameters())
        self.soft_update(self.critic2.parameters(), self.critic2_target.parameters())

    def learn(self, data: ReplayBufferSamples):
        self.actor.train(True)

        # TODO: Update learning rate according to learning rate schedulers

        alpha_losses, alphas = [], []
        actor_losses, critic_losses, critic2_losses = [], [], []

        for gradient_step in range(self.gradient_steps):
            actions, log_prob = self._inference_log_prob(data.observations, False)

            # Update entropy coefficient
            alpha_log_prob = log_prob.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * alpha_log_prob).mean()
            alpha_losses.append(alpha_loss.cpu().item())
            alpha = torch.exp(self.log_alpha.detach())
            alphas.append(alpha.cpu().numpy())

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # Update critics
            critic1_loss = self.optimize_critic(
                data, alpha, self.critic, self.critic_optim
            )
            critic2_loss = self.optimize_critic(
                data, alpha, self.critic2, self.critic2_optim
            )
            critic_losses.append(critic1_loss)
            critic2_losses.append(critic2_loss)

            # Update actor
            q1 = self.critic(data.observations, actions)
            q2 = self.critic2(data.observations, actions)
            actor_loss = (alpha * log_prob - torch.min(q1, q2)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            actor_losses.append(actor_loss.cpu().item())

            self.sync_target_weights()

        self.actor.train(False)

        return (
            np.mean(actor_losses),
            np.mean(critic_losses),
            np.mean(critic2_losses),
            np.mean(alpha_losses),
            np.mean(alphas),
        )
