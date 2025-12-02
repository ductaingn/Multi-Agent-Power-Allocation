from typing import Dict

import attrs
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TextColumn,
)

import torch

import numpy as np

from ..wireless_environment.env.wrapper import SyncVecEnv
from .logger import Logger
from ..algorithms.high_level import Algorithm
from ..algorithms.low_level.utils.replay_buffer import ReplayBuffer, ReplayBufferSamples


@attrs.define
class MultiAgentPolicyManager:
    policies: Dict[str, Algorithm]
    envs: SyncVecEnv

    def learn(self, data: Dict[str, ReplayBufferSamples]):
        res = {}
        for agent_id, policy in self.policies.items():
            agent_data = data[agent_id]
            actor_loss, critic_loss, critic2_loss, alpha_loss, alpha = policy.learn(
                agent_data
            )
            res.update(
                {
                    f"update/ {agent_id}/ actor_loss": actor_loss,
                    f"update/ {agent_id}/ critic1_loss": critic_loss,
                    f"update/ {agent_id}/ critic2_loss": critic2_loss,
                    f"update/ {agent_id}/ alpha_loss": alpha_loss,
                    f"update/ {agent_id}/ alpha": alpha,
                }
            )

        return res


@attrs.define
class MultiAgentTrainer:
    multi_agent_manager: MultiAgentPolicyManager
    replay_buffer: Dict[str, ReplayBuffer] = attrs.field()
    n_step_per_env: int
    batch_size: int
    logger: Logger
    learning_start: int = 200
    num_timesteps: int = attrs.field(default=0, init=False)
    _last_obs: Dict[str, torch.Tensor] = attrs.field(init=False)

    @_last_obs.default
    def _last_obs_factory(self):
        obs, _ = self.multi_agent_manager.envs.reset()
        return {agent_id: np.array([obs[agent_id]]) for agent_id in obs}

    def collect_data(self):
        # Sample action
        actions = {}
        for agent_id, policy in self.multi_agent_manager.policies.items():
            policy.low_level_algorithm.actor.train(False)

            if self.num_timesteps < self.learning_start:
                agent_actions = np.array(
                    [self.multi_agent_manager.envs.action_spaces[agent_id].sample()]
                )
            else:
                agent_actions = policy.low_level_algorithm.inference(
                    self._last_obs[agent_id], deterministic=False
                )
                agent_actions = agent_actions.detach().cpu().numpy()

            actions.update({agent_id: agent_actions})

        next_observations, rewards, terminations, truncations, infos = (
            self.multi_agent_manager.envs.step(actions)
        )

        # TODO: check terminations conditions
        # Update replay buffer
        for agent_id in self.multi_agent_manager.policies.keys():
            self.replay_buffer[agent_id].add(
                self._last_obs[agent_id],
                next_observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                done=terminations[agent_id],
                infos=infos[agent_id],
            )

        return infos

    def sample_data(self):
        data = {
            agent_id: self.replay_buffer[agent_id].sample(self.batch_size)
            for agent_id in self.multi_agent_manager.policies.keys()
        }

        return data

    def train(self):
        # Create a nice Rich progress bar
        progress = Progress(
            TextColumn("[bold blue]Training [/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        # Add progress tracking task
        task = progress.add_task(
            "train",
            total=self.n_step_per_env,
            step=self.num_timesteps,
        )

        with progress:
            while self.num_timesteps < self.n_step_per_env:
                infos = self.collect_data()

                log_data = {"clusters_data": infos}

                # TODO: check terminations conditions:
                data = self.sample_data()
                if self.num_timesteps >= self.learning_start:
                    train_results = self.multi_agent_manager.learn(data)
                    log_data.update({"update_data": train_results})

                self.logger.write(self.num_timesteps, log_data)

                self.num_timesteps += 1

                progress.update(
                    task,
                    advance=1,
                    step=self.num_timesteps,
                )
