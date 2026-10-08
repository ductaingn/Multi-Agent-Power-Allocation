"""
Trainer
"""

import os
from typing import Dict, List, Optional
from copy import deepcopy
import attrs

import torch

from multi_agent_power_allocation.wireless_environment.env import WirelessEnvironment
from multi_agent_power_allocation.wireless_environment.env.wrapper import SyncVecEnv
from multi_agent_power_allocation.utils.logger import Logger
from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBuffer,
)
from multi_agent_power_allocation.utils.multi_agent import (
    MultiAgentTrainer,
    MultiAgentPolicyManager,
)
from multi_agent_power_allocation.algorithms.high_level import Algorithm
from multi_agent_power_allocation.utils.checkpoint import (
    Checkpointer,
    resolve_checkpoint_config,
    slugify,
)


@attrs.define
class Trainer:
    """
    Trainer
    """

    env: str = attrs.field(init=False)
    env_config: Dict = attrs.field()
    num_agent: int = attrs.field(init=False)
    model_config: Dict = attrs.field()
    max_num_step: int = attrs.field(init=False)
    n_warm_up_step: int = attrs.field()
    policies: Dict = attrs.field(init=False)
    wandb_config: Dict = attrs.field()
    SAC_config: Dict = attrs.field()
    device: str = attrs.field()
    num_env: int = attrs.field()
    checkpoint_config: Optional[Dict] = attrs.field(default=None)
    # Plain YAML config, logged to WandB and stored next to checkpoints
    raw_config: Optional[Dict] = attrs.field(default=None)

    @env_config.validator
    def _check_env_config(self, attribute, value: Dict):
        must_have_keys = [
            "num_cluster",
            "wc_clusters_configs",
            "max_num_step",
            "algorithm_mapping",
        ]

        for key in must_have_keys:
            if key not in value:
                raise ValueError(f"env_config must contain {key}!")

        wc_clusters_configs: List[Dict] = value.get("wc_clusters_configs")
        if not isinstance(wc_clusters_configs, List):
            raise ValueError("wc_cluster_config must be a list!")

        must_have_wccc_keys = [
            "h_tilde",
            "num_devices",
            "AP_position",
            "device_positions",
            "obstacle_positions",
            "num_sub_channel",
            "num_beam",
        ]

        for config in wc_clusters_configs:
            for key in must_have_wccc_keys:
                if key not in config:
                    raise ValueError(f"wc_cluster_config must contain {key}!")

    @model_config.validator
    def _check_model_config(self, attribute, value: Dict):
        must_have_keys = ["latent_dim", "num_devices"]

        for key in must_have_keys:
            if key not in value:
                raise ValueError(f"model_config must contain {key}!")

    def __attrs_post_init__(self):
        self.max_num_step = self.env_config["max_num_step"]
        self.num_agent = self.env_config["num_cluster"]
        self.policies = [f"agent_{i}_policy" for i in range(self.num_agent)]

    def get_env(self):
        # Channel realizations are read-only and large (~200MB per cluster for 10 devices):
        # share them across env copies instead of deep-copying them
        memo = {
            id(wc_config["h_tilde"]): wc_config["h_tilde"]
            for wc_config in self.env_config["wc_clusters_configs"]
        }
        return WirelessEnvironment(**deepcopy(self.env_config, memo))

    def get_replay_buffer(self) -> Dict[str, ReplayBuffer]:
        algorithm_mapping: Dict[str, Algorithm] = self.env_config["algorithm_mapping"]
        replay_buffers = {}
        wc_clusters_configs = self.env_config["wc_clusters_configs"]

        for agent_id, algorithm in algorithm_mapping.items():
            wc_config = wc_clusters_configs[int(agent_id)]
            obs_space = algorithm.observation_space(
                wc_config["num_devices"], wc_config["L_max"]
            )
            action_space = algorithm.action_space(wc_config["num_devices"])

            replay_buffer = ReplayBuffer(
                20_000,
                obs_space,
                action_space,
                n_envs=self.num_env,
            )
            replay_buffers.update({agent_id: replay_buffer})

        return replay_buffers

    def build(self, run_name: str) -> MultiAgentTrainer:
        # ======== environment setup =========
        train_envs = SyncVecEnv([self.get_env for _ in range(self.num_env)])

        # ======== agent setup =========
        replay_buffers = self.get_replay_buffer()
        policies: Dict[str, Algorithm] = self.env_config["algorithm_mapping"]
        multi_agent_manager = MultiAgentPolicyManager(policies, train_envs)

        # ======== logging setup =========
        checkpointer_config = resolve_checkpoint_config(self.checkpoint_config)
        history_path = None
        if self.wandb_config.get("local_history", False):
            history_path = os.path.join(
                checkpointer_config["save_dir"], slugify(run_name), "history.npz"
            )
        logger = Logger(
            project=self.wandb_config["project"],
            entity=self.wandb_config.get("entity"),
            config=self._wandb_config(policies),
            name=run_name,
            history_path=history_path,
        )

        for agent_id in policies.keys():
            actor = multi_agent_manager.policies[agent_id].low_level_algorithm.actor
            if isinstance(actor, torch.nn.Module):
                logger.wandb_run.watch(
                    actor,
                    log="gradients",
                    log_freq=100,
                    idx=int(agent_id),
                )

        # ======== trainer setup ========
        checkpointer = Checkpointer(
            policies=policies,
            run_name=run_name,
            config=self.checkpoint_config,
            run_config=self.raw_config,
            wandb_run=logger.wandb_run,
        )

        trainer = MultiAgentTrainer(
            multi_agent_manager,
            replay_buffers,
            self.max_num_step,
            256,
            logger,
            checkpointer=checkpointer,
        )

        return trainer

    def _wandb_config(self, policies: Dict[str, Algorithm]) -> Dict:
        """
        Avoid dumping channel matrices and full module reprs into the WandB config:
        log the YAML config plus the algorithm name of each agent.
        """
        config = dict(self.raw_config or {})
        config.setdefault("env_config", {})
        config["env_config"] = dict(config["env_config"])
        # Kept under the old key so existing analysis code (`.split("(")[0]`) still works
        config["env_config"]["algorithm_mapping"] = {
            agent_id: type(policy).__name__ for agent_id, policy in policies.items()
        }
        return config

    def train(self, run_name: str) -> Dict[str, float | str]:
        trainer = self.build(run_name)

        # torch.autograd.set_detect_anomaly(True)

        trainer.train()
        trainer.logger.save_history()

        trainer.logger.wandb_run.finish(exit_code=0)
