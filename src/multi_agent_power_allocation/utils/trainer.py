"""
Trainer
"""

import os
from typing import Dict, Literal, Tuple, List
import pickle
from copy import deepcopy
import json
import yaml
import attrs

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic
from multi_agent_power_allocation.wireless_environment.env import WirelessEnvironment
from multi_agent_power_allocation.wireless_environment.env.wrapper import SyncVecEnv
from multi_agent_power_allocation.algorithms.sac import SAC
from multi_agent_power_allocation.algorithms.raql import RAQL
from multi_agent_power_allocation.algorithms.random import Random
from multi_agent_power_allocation.utils.logger import Logger
from multi_agent_power_allocation.utils.replay_buffer import ReplayBuffer
from multi_agent_power_allocation.utils.multi_agent import (
    MultiAgentTrainer,
    MultiAgentPolicyManager,
)
from multi_agent_power_allocation.algorithms.base_algorithm import Algorithm


def parse_config(path: str) -> Dict:
    """
    Process the default yaml config file into keyword arguments to parse Trainer class

    Parameters:
    ----------
    path : str
        Path to the config yaml file

    Returns:
    -------
    kwargs : Dict
        Keyword arguments
    """
    try:
        with open(path, "rb") as file:
            config: Dict = yaml.safe_load(file)
    except FileExistsError as e:
        print("Error occured when trying to open default config file!")
        print(e)

    model_config: Dict = config.get("model_config")
    env_config: Dict = config.get("env_config")
    wc_cluster_config: Dict = env_config.get("wc_cluster_config")
    num_cluster: int = env_config["num_cluster"]

    parsed_wc_clusters_configs = []
    for i in range(num_cluster):
        h_tilde_path = os.path.join(
            BASE_DIR,
            "data",
            wc_cluster_config["scenario"],
            f"cluster_{i}",
            "h_tilde.pickle",
        )

        positions_path = os.path.join(
            BASE_DIR,
            "data",
            wc_cluster_config["scenario"],
            f"cluster_{i}",
            "positions.json",
        )

        if not os.path.isfile(h_tilde_path):
            raise FileNotFoundError(f"`h_tilde` path is not valid!: {h_tilde_path}")

        if not os.path.isfile(positions_path):
            raise FileNotFoundError(f"`positions` path is not valid!: {positions_path}")

        positions: Dict = json.load(open(positions_path, "rt", encoding="utf-8"))
        parsed_wc_clusters_configs.append(
            {
                "h_tilde": pickle.load(open(h_tilde_path, "rb")),
                "num_devices": wc_cluster_config["num_devices"],
                "AP_position": np.array(positions["AP"]),
                "device_positions": np.array(positions["devices"]),
                "obstacle_positions": np.array(positions["obstacles"]),
                "num_sub_channel": wc_cluster_config["num_sub_channel"],
                "num_beam": wc_cluster_config["num_beam"],
                "n_warm_up_step": config["n_warm_up_step"],
            }
        )

    model_config.update({"num_devices": wc_cluster_config["num_devices"]})
    env_config.pop("wc_cluster_config")
    env_config.update({"n_warm_up_step": config.get("n_warm_up_step")})
    env_config.update({"wc_clusters_configs": parsed_wc_clusters_configs})

    algorithm_list: List[str] = env_config.pop("algorithm_list")
    if len(algorithm_list) != num_cluster:
        raise ValueError(
            f"""
            Number of algorithm must match the number of clusters!
            Number of cluster: {num_cluster}
            Algorithm list: {algorithm_list}
            """
        )
    parsed_algorithms: List[Algorithm] = []
    for algorithm in algorithm_list:
        try:
            parsed_algorithms.append(Algorithm(algorithm))
        except ValueError as exc:
            raise ValueError(
                f"Unknown algorithm {algorithm}, valid ones: {[a.value for a in Algorithm]}"
            ) from exc

    env_config.update({"algorithm_list": parsed_algorithms})

    return config


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
    num_env: int = attrs.field(default=1)
    device: str = attrs.field(init=False)

    @env_config.validator
    def _check_env_config(self, attribute, value: Dict):
        must_have_keys = [
            "num_cluster",
            "wc_clusters_configs",
            "max_num_step",
            "algorithm_list",
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_env(self):
        return WirelessEnvironment(**deepcopy(self.env_config))

    def get_policies(self) -> Tuple[List, List, List]:
        env = self.get_env()

        algorithm_list = self.env_config["algorithm_list"]
        policies = []
        schedulers = []
        replay_buffers = []

        for agent, algorithm in zip(env.agents, algorithm_list):
            obs_space = env.observation_space(agent)
            action_space = env.action_space(agent)

            if algorithm == Algorithm.SACPA or algorithm == Algorithm.SACPF:
                actor = SACPAACtor(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                ).to(self.device)
                actor_optim = Adam(actor.parameters(), lr=self.SAC_config["lr"])
                critic1 = SACPACritic(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                ).to(self.device)
                critic1_optim = Adam(critic1.parameters(), lr=self.SAC_config["lr"])
                critic2 = SACPACritic(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                ).to(self.device)
                critic2_optim = Adam(critic2.parameters(), lr=self.SAC_config["lr"])

                # auto entropy tuning setup
                target_entropy = float(-np.prod(action_space.shape))
                log_alpha = (
                    torch.log(torch.ones(1) * 1.0).requires_grad_(True).to(self.device)
                )
                alpha_optim = Adam([log_alpha], lr=self.SAC_config["lr"])

                schedulers += [
                    CosineAnnealingLR(
                        actor_optim, T_max=self.env_config["max_num_step"]
                    ),
                    CosineAnnealingLR(
                        critic1_optim, T_max=self.env_config["max_num_step"]
                    ),
                    CosineAnnealingLR(
                        critic2_optim, T_max=self.env_config["max_num_step"]
                    ),
                    CosineAnnealingLR(
                        alpha_optim, T_max=self.env_config["max_num_step"]
                    ),
                ]

                policy = SAC(
                    actor,
                    actor_optim,
                    critic1,
                    critic1_optim,
                    critic2,
                    critic2_optim,
                    target_entropy,
                    log_alpha,
                    alpha_optim,
                )
            elif algorithm == Algorithm.RAQL:
                policy = RAQL(action_space)
            else:
                policy = Random(action_space)

            policies.append(policy)

            replay_buffer = ReplayBuffer(
                20_000,
                obs_space,
                action_space,
                n_envs=self.num_env,
            )
            replay_buffers.append(replay_buffer)

        return policies, replay_buffers

    def build(self, run_name: str) -> MultiAgentTrainer:
        # ======== environment setup =========
        train_envs = SyncVecEnv([self.get_env for _ in range(self.num_env)])

        # ======== agent setup =========
        policies, replay_buffers = self.get_policies()
        multi_agent_manager = MultiAgentPolicyManager(policies, train_envs)

        # ======== logging setup =========
        logger = Logger(
            project=self.wandb_config["project"],
            config={"env_config": self.env_config},
            name=run_name,
        )

        for agent_id in range(self.num_agent):
            actor = multi_agent_manager.policies[agent_id].actor
            if isinstance(actor, torch.nn.Module):
                logger.wandb_run.watch(
                    actor,
                    log="gradients",
                    log_freq=100,
                    idx=agent_id,
                )

        # ======== trainer setup ========
        trainer = MultiAgentTrainer(
            multi_agent_manager, replay_buffers, self.max_num_step, 256, logger
        )

        return trainer

    def train(self, run_name: str) -> Dict[str, float | str]:
        trainer = self.build(run_name)

        # torch.autograd.set_detect_anomaly(True)

        trainer.train()

        trainer.logger.wandb_run.finish(exit_code=0)
