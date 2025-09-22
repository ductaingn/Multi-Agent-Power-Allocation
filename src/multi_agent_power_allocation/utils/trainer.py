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

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import SACPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer, BaseTrainer

from pettingzoo.utils.conversions import parallel_to_aec
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic
from multi_agent_power_allocation.wireless_environment.env import (
    WirelessEnvironmentSACPA,
    WirelessEnvironmentRandom,
)
from multi_agent_power_allocation.utils.collector import Collector
from multi_agent_power_allocation.utils.logger import Logger


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
            raise FileNotFoundError("`h_tilde` path is not valid!")

        if not os.path.isfile(positions_path):
            raise FileNotFoundError("`positions` path is not valid!")

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

    return config


@attrs.define
class Trainer:
    """
    Trainer
    """

    algorithm: Literal["SACPA, SACPF, RAQL, Random"]
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
        must_have_keys = ["num_cluster", "wc_clusters_configs", "max_num_step"]

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

        value.update({"observation_space": self.get_env().observation_space})
        value.update({"action_space": self.get_env().action_space})

    def __attrs_post_init__(self):
        self.env = f"WirelessEnvironment{self.algorithm}-v2"
        self.max_num_step = self.env_config["max_num_step"]
        self.num_agent = self.env_config["num_cluster"]
        self.policies = [f"agent_{i}_policy" for i in range(self.num_agent)]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_env(self):
        env_parallel = None
        if self.algorithm == "SACPA":
            env_parallel = WirelessEnvironmentSACPA(**deepcopy(self.env_config))
        if self.algorithm == "Random":
            env_parallel = WirelessEnvironmentRandom(**deepcopy(self.env_config))
        if env_parallel is not None:
            env_aec = parallel_to_aec(env_parallel)
            return PettingZooEnv(env_aec)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def get_agents(self) -> Tuple[MultiAgentPolicyManager, List]:
        env = self.get_env()

        policies = []
        schedulers = []
        for agent_id, agent in enumerate(env.agents):
            actor = SACPAACtor(**self.model_config).to(self.device)
            actor_optim = Adam(actor.parameters(), lr=self.SAC_config["lr"])
            critic1 = SACPACritic(**self.model_config).to(self.device)
            critic1_optim = Adam(critic1.parameters(), lr=self.SAC_config["lr"])
            critic2 = SACPACritic(**self.model_config).to(self.device)
            critic2_optim = Adam(critic2.parameters(), lr=self.SAC_config["lr"])

            # auto entropy tuning setup
            target_entropy = float(-np.prod(env.env.action_space(agent).shape))
            log_alpha = (
                torch.log(torch.ones(1) * 1.0).requires_grad_(True).to(self.device)
            )
            alpha_optim = Adam([log_alpha], lr=self.SAC_config["lr"])

            schedulers += [
                CosineAnnealingLR(actor_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(critic1_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(critic2_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(alpha_optim, T_max=self.env_config["max_num_step"]),
            ]

            policy = SACPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim,
                alpha=(target_entropy, log_alpha, alpha_optim),
            )

            policies.append(policy)

        policy = MultiAgentPolicyManager(
            policies,
            env,
            # lr_scheduler=MultipleLRSchedulers(*schedulers)
        )

        return policy, env.agents

    def build(self, run_name: str) -> BaseTrainer:
        # ======== environment setup =========
        train_envs = DummyVectorEnv(
            [lambda: self.get_env() for _ in range(self.num_env)]
        )

        # ======== agent setup =========
        policy, agents = self.get_agents()

        # ======== logging setup =========
        logger = Logger(
            train_interval=1,
            test_interval=1,
            update_interval=1,
            project=self.wandb_config["project"],
            config={"algorithm": self.algorithm, "env_config": self.env_config},
            name=run_name,
        )

        logger.wandb_run.watch(
            policy.policies[agents[0]].actor, log="all", log_graph=True, log_freq=100
        )

        def log_params(step):
            logger.wandb_run.log(
                {
                    "Learning rate": policy.policies[
                        agents[0]
                    ].actor_optim.param_groups[0]["lr"]
                },
                step=step,
            )

        # ======== collector setup =========
        train_collector = Collector(
            policy=policy,
            env=train_envs,
            buffer=VectorReplayBuffer(100_000 * self.num_env, buffer_num=self.num_env),
            exploration_noise=True,
        )
        train_collector.load_logger(logger)

        # ======== callback setup ========
        def save_best_fn(policy):
            model_save_path = os.path.join(logger.wandb_run.dir, "model", "policy.pth")
            os.makedirs(os.path.join(logger.wandb_run.dir, "model"), exist_ok=True)
            torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

        def train_fn(epoch, env_step):
            log_params(env_step)

        # ======== trainer setup ========
        trainer = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=1,
            step_per_epoch=10_000,
            step_per_collect=1,
            episode_per_test=1,
            batch_size=256,
            update_per_step=1,
            save_best_fn=save_best_fn,
            train_fn=train_fn,
            logger=logger,
            test_in_train=False,
        )

        return trainer

    def train(self, run_name: str) -> Dict[str, float | str]:
        trainer = self.build(run_name)

        result = trainer.run()

        trainer.logger.wandb_run.finish(exit_code=0)

        return result
