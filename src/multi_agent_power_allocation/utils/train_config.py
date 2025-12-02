import os
from typing import Dict, List

import pickle
import json
import yaml

import attrs

import numpy as np

import torch
from torch.optim import Adam

# from torch.optim.lr_scheduler import CosineAnnealingLR

from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.algorithms.algorithm_register import Algorithms
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic, DQNQNetwork
from multi_agent_power_allocation.algorithms.low_level import (
    SAC,
    RAQL,
    Random,
    DQN,
)
from multi_agent_power_allocation.algorithms.high_level import Algorithm


@attrs.define
class TrainConfig:
    config_file_path: str
    model_config: Dict = attrs.field(init=False)
    env_config: Dict = attrs.field(init=False)
    num_cluster: int = attrs.field(init=False)
    num_env: int = attrs.field(init=False)
    n_warm_up_step: int = attrs.field(init=False)
    wandb_config: Dict = attrs.field(init=False)
    SAC_config: Dict = attrs.field(init=False)
    device: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        try:
            with open(self.config_file_path, "rb") as file:
                config: Dict = yaml.safe_load(file)
        except FileExistsError as e:
            print("Error occured when trying to open default config file!")
            print(e)

        model_config: Dict = config.get("model_config")
        env_config: Dict = config.get("env_config")
        wc_cluster_config: Dict = env_config.get("wc_cluster_config")
        num_cluster: int = env_config["num_cluster"]

        parsed_wc_clusters_configs = []
        obstacles_positions = []
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
                raise FileNotFoundError(
                    f"`positions` path is not valid!: {positions_path}"
                )

            positions: Dict = json.load(open(positions_path, "rt", encoding="utf-8"))
            for obstacle_position in positions["obstacles"]:
                obstacles_positions.append(obstacle_position)

            parsed_wc_clusters_configs.append(
                {
                    "h_tilde": pickle.load(open(h_tilde_path, "rb")),
                    "num_devices": wc_cluster_config["num_devices"],
                    "AP_position": np.array(positions["AP"]),
                    "device_positions": np.array(positions["devices"]),
                    "num_sub_channel": wc_cluster_config["num_sub_channel"],
                    "num_beam": wc_cluster_config["num_beam"],
                    "L_max": wc_cluster_config["L_max"],
                    "n_warm_up_step": config["n_warm_up_step"],
                    "packet_loss_rate_time_window": wc_cluster_config[
                        "packet_loss_rate_time_window"
                    ],
                }
            )

        # Obstacles should be visible for every clusters
        for parsed_wc_cluster_config in parsed_wc_clusters_configs:
            parsed_wc_cluster_config["obstacle_positions"] = np.stack(
                obstacles_positions
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
        parsed_algorithms_class: List[Algorithms] = []
        for algorithm in algorithm_list:
            parsed_algo = None
            for registered_algo in Algorithms:
                if registered_algo.value.__name__ == algorithm:
                    parsed_algo = registered_algo
                    parsed_algorithms_class.append(parsed_algo)
            if parsed_algo is None:
                raise ValueError(
                    f"Algorithm {algorithm} is not registered, valid ones: {[a.value for a in Algorithms]}"
                )
        env_config.update({"algorithm_list": parsed_algorithms_class})

        self.model_config = model_config
        self.num_env = config.get("num_env")
        self.wandb_config = config.get("wandb_config")
        self.SAC_config = config.get("SAC_config")
        self.n_warm_up_step = config.get("n_warm_up_step")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        algorithm_mapping = self.get_algorithm_mapping(env_config)
        env_config.pop("algorithm_list")
        env_config.update({"algorithm_mapping": algorithm_mapping})
        self.env_config = env_config

    def get_algorithm_mapping(self, env_config: Dict) -> Dict[str, Algorithm]:
        list_algorithm_cls: List[Algorithms] = env_config["algorithm_list"]
        algorithm_mapping = {}
        # schedulers = []

        for agent_id, algorithm_cls in enumerate(list_algorithm_cls):
            obs_space = algorithm_cls.value.observation_space(
                env_config["wc_clusters_configs"][agent_id]["num_devices"],
                env_config["wc_clusters_configs"][agent_id]["L_max"],
            )
            action_space = algorithm_cls.value.action_space(
                env_config["wc_clusters_configs"][agent_id]["num_devices"]
            )

            if algorithm_cls == Algorithms.SACPA or algorithm_cls == Algorithms.SACPF:
                actor = SACPAACtor(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                    device=self.device,
                )
                actor_optim = Adam(actor.parameters(), lr=self.SAC_config["lr"])
                critic1 = SACPACritic(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                    device=self.device,
                )
                critic1_optim = Adam(critic1.parameters(), lr=self.SAC_config["lr"])
                critic2 = SACPACritic(
                    observation_space=obs_space,
                    action_space=action_space,
                    **self.model_config,
                    device=self.device,
                )
                critic2_optim = Adam(critic2.parameters(), lr=self.SAC_config["lr"])

                # auto entropy tuning setup
                target_entropy = float(-np.prod(action_space.shape))
                log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
                alpha_optim = Adam([log_alpha], lr=self.SAC_config["lr"])

                # schedulers += [
                #     CosineAnnealingLR(
                #         actor_optim, T_max=self.env_config["max_num_step"]
                #     ),
                #     CosineAnnealingLR(
                #         critic1_optim, T_max=self.env_config["max_num_step"]
                #     ),
                #     CosineAnnealingLR(
                #         critic2_optim, T_max=self.env_config["max_num_step"]
                #     ),
                #     CosineAnnealingLR(
                #         alpha_optim, T_max=self.env_config["max_num_step"]
                #     ),
                # ]

                policy = algorithm_cls.value(
                    SAC(
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
                )
            elif algorithm_cls == Algorithms.RAQL:
                policy = algorithm_cls.value(RAQL(action_space))
            elif algorithm_cls == Algorithms.RANDOM:
                policy = algorithm_cls.value(Random(action_space))
            elif algorithm_cls == Algorithms.DQN:
                q_net = DQNQNetwork(
                    obs_space, action_space, **self.model_config, device=self.device
                )
                q_net_optim = Adam(q_net.parameters(), lr=self.SAC_config["lr"])

                policy = algorithm_cls.value(DQN(q_net, q_net_optim, action_space))
            else:
                raise NotImplementedError

            algorithm_mapping.update({str(agent_id): policy})

        return algorithm_mapping

    def export(self, path: str):
        """
        Export to YAML file
        """
