from pettingzoo import ParallelEnv
import attrs
from ..wireless_communication_cluster import WirelessCommunicationCluster
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import random
import wandb


@attrs.define
class WirelessEnvironmentBase(ParallelEnv):
    """
    Base class for wireless environments in PettingZoo API.
    This class is designed to be extended by specific wireless environment implementations.
    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "wireless_environment_base",
        "is_parallelizable": True,
    }
    reward_coef: Dict[str, float]
    wc_cluster_config: Dict[str, Any]
    num_cluster: int = attrs.field(default=2, kw_only=True)
    max_num_step: int = attrs.field(default=10_000)
    current_step: int = attrs.field(default=1)
    seed: Optional[int] = None

    
    def __attrs_post_init__(self):
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        self.agents: List[str] = [str(i) for i in range(self.num_cluster)]
        self.possible_agents = self.agents[:]

        if not (self.wc_cluster_config.get("LOS_PATH_LOSS") and self.wc_cluster_config.get("NLOS_PATH_LOSS")):
            num_devices = self.wc_cluster_config.get("num_devices")
            self.wc_cluster_config.update(
                {"LOS_PATH_LOSS": np.random.normal(0, 5.8, size=(self.max_num_step + 1, num_devices))}
            )
            self.wc_cluster_config.update(
                {"NLOS_PATH_LOSS": np.random.normal(0, 8.7, size=(self.max_num_step + 1, num_devices))}
            )

        self.wc_clusters:Dict[str, WirelessCommunicationCluster] = {
            id: WirelessCommunicationCluster(
                cluster_id=id,
                **self.wc_cluster_config
            )
            for id in self.agents
        }

    def reset(self, *, seed = None, options = None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def step(self, actions):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def render(self):
        pass

    def observation_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def action_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")