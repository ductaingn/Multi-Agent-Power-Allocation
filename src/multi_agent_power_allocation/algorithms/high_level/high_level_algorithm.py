from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

import attrs

import numpy as np

import torch

from gymnasium.spaces import Space

from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)
from multi_agent_power_allocation.algorithms.low_level.low_level_algorithm import (
    LowLevelAlgorithm,
)

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class Reward:
    reward_sum: float
    reward_components: Dict[str, float]


@attrs.define
class Algorithm(ABC):
    low_level_algorithm: LowLevelAlgorithm

    @classmethod
    @abstractmethod
    def observation_space(cls, *args, **kwargs) -> Space:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def action_space(cls, *args, **kwargs) -> Space:
        raise NotImplementedError

    def learn(
        self, data: ReplayBufferSamples
    ) -> Tuple[float, float, float, float, float]:
        return self.low_level_algorithm.learn(data)

    @abstractmethod
    def get_state(self, wc_cluster: "WirelessCommunicationCluster") -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_number_send_packet_and_power(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        low_level_policy_output: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(
        self, wc_cluster: "WirelessCommunicationCluster", *args, **kwargs
    ) -> Reward:
        raise NotImplementedError
