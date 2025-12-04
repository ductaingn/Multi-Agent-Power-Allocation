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

    def compute_channel_allocation(  # pylint: disable=W0221
        self, wc_cluster: "WirelessCommunicationCluster"
    ):
        """
        Allocate subchannels and beams to devices randomly based on the number of packets to be sent.

        Parameters
        ----------
        num_send_packet : np.ndarray
            Array of shape (num_devices, 2) representing the number of packets to be sent to each device.

        Returns
        -------
        None
        """
        sub = []  # Stores index of subchannel device will allocate
        mW = []  # Stores index of beam device will allocate
        for i in range(wc_cluster.num_devices):
            sub.append(-1)
            mW.append(-1)

        rand_sub = []
        rand_mW = []
        for i in range(wc_cluster.num_sub_channel):
            rand_sub.append(i)
        for i in range(wc_cluster.num_beam):
            rand_mW.append(i)

        for k in range(wc_cluster.num_devices):
            if (
                wc_cluster.num_send_packet[k, 0] > 0
                and wc_cluster.num_send_packet[k, 1] == 0
            ):
                rand_index = int(np.random.randint(0, len(rand_sub)))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
            elif (
                wc_cluster.num_send_packet[k, 0] == 0
                and wc_cluster.num_send_packet[k, 1] > 0
            ):
                rand_index = int(np.random.randint(0, len(rand_mW)))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
            else:
                rand_sub_index = int(np.random.randint(0, len(rand_sub)))
                rand_mW_index = int(np.random.randint(0, len(rand_mW)))

                sub[k] = rand_sub[rand_sub_index]
                mW[k] = rand_mW[rand_mW_index]

                rand_sub.pop(rand_sub_index)
                rand_mW.pop(rand_mW_index)

        allocation = np.array([sub, mW], dtype=int).transpose()
        wc_cluster.set_channel_allocation(allocation)

    @abstractmethod
    def compute_reward(
        self, wc_cluster: "WirelessCommunicationCluster", *args, **kwargs
    ) -> Reward:
        raise NotImplementedError

    def update_environment_info(
        self, wc_cluster: "WirelessCommunicationCluster", *args, **kwargs
    ):
        """
        Update information after applying action and taking feedbacks
        """
        pass
