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


def full_budget_power(num_send_packet: np.ndarray) -> np.ndarray:
    """
    Share the whole power budget uniformly among the devices (1/K each) and, within a device,
    among its active interfaces. Same split as SACPF (SACRA-Va).
    Returns the power share of each link, shape (num_devices, 2).
    """
    num_devices = num_send_packet.shape[0]
    active = (num_send_packet > 0).astype(float)
    n_active = np.maximum(active.sum(axis=1, keepdims=True), 1.0)
    return active / n_active / num_devices


@attrs.define
class CumulativeQoSReward:
    """
    Reward (21) of Dinh et al., ICC 2022 (RAQL):
        r(t) = 1/t * sum_{tau<=t} sum_k PSR_k(tau) - sum_k [(1 - u_k^sub(t)) + (1 - u_k^mW(t))]
    i.e. the packet success ratio is averaged over time while the QoS penalty uses the current
    per-interface PLR, with u_k^v(t) = 1 if rho_k^v(t) <= rho_max (14).
    """

    psr_sum: float = 0.0
    num_frames: int = 0

    def __call__(self, wc_cluster, reward_coef: Dict[str, float]) -> "Reward":
        sent = wc_cluster.num_send_packet.sum(axis=1)
        received = wc_cluster.num_received_packet.sum(axis=1)
        self.psr_sum += float(np.sum(received / sent))
        self.num_frames += 1
        psr_average = self.psr_sum / self.num_frames

        qos_satisfied = wc_cluster.packet_loss_rate <= wc_cluster.qos_threshold  # (K, 2)
        penalty = float(np.sum(1 - qos_satisfied))

        reward_qos = psr_average - penalty
        return Reward(
            reward_sum=reward_coef["reward_qos"] * reward_qos,
            reward_components={
                "reward_qos": reward_qos,
                "reward_psr_average": psr_average,
                "reward_qos_penalty": penalty,
            },
        )


@attrs.define
class Reward:
    reward_sum: float
    reward_components: Dict[str, float]


@attrs.define
class Algorithm(ABC):
    low_level_algorithm: LowLevelAlgorithm
    # True: learn at every frame from the latest transition only, from the first frame on
    # (tabular RAQL, Algorithm 1 of Dinh et al.); False: replay mini-batches after a random warm-up
    learns_online = False

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
