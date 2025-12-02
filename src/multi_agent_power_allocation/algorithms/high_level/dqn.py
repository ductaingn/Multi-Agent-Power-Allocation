from typing import Dict, TYPE_CHECKING

import attrs

import numpy as np

import torch

from gymnasium.spaces import Space, Box, Discrete

from multi_agent_power_allocation.algorithms.high_level.high_level_algorithm import (
    Algorithm,
    Reward,
)
from multi_agent_power_allocation.algorithms.low_level.dqn import DQN as LLDQN

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class DQN(Algorithm):
    low_level_algorithm: LLDQN
    num_iot_devices: int
    interface_hash_map: Dict[int, np.ndarray] = attrs.field(init=False)

    @interface_hash_map.default
    def _interface_hash_map_factory(self):
        grids = np.meshgrid(*[np.arange(3)] * self.num_iot_devices, indexing="ij")
        states = (
            np.stack(grids, axis=0).reshape(self.num_iot_devices, -1).T
        )  # shape: (3**self.num_iot_devices, self.num_iot_devices)

        # store reversed state vectors to match original behavior
        values = states[:, ::-1].copy()
        hash_map = {i: values[i] for i in range(values.shape[0])}

        return hash_map

    @classmethod
    def observation_space(  # pylint: disable=W0221
        cls, num_iot_devices, L_max
    ) -> Space:
        """
        Observation space contains:
            - Quality of Service Satisfaction of each device on Sub6GHz/mmWave, respectively
            - Number of received packets of each device on Sub6GHz/mmWave of previous time step, respectively
        Flattened
        """
        return Box(
            low=np.array(
                [
                    np.zeros((num_iot_devices), dtype=int),
                    np.zeros((num_iot_devices), dtype=int),
                    np.zeros((num_iot_devices), dtype=int),
                    np.zeros((num_iot_devices), dtype=int),
                ]
            )
            .transpose()
            .flatten(),
            high=np.array(
                [
                    np.ones((num_iot_devices), dtype=int),
                    np.ones((num_iot_devices), dtype=int),
                    np.full((num_iot_devices), fill_value=L_max, dtype=int),
                    np.full((num_iot_devices), fill_value=L_max, dtype=int),
                ]
            )
            .transpose()
            .flatten(),
            dtype=int,
        )

    @classmethod
    def action_space(cls, num_iot_devices) -> Space:  # pylint: disable=W0221
        """
        Action space contains:
            - Which interface to send packets
                `0` for Sub-6GHz
                `1` for mmWave
                `2` for both interfaces
        Flattened
        """
        return Discrete(3**num_iot_devices)

    def get_state(self, wc_cluster: "WirelessCommunicationCluster") -> np.ndarray:
        _state = np.zeros(
            shape=(
                wc_cluster.num_devices,
                self.observation_space(wc_cluster.num_devices, wc_cluster.L_max).shape[
                    -1
                ]
                // wc_cluster.num_devices,
            )
        )
        # QoS satisfaction
        _state[:, 0] = (
            wc_cluster.packet_loss_rate[:, 0] <= wc_cluster.qos_threshold
        ).astype(float)
        _state[:, 1] = (
            wc_cluster.packet_loss_rate[:, 1] <= wc_cluster.qos_threshold
        ).astype(float)
        _state[:, 2] = wc_cluster.num_received_packet[:, 0].copy() / wc_cluster.L_max
        _state[:, 3] = wc_cluster.num_received_packet[:, 1].copy() / wc_cluster.L_max

        return _state

    def compute_number_send_packet_and_power(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        low_level_policy_output: torch.Tensor,
    ):
        power = np.full(
            shape=(wc_cluster.num_devices, 2),
            fill_value=1.0 / (wc_cluster.num_sub_channel + wc_cluster.num_beam),
        )

        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
        else:
            number_of_send_packet = np.zeros(
                shape=(wc_cluster.num_devices, 2), dtype=int
            )

            interfaces = self.interface_hash_map[low_level_policy_output]

            for k in range(wc_cluster.num_devices):
                if interfaces[k] == 0:
                    number_of_send_packet[k, 0] = max(
                        1, min(wc_cluster.l_max_estimate[k, 0], wc_cluster.L_max)
                    )

                if interfaces[k] == 1:
                    number_of_send_packet[k, 1] = max(
                        1, min(wc_cluster.l_max_estimate[k, 1], wc_cluster.L_max)
                    )

                if interfaces[k] == 2:
                    if wc_cluster.l_max_estimate[k, 1] < wc_cluster.L_max:
                        number_of_send_packet[k, 1] = max(
                            1, wc_cluster.l_max_estimate[k, 1]
                        )
                        number_of_send_packet[k, 0] = min(
                            max(1, wc_cluster.l_max_estimate[k, 0]),
                            wc_cluster.L_max - number_of_send_packet[k, 1],
                        )
                    else:
                        number_of_send_packet[k, 0] = 1
                        number_of_send_packet[k, 1] = wc_cluster.L_max - 1

                # For analysing purpose other channel
                if number_of_send_packet[k, 0] == 0:
                    power[k, 0] = 0
                if number_of_send_packet[k, 1] == 0:
                    power[k, 1] = 0

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)

    def compute_reward(  # pylint: disable=W0221
        self,
        wc_cluster: "WirelessCommunicationCluster",
        prev_reward_qos: float,
        reward_coef: Dict[str, float],
    ) -> Reward:
        reward_qos = 0.0

        for k in range(wc_cluster.num_devices):
            qos_satisfaction = (
                wc_cluster.packet_loss_rate[k, 0] < wc_cluster.qos_threshold,
                wc_cluster.packet_loss_rate[k, 1] < wc_cluster.qos_threshold,
            )

            num_received_packet = (
                wc_cluster.num_received_packet[k, 0],
                wc_cluster.num_received_packet[k, 1],
            )

            num_send_packet = (
                wc_cluster.num_send_packet[k, 0],
                wc_cluster.num_send_packet[k, 1],
            )

            reward_qos += (
                (num_received_packet[0] + num_received_packet[1])
                / (num_send_packet[0] + num_send_packet[1])
                - (1 - qos_satisfaction[0])
                - (1 - qos_satisfaction[1])
            )
        reward_qos = (
            (wc_cluster.current_step - 1) * prev_reward_qos + reward_qos
        ) / wc_cluster.current_step

        instance_reward = reward_coef["reward_qos"] * reward_qos

        return Reward(
            reward_sum=instance_reward,
            reward_components={
                "reward_qos": reward_qos,
            },
        )
