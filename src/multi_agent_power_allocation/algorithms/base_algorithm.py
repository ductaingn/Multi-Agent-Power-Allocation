from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, TYPE_CHECKING

import attrs

import torch
from torch.nn.functional import softmax

import numpy as np

import gymnasium as gym

from multi_agent_power_allocation.nn.module import SACPAACtor
from multi_agent_power_allocation.utils.replay_buffer import ReplayBufferSamples

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class DummyActor:
    def train(self, mode: bool = False):
        pass

    def __call__(self, obs, **kwds):
        pass


@attrs.define
class BaseAlgorithm(ABC):
    actor: DummyActor | SACPAACtor

    def get_actions(self, obs, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def learn(
        self, data: ReplayBufferSamples
    ) -> Tuple[float, float, float, float, float]:
        """
        Return
            actor_losses
            critic_losses
            critic2_losses
            alpha_losses
            alphas
        """
        raise NotImplementedError()


@dataclass
class SpaceParams:
    num_devices: int
    L_max: int

    def build_obs_space(self):
        """
        Build observation space for SACPA, SACPF, Random algorithm.
        Observation space contains:
            - Quality of Service Satisfaction of each device on Sub6GHz/mmWave, respectively
            - Number of received packets of each device on Sub6GHz/mmWave of previous time step, respectively
            - Average Rate of each device on Sub6GHz/mmWave of previous time step, respectively
            - Power of each device on Sub6GHz on previous time step, respectively
        Flattened
        """
        return gym.spaces.Box(
            low=np.array(
                [
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    # TODO: test with this state space
                    # np.zeros((self.num_devices)),  # Estimated ideal power of each device on Sub6GHz on previous time step
                    # np.zeros((self.num_devices)),  # Estimated ideal power of each device on mmWave on previous time step
                ],
                dtype=np.float32,
            )
            .transpose()
            .flatten(),
            high=np.array(
                [
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    np.full((self.num_devices), fill_value=self.L_max),
                    np.full((self.num_devices), fill_value=self.L_max),
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    # TODO: Test with this state space
                    # np.ones((self.num_devices)),  # Estimated ideal power of each device on Sub6GHz of previous time step
                    # np.ones((self.num_devices)),  # Estimated ideal power of each device on mmWave of previous time step
                ],
                dtype=np.float32,
            )
            .transpose()
            .flatten(),
        )

    def build_obs_space_raql(self):
        """
        Build observation space for SACPA, SACPF, Random algorithm.
        Observation space contains:
            - Quality of Service Satisfaction of each device on Sub6GHz/mmWave, respectively
            - Number of received packets of each device on Sub6GHz/mmWave of previous time step, respectively
        Flattened
        """
        return gym.spaces.Box(
            low=np.array(
                [
                    np.zeros((self.num_devices), dtype=int),
                    np.zeros((self.num_devices), dtype=int),
                    np.zeros((self.num_devices), dtype=int),
                    np.zeros((self.num_devices), dtype=int),
                ]
            )
            .transpose()
            .flatten(),
            high=np.array(
                [
                    np.ones((self.num_devices), dtype=int),
                    np.ones((self.num_devices), dtype=int),
                    np.full((self.num_devices), fill_value=self.L_max, dtype=int),
                    np.full((self.num_devices), fill_value=self.L_max, dtype=int),
                ]
            )
            .transpose()
            .flatten(),
            dtype=int,
        )

    def build_action_space(self):
        """
        Build action space for SACPA, SACPF, Random algorithm.
        Action space contains:
            - Number of packets to send of each device on Sub6GHz/mmWave, respectively
            - Transmit power to each device on Sub6GHz/mmWave of previous time step, respectively
        Flattened
        """
        return gym.spaces.Box(
            low=np.array(
                [
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                    np.zeros((self.num_devices)),
                ],
                dtype=np.float32,
            ).flatten(),
            high=np.array(
                [
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                    np.ones((self.num_devices)),
                ],
                dtype=np.float32,
            ).flatten(),
        )

    def build_action_space_raql(self):
        """
        Build action space for RAQL algorithm.
        Action space contains:
            - Which interface to send packets
                `0` for Sub-6GHz
                `1` for mmWave
                `2` for both interfaces
        Flattened
        """
        return gym.spaces.Box(
            low=np.array(
                [
                    np.zeros((self.num_devices), dtype=int),
                ]
            ).flatten(),
            high=np.array(
                [
                    np.full((self.num_devices), fill_value=2, dtype=int),
                ]
            ).flatten(),
            dtype=int,
        )

    def build_action_space_dqn(self):
        """
        Build action space for DQN algorithm.
        Action space contains:
            - Which interface to send packets
                `0` for Sub-6GHz
                `1` for mmWave
                `2` for both interfaces

        Flattened
        """
        return gym.spaces.Discrete(3**self.num_devices)


class Algorithm(Enum):
    """
    High level algorithms
    """

    SACPA = "SACPA"
    SACPF = "SACPF"
    DQN = "DQN"
    RAQL = "RAQL"
    RANDOM = "Random"

    def observation_space(self, num_devices: int, L_max: int):
        params = SpaceParams(num_devices, L_max)
        if self == Algorithm.RAQL or self == Algorithm.DQN:
            return params.build_obs_space_raql()
        else:
            return params.build_obs_space()

    def action_space(self, num_devices: int, L_max: int):
        params = SpaceParams(num_devices, L_max)
        if self == Algorithm.RAQL:
            return params.build_action_space_raql()
        if self == Algorithm.DQN:
            grids = np.meshgrid(*[np.arange(3)] * num_devices, indexing="ij")
            states = (
                np.stack(grids, axis=0).reshape(num_devices, -1).T
            )  # shape: (3**num_devices, num_devices)

            # store reversed state vectors to match original behavior
            values = states[:, ::-1].copy()
            hash_map = {i: values[i] for i in range(values.shape[0])}

            self.interface_hash_map = hash_map

            return params.build_action_space_dqn()
        else:
            return params.build_action_space()

    def compute_number_send_packet_and_power(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        policy_output: torch.Tensor,
    ) -> None:
        """
        Compute the number of packets to send and power for one cluster based on the its agent's policy network output.

        Parameters
        ----------
        policy_network_output : torch.Tensor
            Tensor containing the output from the policy network of its agent.

        Returns
        -------
        None
        """
        if self == Algorithm.SACPA:
            self.compute_number_send_packet_and_power_SACPA(wc_cluster, policy_output)
        elif self == Algorithm.SACPF:
            self.compute_number_send_packet_and_power_SACPF(wc_cluster, policy_output)
        elif self == Algorithm.RAQL:
            self.compute_number_send_packet_and_power_RAQL(wc_cluster, policy_output)
        elif self == Algorithm.RANDOM:
            self.compute_number_send_packet_and_power_Random(wc_cluster, policy_output)
        elif self == Algorithm.DQN:
            self.compute_number_send_packet_and_power_DQN(wc_cluster, policy_output)
        else:
            raise NotImplementedError

    def compute_number_send_packet_and_power_SACPA(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        policy_network_output: torch.Tensor,
    ) -> None:
        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
            power = np.full_like(
                wc_cluster.transmit_power, 1.0 / (wc_cluster.num_devices * 2)
            )
        else:
            power_start_index = 2 * wc_cluster.num_devices
            interface_score = policy_network_output[:power_start_index].reshape(
                wc_cluster.num_devices, 2
            )
            interface_score = torch.softmax(
                torch.tensor(interface_score), dim=1
            ).numpy()

            number_of_send_packet = np.minimum(
                np.minimum(
                    interface_score * wc_cluster.L_max,
                    wc_cluster.l_max_estimate,
                ).astype(int),
                wc_cluster.L_max,
            )

            power = policy_network_output[power_start_index:]
            power = torch.softmax(torch.tensor(power), dim=-1).numpy()
            power = power.reshape(wc_cluster.num_devices, 2)

            for k in range(wc_cluster.num_devices):
                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1] == 0
                ):  # Force to send at least one packet on more reliable channel
                    if (
                        wc_cluster.packet_loss_rate[k, 0]
                        <= wc_cluster.packet_loss_rate[k, 1]
                    ):
                        number_of_send_packet[k, 0] = 1
                    else:
                        number_of_send_packet[k, 1] = 1

                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1]
                    > wc_cluster.L_max
                ):
                    # If the number of packets to send exceeds the maximum number of packets that can be sent
                    # then send on both channels by the proportion of the packet success rate
                    if np.sum(wc_cluster.packet_loss_rate[k]) == 0:
                        psr_proportion = 0.5
                    else:
                        psr_proportion = 1 - wc_cluster.packet_loss_rate[k, 0] / np.sum(
                            wc_cluster.packet_loss_rate[k]
                        )
                    number_of_send_packet[k, 0] = np.floor(
                        psr_proportion * wc_cluster.L_max
                    )
                    number_of_send_packet[k, 1] = (
                        wc_cluster.L_max - number_of_send_packet[k, 0]
                    )

                # Send the remaining power to the other channel
                if number_of_send_packet[k, 0] == 0:
                    power[k, 1] += power[k, 0]
                    power[k, 0] = 0
                if number_of_send_packet[k, 1] == 0:
                    power[k, 0] += power[k, 1]
                    power[k, 1] = 0

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)

    def compute_number_send_packet_and_power_SACPF(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        policy_network_output: torch.Tensor,
    ) -> None:
        power = np.full_like(
            wc_cluster.transmit_power, 1.0 / (wc_cluster.num_devices * 2)
        )
        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
        else:
            power_start_index = 2 * wc_cluster.num_devices
            interface_score = policy_network_output[:power_start_index].reshape(
                wc_cluster.num_devices, 2
            )
            interface_score = torch.softmax(
                torch.tensor(interface_score), dim=1
            ).numpy()

            number_of_send_packet = np.minimum(
                np.minimum(
                    interface_score * wc_cluster.L_max,
                    wc_cluster.l_max_estimate,
                ).astype(int),
                wc_cluster.L_max,
            )

            for k in range(wc_cluster.num_devices):
                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1] == 0
                ):  # Force to send at least one packet on more reliable channel
                    if (
                        wc_cluster.packet_loss_rate[k, 0]
                        <= wc_cluster.packet_loss_rate[k, 1]
                    ):
                        number_of_send_packet[k, 0] = 1
                    else:
                        number_of_send_packet[k, 1] = 1

                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1]
                    > wc_cluster.L_max
                ):
                    # If the number of packets to send exceeds the maximum number of packets that can be sent
                    # then send on both channels by the proportion of the packet success rate
                    if np.sum(wc_cluster.packet_loss_rate[k]) == 0:
                        psr_proportion = 0.5
                    else:
                        psr_proportion = 1 - wc_cluster.packet_loss_rate[k, 0] / np.sum(
                            wc_cluster.packet_loss_rate[k]
                        )
                    number_of_send_packet[k, 0] = np.floor(
                        psr_proportion * wc_cluster.L_max
                    )
                    number_of_send_packet[k, 1] = (
                        wc_cluster.L_max - number_of_send_packet[k, 0]
                    )

                # Send the remaining power to the other channel
                if number_of_send_packet[k, 0] == 0:
                    power[k, 1] += power[k, 0]
                    power[k, 0] = 0
                if number_of_send_packet[k, 1] == 0:
                    power[k, 0] += power[k, 1]
                    power[k, 1] = 0

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)

    def compute_number_send_packet_and_power_RAQL(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        policy_output: torch.Tensor,
    ) -> None:
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

            for k in range(wc_cluster.num_devices):
                if policy_output[k] == 0:
                    number_of_send_packet[k, 0] = max(
                        1, min(wc_cluster.l_max_estimate[k, 0], wc_cluster.L_max)
                    )

                if policy_output[k] == 1:
                    number_of_send_packet[k, 1] = max(
                        1, min(wc_cluster.l_max_estimate[k, 1], wc_cluster.L_max)
                    )

                if policy_output[k] == 2:
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

    def compute_number_send_packet_and_power_DQN(
        self, wc_cluster: "WirelessCommunicationCluster", policy_output: torch.Tensor
    ) -> None:
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

            interfaces = self.interface_hash_map[policy_output]

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

    def compute_number_send_packet_and_power_Random(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        policy_network_output: torch.Tensor,
    ) -> None:
        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
            power = np.full_like(
                wc_cluster.transmit_power, 1.0 / (wc_cluster.num_devices * 2)
            )
        else:
            number_of_send_packet: np.ndarray = np.random.randint(
                0, wc_cluster.L_max, (wc_cluster.num_devices, 2)
            )
            power = (
                torch.softmax(
                    torch.tensor(np.random.rand(wc_cluster.num_devices * 2)), dim=-1
                )
                .reshape(wc_cluster.num_devices, 2)
                .numpy()
            )

            for k in range(wc_cluster.num_devices):
                if number_of_send_packet[k].sum() == 0:
                    if np.random.rand() > 0.5:
                        number_of_send_packet[k, 0] = 1
                        power[k, 1] = 0  # For analyzing purpose
                    else:
                        number_of_send_packet[k, 1] = 1
                        power[k, 0] = 0
                if number_of_send_packet[k].sum() > wc_cluster.L_max:
                    proportion = (
                        number_of_send_packet[k, 0] / number_of_send_packet[k].sum()
                    )
                    number_of_send_packet[k, 0] = np.floor(
                        proportion * wc_cluster.L_max
                    )
                    number_of_send_packet[k, 1] = (
                        wc_cluster.L_max - number_of_send_packet[k, 0]
                    )

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)

    def get_state(self, wc_cluster: "WirelessCommunicationCluster") -> np.ndarray:
        if self == Algorithm.RAQL or self == Algorithm.DQN:
            _state = np.zeros(
                shape=(
                    wc_cluster.num_devices,
                    self.observation_space(
                        wc_cluster.num_devices, wc_cluster.L_max
                    ).shape[-1]
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
            _state[:, 2] = wc_cluster.num_received_packet[:, 0].copy()
            _state[:, 3] = wc_cluster.num_received_packet[:, 1].copy()

        else:
            _state = np.zeros(
                shape=(
                    wc_cluster.num_devices,
                    self.observation_space(
                        wc_cluster.num_devices, wc_cluster.L_max
                    ).shape[-1]
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
            _state[:, 2] = wc_cluster.num_received_packet[:, 0].copy()
            _state[:, 3] = wc_cluster.num_received_packet[:, 1].copy()
            _state[:, 4] = wc_cluster.average_rate[:, 0] / wc_cluster.maximum_rate[0]
            _state[:, 5] = wc_cluster.average_rate[:, 1] / wc_cluster.maximum_rate[1]
            _state[:, 6] = wc_cluster.transmit_power[:, 0].copy() * 10.0  # Scale up
            _state[:, 7] = wc_cluster.transmit_power[:, 1].copy() * 10.0
            # TODO: Test with this state space
            # _state[:, 8] = wc_cluster.estimated_ideal_power[:, 0].copy() * 10.0
            # _state[:, 9] = wc_cluster.estimated_ideal_power[:, 1].copy() * 10.0

        return _state

    def compute_reward(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        prev_reward_qos: float,
        reward_coef: Dict[str, float],
    ) -> Dict[str, float]:
        if self == Algorithm.RAQL or self == Algorithm.DQN:
            return self.compute_reward_RAQL(wc_cluster, prev_reward_qos, reward_coef)
        elif (
            self == Algorithm.SACPA
            or self == Algorithm.SACPF
            or self == Algorithm.RANDOM
        ):
            return self.compute_reward_SACPA(wc_cluster, prev_reward_qos, reward_coef)
        else:
            raise NotImplementedError

    def compute_reward_SACPA(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        prev_reward_qos: float,
        reward_coef: Dict[str, float],
    ):
        def estimate_ideal_power(num_send_packet, CGINR, W):
            if CGINR == 0:
                return 1.0

            ideal_power = (
                2 ** ((num_send_packet * wc_cluster.D) / (W * wc_cluster.T)) - 1
            ) / CGINR
            return min(ideal_power / wc_cluster.P_sum, 1.0)

        reward_qos = 0
        reward_power = 0
        target_power = []
        predicted_power = []

        for k in range(wc_cluster.num_devices):
            # Unit: percentage
            transmit_power = (
                wc_cluster.transmit_power[k, 0],
                wc_cluster.transmit_power[k, 1],
            )

            CGINR = (
                wc_cluster.estimated_CGINR[k, 0],
                wc_cluster.estimated_CGINR[k, 1],
            )

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

            if num_send_packet[0] > 0:
                wc_cluster.estimated_ideal_power[k, 0] = estimate_ideal_power(
                    num_send_packet[0], CGINR[0], wc_cluster.W_sub
                )
                target_power.append(wc_cluster.estimated_ideal_power[k, 0])
                predicted_power.append(transmit_power[0])
            else:
                wc_cluster.estimated_ideal_power[k, 0] = 0.0

            if num_send_packet[1] > 0:
                wc_cluster.estimated_ideal_power[k, 1] = estimate_ideal_power(
                    num_send_packet[1], CGINR[1], wc_cluster.W_mw
                )
                target_power.append(wc_cluster.estimated_ideal_power[k, 1])
                predicted_power.append(transmit_power[1])
            else:
                wc_cluster.estimated_ideal_power[k, 1] = 0.0

        target_power = torch.tensor(target_power)
        target_power = softmax(target_power, dim=-1)
        predicted_power = torch.tensor(predicted_power)

        reward_power = -wc_cluster.num_devices * np.tanh(
            (target_power * (target_power.log() - predicted_power.log())).sum().item()
        )
        reward_qos = (
            (wc_cluster.current_step - 1) * prev_reward_qos + reward_qos
        ) / wc_cluster.current_step

        instance_reward = (
            reward_coef["reward_qos"] * reward_qos
            + reward_coef["reward_power"] * reward_power
        )

        return {
            "reward_qos": reward_qos,
            "reward_power": reward_power,
            "instant_reward": instance_reward,
        }

    def compute_reward_RAQL(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        prev_reward_qos: float,
        reward_coef: Dict[str, float],
    ):
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

        return {
            "reward_qos": reward_qos,
            "instant_reward": instance_reward,
        }
