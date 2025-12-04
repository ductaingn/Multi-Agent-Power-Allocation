from typing import Dict, TYPE_CHECKING

import attrs

import numpy as np

import torch
from torch.nn.functional import softmax

from gymnasium.spaces import Space, Box

from multi_agent_power_allocation.algorithms.high_level.high_level_algorithm import (
    Algorithm,
    Reward,
)
from multi_agent_power_allocation.algorithms.low_level.sac import SAC

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class SACPA(Algorithm):
    low_level_algorithm: SAC
    num_devices: int = attrs.field(
        metadata={"description": "Number of IoT device in cluster"}
    )
    environment_info_time_window: int = attrs.field(default=100)
    packet_loss_rate_stacked: np.ndarray = attrs.field(
        init=False,
        metadata={
            "description": "Instant packet loss rate of each device on each interface, stacked by `environment_info_time_window` time frames"
        },
    )
    average_rate_stacked: np.ndarray = attrs.field(
        init=False,
        metadata={
            "description": "Instant rate of each device on each interface, stacked by `environment_info_time_window` time frames"
        },
    )

    @packet_loss_rate_stacked.default
    def _packet_loss_rate_stacked_factory(self):
        return np.zeros(shape=(self.environment_info_time_window, self.num_devices, 2))

    @average_rate_stacked.default
    def _average_rate_stacked_factory(self):
        return np.zeros(shape=(self.environment_info_time_window, self.num_devices, 2))

    @classmethod
    def observation_space(  # pylint: disable=W0221
        cls, num_iot_devices, L_max
    ) -> Space:
        """
        Observation space contains:
            - Quality of Service Satisfaction of each device on Sub6GHz/mmWave, respectively
            - Number of received packets of each device on Sub6GHz/mmWave of previous time step, respectively
            - Average Rate of each device on Sub6GHz/mmWave of previous time step, respectively
            - Power of each device on Sub6GHz on previous time step, respectively
        Flattened
        """
        return Box(
            low=np.array(
                [
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    # TODO: test with this state space
                    # np.zeros((num_iot_devices)),  # Estimated ideal power of each device on Sub6GHz on previous time step
                    # np.zeros((num_iot_devices)),  # Estimated ideal power of each device on mmWave on previous time step
                ],
                dtype=np.float32,
            )
            .transpose()
            .flatten(),
            high=np.array(
                [
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    np.full((num_iot_devices), fill_value=L_max),
                    np.full((num_iot_devices), fill_value=L_max),
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    # TODO: Test with this state space
                    # np.ones((self.num_iot_devices)),  # Estimated ideal power of each device on Sub6GHz of previous time step
                    # np.ones((self.num_iot_devices)),  # Estimated ideal power of each device on mmWave of previous time step
                ],
                dtype=np.float32,
            )
            .transpose()
            .flatten(),
        )

    @classmethod
    def action_space(cls, num_iot_devices) -> Space:  # pylint: disable=W0221
        """
        Action space contains:
            - Number of packets to send of each device on Sub6GHz/mmWave, respectively
            - Transmit power to each device on Sub6GHz/mmWave of previous time step, respectively
        Flattened
        """
        return Box(
            low=np.array(
                [
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                ],
                dtype=np.float32,
            ).flatten(),
            high=np.array(
                [
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                    np.ones((num_iot_devices)),
                ],
                dtype=np.float32,
            ).flatten(),
        )

    def get_state(self, wc_cluster: "WirelessCommunicationCluster") -> np.ndarray:
        _state = np.zeros(
            shape=(
                wc_cluster.num_devices,
                self.__class__.observation_space(
                    wc_cluster.num_devices, wc_cluster.L_max
                ).shape[-1]
                // wc_cluster.num_devices,
            )
        )
        # QoS satisfaction
        _state[:, 0] = (
            self.packet_loss_rate_stacked.mean(axis=0)[:, 0] <= wc_cluster.qos_threshold
        ).astype(float)
        _state[:, 1] = (
            self.packet_loss_rate_stacked.mean(axis=0)[:, 1] <= wc_cluster.qos_threshold
        ).astype(float)
        _state[:, 2] = wc_cluster.num_received_packet[:, 0].copy() / wc_cluster.L_max
        _state[:, 3] = wc_cluster.num_received_packet[:, 1].copy() / wc_cluster.L_max
        _state[:, 4] = (
            self.average_rate_stacked.mean(axis=0)[:, 0] / wc_cluster.maximum_rate[0]
        )
        _state[:, 5] = (
            self.average_rate_stacked.mean(axis=0)[:, 1] / wc_cluster.maximum_rate[1]
        )
        _state[:, 6] = wc_cluster.transmit_power[:, 0].copy() * 10.0  # Scale up
        _state[:, 7] = wc_cluster.transmit_power[:, 1].copy() * 10.0
        # TODO: Test with this state space
        # _state[:, 8] = wc_cluster.estimated_ideal_power[:, 0].copy() * 10.0
        # _state[:, 9] = wc_cluster.estimated_ideal_power[:, 1].copy() * 10.0

        return _state

    def estimate_l_max(self, wc_cluster: "WirelessCommunicationCluster"):
        """
        Estimate the maximum number of packets that can be sent to each device based on the average rate and current QoS state of each device.

        Returns
        -------
        None
        """
        l = np.multiply(
            self.average_rate_stacked.mean(axis=0), wc_cluster.T / wc_cluster.D
        )

        packet_successful_rate = np.ones(
            shape=(wc_cluster.num_devices, 2)
        ) - self.packet_loss_rate_stacked.mean(axis=0)
        estimated_l_max = np.floor(l * packet_successful_rate)

        # After a long time of not sending via one interface,
        # the average rate drop so much that `l` becomes 0.0, eventhough the packet successful rate is 1.0
        # This prevents under-use of interfaces and improve exploration of the policy
        packet_successful_rate_warm_up_threshold = 1.0
        indx = np.where(
            packet_successful_rate >= packet_successful_rate_warm_up_threshold
        )
        estimated_l_max[indx] = np.full_like(estimated_l_max[indx], wc_cluster.L_max)

        return estimated_l_max

    def estimate_CGINR(self, wc_cluster: "WirelessCommunicationCluster") -> np.ndarray:
        """
        Estimate Channel Gain to Interference plus Noise Ratio of the previous step base on ACK/NACK feedback only, to calculate reward. (Lower bound estimation)
        """
        estimated_CGINR = np.zeros(shape=(wc_cluster.num_devices, 2))
        for k in range(wc_cluster.num_devices):
            if wc_cluster.num_send_packet[k, 0] > 0:
                wc_cluster.estimated_CGINR[k, 0] = (
                    2
                    ** (
                        (wc_cluster.num_received_packet[k, 0] * wc_cluster.D)
                        / (wc_cluster.W_sub * wc_cluster.T)
                    )
                    - 1
                ) / (wc_cluster.transmit_power[k, 0] * wc_cluster.P_sum)

            if wc_cluster.num_send_packet[k, 1] > 0:
                wc_cluster.estimated_CGINR[k, 1] = (
                    2
                    ** (
                        (wc_cluster.num_received_packet[k, 1] * wc_cluster.D)
                        / (wc_cluster.W_mw * wc_cluster.T)
                    )
                    - 1
                ) / (wc_cluster.transmit_power[k, 1] * wc_cluster.P_sum)

        return estimated_CGINR

    def compute_number_send_packet_and_power(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        low_level_policy_output: torch.Tensor,
    ):
        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
            power = np.full_like(
                wc_cluster.transmit_power, 1.0 / (wc_cluster.num_devices * 2)
            )
        else:
            estimated_l_max = self.estimate_l_max(wc_cluster)

            power_start_index = 2 * wc_cluster.num_devices
            interface_score = low_level_policy_output[:power_start_index].reshape(
                wc_cluster.num_devices, 2
            )
            interface_score = torch.softmax(
                torch.tensor(interface_score), dim=1
            ).numpy()

            number_of_send_packet = np.minimum(
                np.minimum(
                    interface_score * wc_cluster.L_max,
                    estimated_l_max,
                ).astype(int),
                wc_cluster.L_max,
            )

            power = low_level_policy_output[power_start_index:]
            power = torch.softmax(torch.tensor(power), dim=-1).numpy()
            power = power.reshape(wc_cluster.num_devices, 2)

            # Use time window packet loss rate
            packet_loss_rate = self.packet_loss_rate_stacked.mean(axis=0)

            for k in range(wc_cluster.num_devices):
                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1] == 0
                ):  # Force to send at least one packet on more reliable channel
                    if packet_loss_rate[k, 0] <= packet_loss_rate[k, 1]:
                        number_of_send_packet[k, 0] = 1
                    else:
                        number_of_send_packet[k, 1] = 1

                if (
                    number_of_send_packet[k, 0] + number_of_send_packet[k, 1]
                    > wc_cluster.L_max
                ):
                    # If the number of packets to send exceeds the maximum number of packets that can be sent
                    # then send on both channels by the proportion of the packet success rate
                    if np.sum(packet_loss_rate[k]) == 0:
                        psr_proportion = 0.5
                    else:
                        psr_proportion = 1 - packet_loss_rate[k, 0] / np.sum(
                            packet_loss_rate[k]
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

    def compute_reward(  # pylint: disable=W0221
        self,
        wc_cluster: "WirelessCommunicationCluster",
        prev_reward_qos: float,
        reward_coef: Dict[str, float],
    ) -> Reward:
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
        estimated_CGINR = self.estimate_CGINR(wc_cluster)

        for k in range(wc_cluster.num_devices):
            # Unit: percentage
            transmit_power = (
                wc_cluster.transmit_power[k, 0],
                wc_cluster.transmit_power[k, 1],
            )

            CGINR = (
                estimated_CGINR[k, 0],
                estimated_CGINR[k, 1],
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
        # reward_qos = (
        #     (wc_cluster.current_step - 1) * prev_reward_qos + reward_qos
        # ) / wc_cluster.current_step

        instance_reward = (
            reward_coef["reward_qos"] * reward_qos
            + reward_coef["reward_power"] * reward_power
        )

        return Reward(
            reward_sum=instance_reward,
            reward_components={
                "reward_qos": reward_qos,
                "reward_power": reward_power,
            },
        )

    def update_packet_loss_rate_stacked(
        self, wc_cluster: "WirelessCommunicationCluster"
    ):
        packet_loss_rate_instant = 1 - np.divide(
            wc_cluster.num_received_packet,
            wc_cluster.num_send_packet,
            out=np.ones_like(
                wc_cluster.packet_loss_rate, dtype=wc_cluster.packet_loss_rate.dtype
            ),
            where=wc_cluster.num_send_packet > 0,
        )
        self.packet_loss_rate_stacked[1:] = self.packet_loss_rate_stacked[:-1]
        self.packet_loss_rate_stacked[0] = packet_loss_rate_instant

    def update_average_rate_stacked(self, wc_cluster: "WirelessCommunicationCluster"):
        self.average_rate_stacked[1:] = self.average_rate_stacked[:-1]
        self.average_rate_stacked[0] = wc_cluster.instant_rate

    def update_environment_info(
        self, wc_cluster: "WirelessCommunicationCluster"
    ):  # pylint: disable=W0221
        # NOTE: The execution order of these functions calls matter
        self.update_packet_loss_rate_stacked(wc_cluster)
        self.update_average_rate_stacked(wc_cluster)
