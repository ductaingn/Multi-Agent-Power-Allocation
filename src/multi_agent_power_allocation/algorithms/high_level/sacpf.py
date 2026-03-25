from typing import TYPE_CHECKING

import attrs

import numpy as np

import torch

from gymnasium.spaces import Space, Box

from multi_agent_power_allocation.algorithms.high_level.sacpa import SACPA

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class SACPF(SACPA):
    @classmethod
    def observation_space(  # pylint: disable=W0221
        cls, num_iot_devices, L_max
    ) -> Space:
        """
        Observation space contains:
            - Quality of Service Satisfaction of each device on Sub6GHz/mmWave, respectively
            - Number of received packets of each device on Sub6GHz/mmWave of previous time step, respectively
            - Average Rate of each device on Sub6GHz/mmWave of previous time step, respectively
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
        Flattened
        """
        return Box(
            low=np.array(
                [
                    np.zeros((num_iot_devices)),
                    np.zeros((num_iot_devices)),
                ],
                dtype=np.float32,
            ).flatten(),
            high=np.array(
                [
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

        return _state

    def compute_number_send_packet_and_power(
        self,
        wc_cluster: "WirelessCommunicationCluster",
        low_level_policy_output: torch.Tensor,
    ):
        power = np.full_like(
            wc_cluster.transmit_power, 1.0 / (wc_cluster.num_devices * 2)
        )
        if wc_cluster.current_step <= wc_cluster.n_warm_up_step:
            number_of_send_packet = np.full_like(
                wc_cluster.num_send_packet, wc_cluster.L_max
            )
        else:
            estimated_l_max = self.estimate_l_max(wc_cluster)

            interface_score = low_level_policy_output.reshape(wc_cluster.num_devices, 2)
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

        assert np.all(
            number_of_send_packet.sum(axis=1) > 0
        ), "AP must send packet to every IoT devices"
        assert np.all(power.sum(axis=1) > 0), "AP must send packet to every IoT devices"

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)
