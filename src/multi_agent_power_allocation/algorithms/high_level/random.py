from typing import Dict, TYPE_CHECKING

import attrs

import numpy as np

import torch

from multi_agent_power_allocation.algorithms.low_level.random import Random as LLRandom
from multi_agent_power_allocation.algorithms.high_level.sacpa import SACPA

if TYPE_CHECKING:
    from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
        WirelessCommunicationCluster,
    )


@attrs.define
class Random(SACPA):
    low_level_algorithm: LLRandom = attrs.field(factory=LLRandom)

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

        assert np.all(
            number_of_send_packet.sum(axis=1) > 0
        ), "AP must send packet to every IoT devices"
        assert np.all(power.sum(axis=1) > 0), "AP must send packet to every IoT devices"

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)
