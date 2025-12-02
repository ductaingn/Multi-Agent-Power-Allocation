"""
Wireless Communication Cluster Module
This module defines the base class for wireless communication cluster, each cluster represents a group of one Access Point (AP) serves K IoT devices through wireless communication.
"""

import os
from typing import Dict, Union
import random
import json
import pickle
import attrs

import numpy as np

from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.wireless_environment.utils import (
    signal_power,
    gamma,
    compute_rate,
    compute_h_sub,
    compute_h_mW,
    generate_h_tilde,
    segments_intersect,
    rotate_points,
)
from multi_agent_power_allocation.wireless_environment.constants import AP_RANGE
from multi_agent_power_allocation.algorithms.algorithm_register import Algorithms
from multi_agent_power_allocation.algorithms.high_level import Reward


@attrs.define(slots=False)
class WirelessCommunicationCluster:
    """
    Base class for wireless communication clusters.
    This class is designed to be extended by specific wireless communication cluster implementations.
    """

    n_warm_up_step: int = attrs.field(
        metadata={"description": "Number of warm up step."}
    )

    current_step: int = attrs.field(
        default=1, init=False, metadata={"description": "Holds the step count."}
    )

    h_tilde: np.ndarray = attrs.field(
        metadata={"description": "Channel state information matrix."}
    )

    num_devices: int = attrs.field(
        metadata={"description": "Number of IoT device in cluster"}
    )

    device_positions: np.ndarray = attrs.field(
        metadata={"description": "Positions of devices in the cluster."}
    )

    num_sub_channel: int = attrs.field(
        metadata={"description": "Number of Sub-6GHz subchannel in the cluster"}
    )

    num_beam: int = attrs.field(
        metadata={"description": "Number of mmWave beam in the cluster"}
    )

    obstacle_positions: np.ndarray = attrs.field(
        metadata={"description": "Positions of obstacles in the cluster."}
    )

    LOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Line-of-sight path loss for mmWave connections."}
    )

    NLOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Non-line-of-sight path loss for mmWave connections."}
    )

    cluster_id: int = attrs.field(
        default=0, metadata={"description": "Unique identifier for the cluster."}
    )

    AP_position: np.ndarray = attrs.field(
        default=np.array([0.0, 0.0]),
        metadata={"description": "Position of the Access Point (AP) in the cluster."},
    )

    distance_to_AP: np.ndarray = attrs.field(
        default=None,
        metadata={
            "description": "Distances from each IoT device to its Access Point (AP) in the cluster."
        },
        init=False,
    )

    L_max: int = attrs.field(
        default=10,
        metadata={"description": "Maximum number of packet AP send to each device."},
    )

    P_sum: float = attrs.field(
        default=0.00316,
        metadata={
            "description": "Total transmision power available for the cluster in Watt."
        },
    )

    D: int = attrs.field(
        default=8000, metadata={"description": "Size of one packet in bit."}
    )

    T: int = attrs.field(
        default=1e-3, metadata={"description": "Time duration of one step in seconds."}
    )

    qos_threshold: float = attrs.field(
        default=0.1,
        metadata={
            "description": "Quality of service threshold (by PLR) for the cluster."
        },
    )

    n_sub_channels: int = attrs.field(
        default=4,
        metadata={"description": "Number of subchannels available in the cluster."},
    )

    n_beams: int = attrs.field(
        default=4, metadata={"description": "Number of beams available in the cluster."}
    )

    W_sub_total: float = attrs.field(
        default=1e8, metadata={"description": "Total Sub-6GHz bandwidth in Hz."}
    )

    W_mw_total: float = attrs.field(
        default=1e9, metadata={"description": "Total mmWave bandwidth in Hz."}
    )

    Sigma_sqr: float = attrs.field(
        default=pow(10, -169 / 10) * 1e-3,
        metadata={"description": "Noise power at device sides (-169 dBm/Hz)."},
    )

    W_sub: float = attrs.field(init=False)
    W_mw: float = attrs.field(init=False)

    _init_num_send_packet: np.ndarray = attrs.field(init=False)
    num_send_packet: np.ndarray = attrs.field(init=False)
    num_sent_packet_acc: np.ndarray = attrs.field(init=False)

    _init_num_received_packet: np.ndarray = attrs.field(init=False)
    num_received_packet: np.ndarray = attrs.field(init=False)
    num_received_packet_acc: np.ndarray = attrs.field(init=False)
    l_max_estimate: np.ndarray = attrs.field(init=False)

    _init_transmit_power: np.ndarray = attrs.field(init=False)
    transmit_power: np.ndarray = attrs.field(init=False)

    _init_allocation: np.ndarray = attrs.field(init=False)
    allocation: np.ndarray = attrs.field(init=False)

    _init_signal_power: np.ndarray = attrs.field(init=False)
    signal_power: np.ndarray = attrs.field(init=False)

    estimated_channel_power: np.ndarray = attrs.field(init=False)
    estimated_CGINR: np.ndarray = attrs.field(init=False)

    _init_rate: np.ndarray = attrs.field(init=False)
    average_rate: np.ndarray = attrs.field(init=False)
    previous_rate: np.ndarray = attrs.field(init=False)
    instant_rate: np.ndarray = attrs.field(init=False)
    maximum_rate: np.ndarray = attrs.field(init=False)

    packet_loss_rate: np.ndarray = attrs.field(init=False)
    global_packet_loss_rate: np.ndarray = attrs.field(init=False)
    sum_packet_loss_rate: float = attrs.field(init=False)
    packet_loss_rate_time_window: int = attrs.field(default=10)
    packet_loss_rate_stacked: np.ndarray = attrs.field(
        init=False,
        metadata={
            "description": "instant packet loss rate of each device on each interface, stacked by `packet_loss_rate_time_window` time frames"
        },
    )
    average_rate_stacked: np.ndarray = attrs.field(
        init=False,
        metadata={
            "description": "instant rate of each device on each interface, stacked by `packet_loss_rate_time_window` time frames"
        },
    )

    estimated_ideal_power: np.ndarray = attrs.field(init=False)
    per_device_interference: np.ndarray = attrs.field(init=False)

    def __attrs_post_init__(self):
        """
        Post-initialization method to validate the cluster configuration.
        """
        self.W_sub = self.W_sub_total / self.n_sub_channels
        self.W_mw = self.W_mw_total / self.n_beams
        assert (
            self.num_devices == self.device_positions.shape[0]
        ), f"Number of devices ({self.num_devices}) doesn't match the shape of device positions ({self.device_positions.shape})"
        assert (
            self.num_sub_channel <= self.h_tilde.shape[-1]
        ), "Number of subchannel doesn't match the shape of h_tilde"
        assert (
            self.num_beam <= self.h_tilde.shape[-1]
        ), "Number of beam doesn't match the shape of h_tilde"

        self.distance_to_AP = np.linalg.norm(
            self.device_positions - self.AP_position, axis=1
        )

        self.current_step = 1

        self._init_num_send_packet: np.ndarray = np.zeros(
            shape=(self.num_devices, 2), dtype=int
        )
        self.num_send_packet = self._init_num_send_packet.copy()
        self.num_sent_packet_acc = (
            self._init_num_send_packet.copy()
        )  # Number of sent packet accumulated

        self._init_num_received_packet: np.ndarray = np.zeros_like(
            self._init_num_send_packet, dtype=int
        )
        self.num_received_packet = self._init_num_received_packet.copy()
        self.num_received_packet_acc = (
            self._init_num_received_packet.copy()
        )  # Number of sent packet accumulated

        self._init_transmit_power: np.ndarray = np.full(
            shape=(self.num_devices, 2), fill_value=1.0 / (self.num_devices * 2)
        )
        self.transmit_power = self._init_transmit_power.copy()

        self._init_allocation: np.ndarray = self.update_allocation(init=True)
        self.allocation = self._init_allocation.copy()

        self._init_signal_power: np.ndarray = self.update_signal_power(init=True)
        self.signal_power = self._init_signal_power.copy()

        self.estimated_channel_power = np.zeros(shape=(self.num_devices, 2))
        self.estimated_CGINR = np.zeros(shape=(self.num_devices, 2))

        # self._init_rate:np.ndarray = self.update_instant_rate(interference=np.zeros_like(self.signal_power), init=True)
        self._init_rate: np.ndarray = np.zeros(shape=(self.num_devices, 2))

        self.average_rate = self._init_rate.copy()
        self.previous_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.average_rate_stacked = np.zeros(
            shape=(self.packet_loss_rate_time_window, self.num_devices, 2)
        )
        self.average_rate_stacked[:, ...] = self._init_rate.copy()

        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        self.global_packet_loss_rate = np.zeros(shape=self.num_devices)
        self.sum_packet_loss_rate = 0
        self.packet_loss_rate_stacked = np.zeros(
            shape=(self.packet_loss_rate_time_window, self.num_devices, 2)
        )

        self.maximum_rate: np.ndarray = np.array(
            [
                [
                    compute_rate(
                        w=self.W_sub,
                        sinr=gamma(
                            w=self.W_sub,
                            s=self.P_sum,
                            interference=0,
                            noise=self.Sigma_sqr,
                        ),
                    )
                    for k in range(self.num_devices)
                ],
                [
                    compute_rate(
                        w=self.W_mw,
                        sinr=gamma(
                            w=self.W_mw,
                            s=self.P_sum,
                            interference=0,
                            noise=self.Sigma_sqr,
                        ),
                    )
                    for k in range(self.num_devices)
                ],
            ]
        )

        self.estimated_ideal_power = np.zeros(
            shape=(self.num_devices, 2)
        )  # Unit: Percentage
        self.per_device_interference = np.zeros_like(self.transmit_power)

    @classmethod
    def generate_postitions(
        cls,
        scenario_name: str,
        num_cluster: int,
        num_device: int,
    ):
        """
        Generate AP, IoT devices and obstacles positions
        They are fixed for now
        """
        clusters = []
        clusters.append(
            {
                "AP": [100.0, 100.0],
                "devices": [[120.0, 100.0], [100.0, 120.0], [15.0, 20.0]],
                "obstacles": [[[90.0, 110.0], [110.0, 110.0]]],
            }
        )
        clusters.append(
            {
                "AP": [-100.0, 100.0],
                "devices": [[-80.0, 100.0], [-100.0, 120.0], [-185.0, 20.0]],
                "obstacles": [[[-90.0, 110.0], [-110.0, 110.0]]],
            }
        )
        clusters.append(
            {
                "AP": [-100.0, -100.0],
                "devices": [[-80.0, -100.0], [-100.0, -80.0], [-185.0, -180.0]],
                "obstacles": [[[-90.0, -90.0], [-110.0, -90.0]]],
            }
        )
        clusters.append(
            {
                "AP": [100.0, -100.0],
                "devices": [[80.0, -100.0], [100.0, -80.0], [15.0, -180.0]],
                "obstacles": [[[110.0, -90.0], [90.0, -90.0]]],
            }
        )

        if num_cluster > 4:
            raise NotImplementedError("Supported upto 4 APs only!")

        for i in range(num_cluster):

            AP_pos = clusters[i]["AP"]
            for k in range(num_device):

                if k >= 3:
                    clusters[i]["devices"].append(
                        [
                            np.random.randint(
                                AP_pos[0] - AP_RANGE / 2, AP_pos[1] - AP_RANGE / 2
                            ),
                            np.random.randint(
                                AP_pos[1] - AP_RANGE / 2, AP_pos[1] - AP_RANGE / 2
                            ),
                        ]
                    )

            save_path = os.path.join(
                BASE_DIR, "data", scenario_name, f"cluster_{i}", "positions.json"
            )

            with open(save_path, "wt", encoding="utf-8") as file:
                json.dump(clusters[i], file, indent=4)

    @classmethod
    def generate_h_tilde(
        cls,
        scenario_name: str,
        num_cluster: int,
        num_timestep: int,
        num_device: int,
        num_subchannel: int,
        num_beam: int,
        mu: float,
        sigma: float,
        seed: int,
    ):
        """
        Generate channel power gain for all IoT devices and subchannel/beam pair
        Array of generated complex channel coefficients with shape (num_AP, num_timestep, 2, num_device, num_subchannel + num_beam).
        """
        for i in range(num_cluster):
            save_path = os.path.join(
                BASE_DIR, "data", scenario_name, f"cluster_{i}", "h_tilde.pickle"
            )

            h = []
            for j in range(num_cluster):
                np.random.seed(seed * i + j)
                random.seed(seed * i + j)

                h_ij = generate_h_tilde(
                    num_timestep + 1,
                    num_device,
                    num_subchannel,
                    num_beam,
                    mu,
                    sigma,
                )

                h.append(h_ij)

            h = np.array(h)

            with open(save_path, "wb") as file:
                pickle.dump(h, file)

    @classmethod
    def generate_data(
        cls,
        scenario_name: str = "scenario_1",
        num_cluster: int = 2,
        num_timestep: int = 10_000,
        num_device: int = 3,
        num_subchannel: int = 5,
        num_beam: int = 5,
        mu: float = 0,
        sigma: float = 1,
        seed: int = 1,
    ):
        for i in range(num_cluster):
            os.makedirs(
                os.path.join(BASE_DIR, "data", scenario_name, f"cluster_{i}"),
                exist_ok=True,
            )

        cls.generate_h_tilde(
            scenario_name,
            num_cluster,
            num_timestep,
            num_device,
            num_subchannel,
            num_beam,
            mu,
            sigma,
            seed,
        )
        cls.generate_postitions(scenario_name, num_cluster, num_device)

    def set_num_send_packet(self, num_send_packet: np.ndarray):
        self.num_send_packet = num_send_packet.copy()

    def set_transmit_power(self, transmit_power: np.ndarray):
        self.transmit_power = transmit_power.copy()

    def update_allocation(self, init: bool = False) -> Union[None, np.ndarray]:
        """
        Allocate subchannels and beams to devices randomly based on the number of packets to be sent.

        Parameters
        ----------
        num_send_packet : np.ndarray
            Array of shape (num_devices, 2) representing the number of packets to be sent to each device.
        init: bool
            Whether if this function is called at initiation or not.

        Returns
        -------
        None
        """
        sub = []  # Stores index of subchannel device will allocate
        mW = []  # Stores index of beam device will allocate
        for i in range(self.num_devices):
            sub.append(-1)
            mW.append(-1)

        rand_sub = []
        rand_mW = []
        for i in range(self.num_sub_channel):
            rand_sub.append(i)
        for i in range(self.num_beam):
            rand_mW.append(i)

        for k in range(self.num_devices):
            if self.num_send_packet[k, 0] > 0 and self.num_send_packet[k, 1] == 0:
                rand_index = int(np.random.randint(0, len(rand_sub)))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
            elif self.num_send_packet[k, 0] == 0 and self.num_send_packet[k, 1] > 0:
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
        self.allocation = allocation

        if init:
            return allocation

    def is_blocked(self, device_index: int) -> bool:
        """
        Check whether a device is blocked from AP by obstacles

        Parameters
        ----------
        device_index : int
            The index of the IoT device

        Returns
        -------
        res : bool
            Whether the device is blocked or not
        """
        for obs in self.obstacle_positions:
            if segments_intersect(
                obs, np.array([self.device_positions[device_index], self.AP_position])
            ):
                return True

        return False

    def update_signal_power(self, init: bool = False) -> Union[None, np.ndarray]:
        """
        Update the signal power for each device based on the current allocation and transmit power levels.

        Parameters
        ----------
        init: bool
            Whether if this function is called at initiation or not.

        Returns
        -------
        None
        """
        _signal_power = np.zeros(shape=(2, max(self.num_sub_channel, self.num_beam)))
        _channel_power = np.zeros(shape=(self.num_devices, 2))

        for k in range(self.num_devices):
            sub_channel_index = self.allocation[k, 0]
            mW_beam_index = self.allocation[k, 1]

            if sub_channel_index != -1:
                _channel_power[k, 0] = compute_h_sub(
                    distance_to_AP=self.distance_to_AP[k],
                    h_tilde=self.h_tilde[
                        self.cluster_id, self.current_step, 0, k, sub_channel_index
                    ],
                )

                _signal_power[0, sub_channel_index] = signal_power(
                    p=self.transmit_power[k, 0] * self.P_sum,
                    h=_channel_power[k, 0],
                )

            if mW_beam_index != -1:
                blocked = self.is_blocked(k)
                x = (
                    self.NLOS_PATH_LOSS[self.current_step, k]
                    if blocked
                    else self.LOS_PATH_LOSS[self.current_step, k]
                )
                _channel_power[k, 1] = compute_h_mW(
                    distance_to_AP=self.distance_to_AP[k], is_blocked=blocked, x=x
                )
                _signal_power[1, mW_beam_index] = signal_power(
                    p=self.transmit_power[k, 1] * self.P_sum,
                    h=_channel_power[k, 1],
                )

        self.signal_power = _signal_power

        if init:
            return _signal_power

    def update_instant_rate(
        self, interference: np.ndarray, init=False
    ) -> Union[None, np.ndarray]:
        """
        Compute the instantaneous rate for each device based on the current allocation and power levels.

        Parameters
        ----------
        interference : np.ndarray
            Array of shape (2, num_subchannels or num_beams) representing the interference at each subchannel/beam.
        init: bool
            Whether if this function is called at initiation or not.

        Returns
        -------
        instant_rate : np.ndarray
            Array of shape (num_devices, 2) representing the instantaneous rate for each device.

        """
        rate = np.zeros(shape=(self.num_devices, 2))
        per_device_interference = np.zeros_like(self.transmit_power)

        for k in range(self.num_devices):
            sub_channel_index = self.allocation[k, 0]
            mW_beam_index = self.allocation[k, 1]

            if sub_channel_index != -1:
                per_device_interference[k, 0] = interference[0, sub_channel_index]
                sinr = gamma(
                    w=self.W_sub,
                    s=self.signal_power[0, sub_channel_index],
                    interference=interference[0, sub_channel_index],
                    noise=self.Sigma_sqr,
                )

                rate[k, 0] = compute_rate(
                    w=self.W_sub,
                    sinr=sinr,
                )

            if mW_beam_index != -1:
                per_device_interference[k, 1] = interference[1, mW_beam_index]
                sinr = gamma(
                    w=self.W_mw,
                    s=self.signal_power[1, mW_beam_index],
                    interference=interference[1, mW_beam_index],
                    noise=self.Sigma_sqr,
                )
                rate[k, 1] = compute_rate(w=self.W_mw, sinr=sinr)

        self.instant_rate = rate
        self.per_device_interference = per_device_interference

        if init:
            return rate

    def update_average_rate(self):
        """
        Update the average rate for each device based on the current step and previous average rate.

        Returns
        -------
        None
        """
        average_rate = (
            1
            / self.current_step
            * (self.instant_rate + self.average_rate * (self.current_step - 1))
        )

        self.average_rate = average_rate

    def update_average_rate_stacked(self):
        self.average_rate_stacked[1:] = self.average_rate_stacked[:-1]
        self.average_rate_stacked[0] = self.instant_rate

    def update_packet_loss_rate(self):
        """
        Updates packet loss rate on each interfaces, devices packet loss rate on the whole, and system packet loss rate
        """
        num_send_packet = self.num_send_packet
        num_received_packet = self.num_received_packet
        packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        global_packet_loss_rate = np.zeros(shape=(self.num_devices))
        for k in range(self.num_devices):
            if num_send_packet[k, 0] > 0:
                packet_loss_rate[k, 0] = (
                    1
                    / self.current_step
                    * (
                        self.packet_loss_rate[k, 0] * (self.current_step - 1)
                        + (1 - num_received_packet[k, 0] / num_send_packet[k, 0])
                    )
                )
            else:
                packet_loss_rate[k, 0] = (
                    1
                    / self.current_step
                    * (self.packet_loss_rate[k, 0] * (self.current_step - 1))
                )

            if num_send_packet[k, 1] > 0:
                packet_loss_rate[k, 1] = (
                    1
                    / self.current_step
                    * (
                        self.packet_loss_rate[k, 1] * (self.current_step - 1)
                        + (1 - num_received_packet[k, 1] / num_send_packet[k, 1])
                    )
                )
            else:
                packet_loss_rate[k, 1] = (
                    1
                    / self.current_step
                    * (self.packet_loss_rate[k, 1] * (self.current_step - 1))
                )

            global_packet_loss_rate[k] = (
                1
                / self.current_step
                * (
                    self.global_packet_loss_rate[k] * (self.current_step - 1)
                    + (
                        1
                        - (num_received_packet[k, 0] + num_received_packet[k, 1])
                        / (num_send_packet[k, 0] + num_send_packet[k, 1])
                    )
                )
            )

        sum_packet_loss_rate = (
            1
            / self.current_step
            * (
                self.sum_packet_loss_rate * (self.current_step - 1)
                + (1 - num_received_packet.sum() / num_send_packet.sum())
            )
        )

        self.packet_loss_rate = packet_loss_rate
        self.global_packet_loss_rate = global_packet_loss_rate
        self.sum_packet_loss_rate = sum_packet_loss_rate

    def update_packet_loss_rate_stacked(self):
        packet_loss_rate_instant = 1 - np.divide(
            self.num_received_packet,
            self.num_send_packet,
            out=np.ones_like(self.packet_loss_rate, dtype=self.packet_loss_rate.dtype),
            where=self.num_send_packet > 0,
        )
        self.packet_loss_rate_stacked[1:] = self.packet_loss_rate_stacked[:-1]
        self.packet_loss_rate_stacked[0] = packet_loss_rate_instant

    def update_feedback(self, interference: np.ndarray):
        """
        Update the number of received packet at device side

        Parameters
        ----------
        interference : np.ndarray
            Array of shape (num_devices, 2) representing the interference subchannels and beams for each device.

        Returns
        -------
        None
        """
        self.update_instant_rate(interference)
        l_max = np.floor(np.multiply(self.instant_rate, self.T / self.D)).astype(int)

        self.num_received_packet = np.minimum(self.num_send_packet, l_max, dtype=int)

    def estimate_l_max(self, algorithm: "Algorithms"):
        """
        Estimate the maximum number of packets that can be sent to each device based on the average rate and current QoS state of each device.

        Returns
        -------
        None
        """
        l = np.multiply(self.average_rate_stacked.mean(axis=0), self.T / self.D)
        if isinstance(algorithm, Algorithms.RAQL.value) or isinstance(
            algorithm, Algorithms.DQN.value
        ):
            l_max_estimate = np.floor(l)
        elif (
            isinstance(algorithm, Algorithms.SACPA.value)
            or isinstance(algorithm, Algorithms.SACPF.value)
            or isinstance(algorithm, Algorithms.RANDOM.value)
        ):
            packet_successful_rate = np.ones(
                shape=(self.num_devices, 2)
            ) - self.packet_loss_rate_stacked.mean(axis=0)
            l_max_estimate = np.floor(l * packet_successful_rate)

            # After a long time of not sending via one interface,
            # the average rate drop so much that `l` becomes 0.0, eventhough the packet successful rate is 1.0
            # This prevents under-use of interfaces and improve exploration of the policy
            packet_successful_rate_warm_up_threshold = 1.0
            indx = np.where(
                packet_successful_rate >= packet_successful_rate_warm_up_threshold
            )
            l_max_estimate[indx] = np.full_like(l_max_estimate[indx], self.L_max)
        else:
            raise NotImplementedError

        self.l_max_estimate = l_max_estimate

    def estimate_channel_power(self):
        # for k in range(self.num_devices):
        #     if num_sent_packet[k, 0] > 0:
        #         sub_channel_index = allocation[k,0]
        #         # self.estimated_channel_power[k,0] = W_SUB*SIGMA_SQR*(2**(self.rate[k,0]/W_SUB))/(power[k,0]*self.P_sum)
        #         self.channel_power[k,0] = self.compute_h_sub(self.device_positions[k], self.h_tilde[self.current_step, 0, k, sub_channel_index])

        #     if num_sent_packet[k, 1] > 0:
        #         mW_beam_index = allocation[k,1]
        #         self.channel_power[k,1] = self.compute_h_mW(
        #             device_position=self.device_positions[k], device_index=k,
        #             h_tilde=self.h_tilde[self.current_step, 1, k, mW_beam_index])

        # self.estimated_channel_power
        ...

    def estimate_CGINR(self):
        """
        Estimate Channel Gain to Interference plus Noise Ratio of the previous step base on ACK/NACK feedback only, to calculate reward. (Lower bound estimation)
        """
        for k in range(self.num_devices):
            if self.num_send_packet[k, 0] > 0:
                self.estimated_CGINR[k, 0] = (
                    2
                    ** (
                        (self.num_received_packet[k, 0] * self.D)
                        / (self.W_sub * self.T)
                    )
                    - 1
                ) / (self.transmit_power[k, 0] * self.P_sum)

            if self.num_send_packet[k, 1] > 0:
                self.estimated_CGINR[k, 1] = (
                    2
                    ** (
                        (self.num_received_packet[k, 1] * self.D) / (self.W_mw * self.T)
                    )
                    - 1
                ) / (self.transmit_power[k, 1] * self.P_sum)

    def get_info(self, reward: Reward) -> Dict[str, float]:
        info = {}
        prefix = f"Agent {self.cluster_id}"

        info[f"{prefix}/ Overall/ Reward"] = reward.reward_sum
        info[f"{prefix}/ Overall/ Reward QoS"] = reward.reward_components.get(
            "reward_qos"
        )
        info[f"{prefix}/ Overall/ Reward Power"] = reward.reward_components.get(
            "reward_power"
        )
        info[f"{prefix}/ Overall/ Sum Packet loss rate"] = self.sum_packet_loss_rate
        info[f"{prefix}/ Overall/ Average rate/ Sub6GHz"] = self.average_rate[
            :, 0
        ].sum() / (self.num_devices)
        info[f"{prefix}/ Overall/ Average rate/ mmWave"] = self.average_rate[
            :, 1
        ].sum() / (self.num_devices)
        info[f"{prefix}/ Overall/ Average rate/ Global"] = (
            info[f"{prefix}/ Overall/ Average rate/ mmWave"]
            + info[f"{prefix}/ Overall/ Average rate/ mmWave"]
        )
        info[f"{prefix}/ Overall/ Power usage"] = self.transmit_power.sum()
        info[f"{prefix}/ Accumulate/ Num. Sent packet"] = self.num_sent_packet_acc
        info[f"{prefix}/ Accumulate/ Num. Received packet"] = (
            self.num_received_packet_acc
        )

        for k in range(self.num_devices):
            info[f"{prefix}/ Device {k+1}/ Num. Sent packet/ Sub6GHz"] = (
                self.num_send_packet[k, 0]
            )
            info[f"{prefix}/ Device {k+1}/ Num. Sent packet/ mmWave"] = (
                self.num_send_packet[k, 1]
            )

            info[f"{prefix}/ Device {k+1}/ Num. Received packet/ Sub6GHz"] = (
                self.num_received_packet[k, 0]
            )
            info[f"{prefix}/ Device {k+1}/ Num. Received packet/ mmWave"] = (
                self.num_received_packet[k, 1]
            )

            info[f"{prefix}/ Device {k+1}/ Num. Dropped packet/ Sub6GHz"] = (
                self.num_send_packet[k, 0] - self.num_received_packet[k, 0]
            )
            info[f"{prefix}/ Device {k+1}/ Num. Dropped packet/ mmWave"] = (
                self.num_send_packet[k, 1] - self.num_received_packet[k, 1]
            )

            info[f"{prefix}/ Device {k+1}/ Power/ Sub6GHz"] = self.transmit_power[k, 0]
            info[f"{prefix}/ Device {k+1}/ Power/ mmWave"] = self.transmit_power[k, 1]

            info[f"{prefix}/ Device {k+1}/ Packet loss rate/ Global"] = (
                self.global_packet_loss_rate[k]
            )
            info[f"{prefix}/ Device {k+1}/ Packet loss rate/ Sub6GHz"] = (
                self.packet_loss_rate[k, 0]
            )
            info[f"{prefix}/ Device {k+1}/ Packet loss rate/ mmWave"] = (
                self.packet_loss_rate[k, 1]
            )
            info[f"{prefix}/ Device {k+1}/ Packet loss rate time window/ Sub6GHz"] = (
                self.packet_loss_rate_stacked[:, k, 0].mean()
            )
            info[f"{prefix}/ Device {k+1}/ Packet loss rate time window/ mmWave"] = (
                self.packet_loss_rate_stacked[:, k, 1].mean()
            )
            info[f"{prefix}/ Device {k+1}/ Average rate/ Sub6GHz"] = self.average_rate[
                k, 0
            ]
            info[f"{prefix}/ Device {k+1}/ Average rate/ mmWave"] = self.average_rate[
                k, 1
            ]
            info[f"{prefix}/ Device {k+1}/ Average rate time window/ Sub6GHz"] = (
                self.average_rate_stacked[:, k, 0].mean()
            )
            info[f"{prefix}/ Device {k+1}/ Average rate time window/ mmWave"] = (
                self.average_rate_stacked[:, k, 1].mean()
            )
            info[f"{prefix}/ Device {k+1}/ Interference/ Sub6GHz"] = (
                self.per_device_interference[k, 0]
            )
            info[f"{prefix}/ Device {k+1}/ Interference/ mmWave"] = (
                self.per_device_interference[k, 1]
            )

            if hasattr(self, "estimated_ideal_power"):
                info[f"{prefix}/ Device {k+1}/ Estimated ideal power/ Sub6GHz"] = (
                    self.estimated_ideal_power[k, 0]
                )
                info[f"{prefix}/ Device {k+1}/ Estimated ideal power/ mmWave"] = (
                    self.estimated_ideal_power[k, 1]
                )

        return info

    def change_obstacle_positions(self, AP_positions):
        # Change obstacles to block Device 1 instead of Device 2 in all clusters
        # by rotating the obstacles around the center of their clusters
        new_obstacle_positions = []
        for obstacle_position, AP_position in zip(
            self.obstacle_positions, AP_positions
        ):
            new_obstacle_positions.append(
                rotate_points(obstacle_position, -90.0, AP_position)
            )

        self.obstacle_positions = np.array(new_obstacle_positions)

    def step(self):
        self.current_step += 1
        self.num_sent_packet_acc += self.num_send_packet
        self.num_received_packet_acc += self.num_received_packet

    def reset(self):
        self.current_step = 1
        self.average_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.average_rate_stacked = np.zeros_like(self.average_rate_stacked)
        self.average_rate_stacked[:, ...] = self._init_rate.copy()
        self.num_send_packet = self._init_num_send_packet
        self.num_sent_packet_acc = self._init_num_send_packet
        self.num_received_packet = self._init_num_received_packet
        self.num_received_packet_acc = self._init_num_received_packet
        self.transmit_power = self._init_transmit_power
        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        self.global_packet_loss_rate = np.zeros(shape=(self.num_devices))
        self.sum_packet_loss_rate = 0
        self.packet_loss_rate_stacked = np.zeros(
            shape=(self.packet_loss_rate_time_window, self.num_devices, 2)
        )
