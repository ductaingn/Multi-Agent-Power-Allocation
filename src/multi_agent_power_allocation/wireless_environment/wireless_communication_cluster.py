"""
Wireless Communication Cluster Module
This module defines the base class for wireless communication cluster, each cluster represents a group of one Access Point (AP) serves K IoT devices through wireless communication.
"""
import numpy as np
import attrs
from multi_agent_power_allocation.wireless_environment.utils import (
    signal_power, 
    gamma,compute_rate, 
    compute_h_sub, 
    compute_h_mW
)
from typing import Dict, Union


@attrs.define(slots=False)
class WirelessCommunicationCluster:
    """
    Base class for wireless communication clusters.
    This class is designed to be extended by specific wireless communication cluster implementations.
    """
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

    device_blockages: list = attrs.field(
        metadata={"description": "Hash array indicating whether each device is blocked by obstacles."}
    )

    LOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Line-of-sight path loss for mmWave connections."}
    )

    NLOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Non-line-of-sight path loss for mmWave connections."}
    )

    cluster_id: int = attrs.field(
        default=0,
        metadata={"description": "Unique identifier for the cluster."}
    )

    AP_position: np.ndarray = attrs.field(
        default=np.array([0.0, 0.0]),
        metadata={"description": "Position of the Access Point (AP) in the cluster."}
    )

    L_max: int = attrs.field(
        default=10,
        metadata={"description": "Maximum number of packet AP send to each device."}
    )

    P_sum: float = attrs.field(
        default=0.00316,
        metadata={"description": "Total transmision power available for the cluster in Watt."}
    )

    D: int = attrs.field(
        default=8000,
        metadata={"description": "Size of one packet in bit."}
    )

    T: int = attrs.field(
        default=1e-3,
        metadata={"description": "Time duration of one step in seconds."}
    )

    qos_threshold: float = attrs.field(
        default=0.1,
        metadata={"description": "Quality of service threshold (by PLR) for the cluster."}
    )

    n_sub_channels: int = attrs.field(
        default=4,
        metadata={"description": "Number of subchannels available in the cluster."}
    )

    n_beams: int = attrs.field(
        default=4,
        metadata={"description": "Number of beams available in the cluster."}
    )

    W_sub_total: float = attrs.field(
        default=1e8,
        metadata={"description": "Total Sub-6GHz bandwidth in Hz."}
    )

    W_mw_total: float = attrs.field(
        default=1e9,
        metadata={"description": "Total mmWave bandwidth in Hz."}
    )

    Sigma_sqr: float = attrs.field(
        default=pow(10, -169/10)*1e-3,
        metadata={"description": "Noise power at device sides (-169 dBm/Hz)."}
    )

    # _W_sub: float = attrs.field(init=False)
    # _W_mw: float = attrs.field(init=False)

    def __attrs_post_init__(self):
        """
        Post-initialization method to validate the cluster configuration.
        """
        self._W_sub = self.W_sub_total / self.n_sub_channels
        self._W_mw = self.W_mw_total / self.n_beams
        assert self.num_devices == self.device_positions.shape[0], f"Number of devices ({self.num_devices}) doesn't match the shape of device positions ({self.device_positions.shape})"
        assert self.num_sub_channel == self.h_tilde.shape[-1], "Number of subchannel doesn't match the shape of h_tilde"
        assert self.num_beam == self.h_tilde.shape[-1], "Number of beam doesn't match the shape of h_tilde"

        self.distance_to_AP = np.linalg.norm(
            self.device_positions - self.AP_position, axis=1
        )

        self.current_step = 1

        self._init_num_send_packet:np.ndarray = np.zeros(shape=(self.num_devices, 2), dtype=int)
        self.num_send_packet = self._init_num_send_packet.copy()

        self._init_num_received_packet:np.ndarray = np.zeros_like(self._init_num_send_packet, dtype=int)
        self.num_received_packet = self._init_num_received_packet

        self._init_transmit_power:np.ndarray = np.ones(shape=(self.num_devices, 2))
        self.transmit_power = self._init_transmit_power.copy()
        
        self._init_allocation:np.ndarray = self.update_allocation(init=True)
        self.allocation = self._init_allocation.copy()

        self._init_signal_power:np.ndarray = self.update_signal_power(init=True)
        self.signal_power = self._init_signal_power.copy()

        self.channel_power_gain = np.zeros(shape=(self.num_devices, 2))

        self._init_rate:np.ndarray = self.update_instant_rate(interference=np.zeros_like(self.signal_power), init=True)
        
        self.average_rate = self._init_rate.copy()
        self.previous_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()

        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        self.global_packet_loss_rate = np.zeros(shape=self.num_devices)
        self.sum_packet_loss_rate = 0

        self.maximum_rate:np.ndarray = np.array([
            [
                compute_rate(
                    w=self._W_sub, 
                    sinr=gamma(w=self._W_sub, s=self._init_transmit_power[k, 0], interference=0, noise=self.Sigma_sqr)
                ) for k in range(self.num_devices)
            ],
            [
                compute_rate(
                    w=self._W_mw, 
                    sinr=gamma(w=self._W_mw, s=self._init_transmit_power[k, 1], interference=0, noise=self.Sigma_sqr)
                ) for k in range(self.num_devices)
            ],
        ])

        self.estimated_ideal_power = np.zeros(shape=(self.num_devices, 2))


    def set_num_send_packet(self, num_send_packet: np.ndarray) -> None:
        self.num_send_packet = num_send_packet.copy()

    def set_transmit_power(self, transmit_power: np.ndarray) -> None:
        self.transmit_power = transmit_power.copy()


    def update_allocation(self, init:bool=False) -> Union[None, np.ndarray]:
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
            if (self.num_send_packet[k,0]>0 and self.num_send_packet[k,1]==0):
                rand_index = np.random.randint(0,len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
            elif (self.num_send_packet[k,0]==0 and self.num_send_packet[k,1]>0):
                rand_index = np.random.randint(0,len(rand_mW))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
            else:
                rand_sub_index = np.random.randint(0,len(rand_sub))
                rand_mW_index = np.random.randint(0,len(rand_mW))

                sub[k] = rand_sub[rand_sub_index]
                mW[k] = rand_mW[rand_mW_index]

                rand_sub.pop(rand_sub_index)
                rand_mW.pop(rand_mW_index)

        allocation = np.array([sub, mW], dtype=int).transpose()
        self.allocation = allocation

        if init:
            return allocation


    def update_signal_power(self, init:bool=False) -> Union[None, np.ndarray]:
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
        _channel_power_gain = np.zeros(shape=(self.num_devices, 2))
        
        for k in range(self.num_devices):
            sub_channel_index = self.allocation[k, 0]
            mW_beam_index = self.allocation[k, 1]

            if (sub_channel_index != -1):
                _channel_power_gain[k, 0] = compute_h_sub(
                    distance_to_AP=self.distance_to_AP[k],
                    h_tilde=self.h_tilde[self.current_step, 0, k, sub_channel_index]
                )

                _signal_power[0, sub_channel_index] = signal_power(
                    p = self.transmit_power[k, 0]* self.P_sum,
                    h = _channel_power_gain[k, 0]
                )

            if (mW_beam_index != -1):
                is_blocked = self.device_blockages[k]
                x = self.NLOS_PATH_LOSS[self.current_step, k] if is_blocked else self.LOS_PATH_LOSS[self.current_step, k]
                _channel_power_gain[k, 1] = compute_h_mW(
                    distance_to_AP=self.distance_to_AP[k],
                    is_blocked=is_blocked,
                    x=x
                )
                _signal_power[1, mW_beam_index] = signal_power(
                    p = self.transmit_power[k, 1]* self.P_sum,
                    h = _channel_power_gain[k, 1]
                )
                
        self.signal_power = _signal_power
        self.channel_power_gain = _channel_power_gain

        if init:
            return _signal_power

    def update_instant_rate(self, interference:np.ndarray, init=False) -> Union[None, np.ndarray]:
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
        
        for k in range(self.num_devices):
            sub_channel_index = self.allocation[k, 0]
            mW_beam_index = self.allocation[k, 1]

            if (sub_channel_index != -1):
                sinr = gamma(
                    w=self._W_sub,
                    s=self.signal_power[0, sub_channel_index],
                    interference=interference[0, sub_channel_index],
                    noise=self.Sigma_sqr
                )

                rate[k,0] = compute_rate(
                    w=self._W_sub,
                    sinr=sinr,
                )

            if (mW_beam_index != -1):
                sinr = gamma(
                    w=self._W_mw,
                    s=self.signal_power[1, mW_beam_index],
                    interference=interference[1, mW_beam_index],
                    noise=self.Sigma_sqr
                )
                rate[k,1] = compute_rate(
                    w=self._W_mw,
                    sinr=sinr
                )

        self.instant_rate = rate

        if init:
            return rate
    

    def update_average_rate(self) -> None:
        """
        Update the average rate for each device based on the current step and previous average rate.
        
        Returns
        -------
        None
        """
        average_rate = 1/self.current_step*(self.instant_rate + self.average_rate*(self.current_step-1))

        self.average_rate = average_rate


    def update_packet_loss_rate(self) -> None:
        '''
        Updates packet loss rate on each interfaces, devices packet loss rate on the whole, and system packet loss rate
        '''
        num_send_packet = self.num_send_packet
        num_received_packet = self.num_received_packet
        packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        global_packet_loss_rate = np.zeros(shape=(self.num_devices))
        for k in range(self.num_devices):
            if num_send_packet[k,0] > 0:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1) + (1 - num_received_packet[k,0]/num_send_packet[k,0]))
            else:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1))
            
            if num_send_packet[k,1] > 0:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1) + (1 - num_received_packet[k,1]/num_send_packet[k,1]))
            else:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1))

            global_packet_loss_rate[k] = 1/self.current_step*(self.global_packet_loss_rate[k]*(self.current_step-1) + (1 - (num_received_packet[k,0] + num_received_packet[k,1])/(num_send_packet[k,0] + num_send_packet[k,1])))

        sum_packet_loss_rate = 1/self.current_step*(self.sum_packet_loss_rate*(self.current_step-1) + (1-num_received_packet.sum()/num_send_packet.sum()))

        self.packet_loss_rate = packet_loss_rate
        self.global_packet_loss_rate = global_packet_loss_rate
        self.sum_packet_loss_rate = sum_packet_loss_rate    
    

    def update_feedback(self, interference:np.ndarray) -> None:
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
        l_max = np.floor(np.multiply(self.instant_rate, self.T/self.D))

        self.num_received_packet = np.minimum(self.num_send_packet, l_max)
    

    def estimate_l_max(self) -> None:
        """
        Estimate the maximum number of packets that can be sent to each device based on the average rate and current QoS state of each device.

        Returns
        -------
        None
        """
        l = np.multiply(self.average_rate, self.T/self.D)
        packet_successful_rate = np.ones(shape=(self.num_devices,2)) - self.packet_loss_rate
        l_max_estimate = np.floor(l*packet_successful_rate)

        self.l_max_estimate = l_max_estimate

    def estimate_average_channel_power(self):
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

        self.channel_power_gain


    def get_info(self, reward:Dict[str, float]) -> Dict[str, float]:
        info = {}
        prefix = f'Agent {self.cluster_id}'

        info[f'{prefix}/ Overall/ Reward'] = reward.get("instant_reward")
        info[f'{prefix}/ Overall/ Reward QoS'] = reward.get("reward_qos")
        info[f'{prefix}/ Overall/ Reward Power'] = reward.get("reward_power")
        info[f'{prefix}/ Overall/ Sum Packet loss rate'] = self.sum_packet_loss_rate
        info[f'{prefix}/ Overall/ Average rate/ Sub6GHz'] = self.average_rate[:,0].sum()/(self.num_devices)
        info[f'{prefix}/ Overall/ Average rate/ mmWave'] = self.average_rate[:,1].sum()/(self.num_devices)
        info[f'{prefix}/ Overall/ Average rate/ Global'] = info[f'{prefix}/ Overall/ Average rate/ mmWave'] + info[f'{prefix}/ Overall/ Average rate/ mmWave']
        info[f'{prefix}/ Overall/ Power usage'] = self.transmit_power.sum()
        
        for k in range(self.num_devices):
            info[f'{prefix}/ Device {k+1}/ Num. Sent packet/ Sub6GHz'] = self.num_send_packet[k,0]
            info[f'{prefix}/ Device {k+1}/ Num. Sent packet/ mmWave'] = self.num_send_packet[k,1]

            info[f'{prefix}/ Device {k+1}/ Num. Received packet/ Sub6GHz'] = self.num_received_packet[k,0]
            info[f'{prefix}/ Device {k+1}/ Num. Received packet/ mmWave'] = self.num_received_packet[k,1]
            
            info[f'{prefix}/ Device {k+1}/ Num. Droped packet/ Sub6GHz'] = self.num_send_packet[k,0] - self.num_received_packet[k,0]
            info[f'{prefix}/ Device {k+1}/ Num. Droped packet/ mmWave'] = self.num_send_packet[k,1] - self.num_received_packet[k,1]

            info[f'{prefix}/ Device {k+1}/ Power/ Sub6GHz'] = self.transmit_power[k,0]
            info[f'{prefix}/ Device {k+1}/ Power/ mmWave'] = self.transmit_power[k,1]

            info[f'{prefix}/ Device {k+1}/ Packet loss rate/ Global'] = self.global_packet_loss_rate[k]
            info[f'{prefix}/ Device {k+1}/ Packet loss rate/ Sub6GHz'] = self.packet_loss_rate[k,0]
            info[f'{prefix}/ Device {k+1}/ Packet loss rate/ mmWave'] = self.packet_loss_rate[k,1]
            info[f'{prefix}/ Device {k+1}/ Average rate/ Sub6GHz'] = self.average_rate[k,0]
            info[f'{prefix}/ Device {k+1}/ Average rate/ mmWave'] = self.average_rate[k,1]

            if hasattr(self, '_estimated_ideal_power'):
                info[f'{prefix}/ Device {k+1}/ Estimated ideal power/ Sub6GHz'] = self.estimated_ideal_power[k,0]/self.P_sum
                info[f'{prefix}/ Device {k+1}/ Estimated ideal power/ mmWave'] = self.estimated_ideal_power[k,1]/self.P_sum
        
        return info
    

    def step(self):
        self.current_step += 1


    def reset(self):
        self.current_step = 1
        self.average_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.num_send_packet = self._init_num_send_packet
        self.num_send_packet = self._init_num_received_packet
        self.transmit_power = self._init_transmit_power
        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        self.global_packet_loss_rate = np.zeros(shape=(self.num_devices))
        self.sum_packet_loss_rate = 0