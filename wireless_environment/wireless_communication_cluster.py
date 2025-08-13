"""
Wireless Communication Cluster Module
This module defines the base class for wireless communication cluster, each cluster represents a group of one Access Point (AP) serves K IoT devices through wireless communication.
"""
import numpy as np
import attrs
from .utils import compute_rate, compute_h_sub, compute_h_mW

@attrs.define
class WirelessCommunicationCluster:
    """
    Base class for wireless communication clusters.
    This class is designed to be extended by specific wireless communication cluster implementations.
    """
    cluster_id: int = attrs.field(
        default=0,
        metadata={"description": "Unique identifier for the cluster."}
    )

    h_tilde: np.ndarray = attrs.field(
        metadata={"description": "Channel state information matrix."}
    )

    AP_position: np.ndarray = attrs.field(
        default=np.array([0.0, 0.0]),
        metadata={"description": "Position of the Access Point (AP) in the cluster."}
    )

    device_positions: np.ndarray = attrs.field(
        metadata={"description": "Positions of devices in the cluster."}
    )

    device_blockages: list = attrs.field(
        default=attrs.Factory(list),
        metadata={"description": "Hash array indicating whether each device is blocked by obstacles."}
    )

    LOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Line-of-sight path loss for mmWave connections."}
    )

    NLOS_PATH_LOSS: np.ndarray = attrs.field(
        metadata={"description": "Non-line-of-sight path loss for mmWave connections."}
    )

    L_max: int = attrs.field(
        default=1,
        metadata={"description": "Maximum number of packet AP send to each device."}
    )

    P_sum: float = attrs.field(
        default=1.0,
        metadata={"description": "Total transmision power available for the cluster."}
    )

    D: int = attrs.field(
        default=1000,
        metadata={"description": "Size of one packet in bit."}
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

    def __attrs_post_init__(self):
        """
        Post-initialization method to validate the cluster configuration.
        """
        self._W_sub = self.W_sub_total / self.n_sub_channels
        self._W_mw = self.W_mw_total / self.n_beams
        self._num_devices = self.device_positions.shape[0]
        self.num_sub_channel = self.h_tilde.shape[-1] # implicitly defined
        self.num_beam = self.h_tilde.shape[-1]

        self.distance_to_AP = np.linalg.norm(
            self.device_positions - self.AP_position, axis=1
        )

        self.current_step = 1

        self._init_num_send_packet:np.ndarray = attrs.field(
            default=np.ones(shape=(self._num_devices, 2)),
            metadata={"description": "Initial number of packets sent to each device in the cluster at the start of each episode."}
        )

        self._init_power:np.ndarray = attrs.field(
            default=np.ones(shape=(self._num_devices, 2)),
            metadata={"description": "Initial power levels for each device in the cluster at the start of each episode."}
        )

        self._init_allocation:np.ndarray = attrs.field(
            converter=self.allocate(self._init_num_send_packet),
            metadata={"description": "Initial subchannel/beam allocation for each device in the cluster at the start of each episode."}
        )

        self.channel_power_gain = np.zeros(shape=(self._num_devices, 2))

        self._init_rate:np.ndarray = attrs.field(
            converter=self.compute_instant_rate(allocation=self._init_allocation, power=self._init_power),
            metadata={"description": "Initial rate for each device in the cluster at the start of each episode."}
        )
        
        self.average_rate = self._init_rate.copy()
        self.previous_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()

        self.maximum_rate:np.ndarray = attrs.field(
            default=np.array([
                compute_rate(w=self._W_sub, h=1.0,  power=self.P_sum, interference=0.0, noise=self.Sigma_sqr),
                compute_rate(w=self._W_mw, h=1.0,  power=self.P_sum, interference=0.0, noise=self.Sigma_sqr)
            ]),
            metadata={"description": "Maximum rate for each device in the cluster, for normalizing rate to [0, 1]."}
        )

        self.interference:np.ndarray = np.zeros(shape=(2, max(self.num_sub_channel, self.num_beam)))

        self._estimated_ideal_power = np.zeros(shape=(self._num_devices, 2))

        self.P_sum = np.ones(shape=(self._num_devices, 2))


    def allocate(self, num_send_packet:np.ndarray) -> np.ndarray:
        """
        Allocate subchannels and beams to devices randomly based on the number of packets to be sent.

        Parameters
        ----------
        num_send_packet : np.ndarray
            Array of shape (num_devices, 2) representing the number of packets to be sent to each device.

        Returns
        -------
        allocation : np.ndarray
            Hash array of shape (num_devices, 2) representing the allocated subchannels and beams for each device.
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
            if (num_send_packet[k,0]>0 and num_send_packet[k,1]==0):
                rand_index = np.random.randint(0,len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
            elif (num_send_packet[k,0]==0 and num_send_packet[k,1]>0):
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

        allocation = np.array([sub, mW]).transpose()
        return allocation


    def compute_instant_rate(self, allocation:np.ndarray, power:np.ndarray, h_tilde:np.ndarray, interference:np.ndarray) -> np.ndarray:
        """
        Compute the instantaneous rate for each device based on the current allocation and power levels.

        Parameters
        ----------
        allocation : np.ndarray
            Array of shape (num_devices, 2) representing the allocated subchannels and beams for each device.
        power : np.ndarray
            Array of shape (num_devices, 2) representing the power levels for each device.
        h_tilde : np.ndarray
            Array of shape (2, num_devices, num_subchannel or num_beam) representing the complex channel coefficient at one time step.
        interference : np.ndarray
            Array of shape (2, num_subchannels or num_beams) representing the interference at each subchannel/beam

        Returns
        -------
        instant_rate : np.ndarray
            Array of shape (num_devices, 2) representing the instantaneous rate for each device.

        Side effects
        -------
        Also compute and write on self.interference
        """
        rate = np.zeros(shape=(self._num_devices, 2))
        self.interference = np.zeros_like(self.interference)
        
        for k in range(self._num_devices):
            sub_channel_index = allocation[k, 0]
            mW_beam_index = allocation[k, 1]
            if (sub_channel_index != -1):
                self.channel_power_gain[k, 0] = compute_h_sub(
                    distance_to_AP=self.distance_to_AP[k],
                    h_tilde=self.h_tilde[0, k, sub_channel_index]
                )

                p = power[k,0]*self.P_sum
                rate[k,0], self.interference[0,sub_channel_index] = compute_rate(
                    w=self._W_sub,
                    h=self.channel_power_gain[k, 0], p=p,
                    interference=interference[0, sub_channel_index],
                    noise=self.Sigma_sqr,
                    return_power=True
                )

            if (mW_beam_index != -1):
                is_blocked = self.device_blockages[k]
                x = self.NLOS_PATH_LOSS[self.current_step, k] if is_blocked else self.LOS_PATH_LOSS[self.current_step, k]
                self.channel_power_gain[k, 1] = compute_h_mW(
                    distance_to_AP=self.distance_to_AP[k],
                    is_blocked=is_blocked,
                    x=x
                )

                p = power[k,1]*self.P_sum
                rate[k,1], self.interference[1,mW_beam_index] = compute_rate(
                    w=self._W_mw,
                    h=self.channel_power_gain[k, 1],
                    p=p,
                    interference=interference[1, mW_beam_index],
                    noise=self.Sigma_sqr,
                    return_power=True
                )

        return rate
    

    def compute_average_rate(self, current_step:int) -> np.ndarray:
        """
        
        """
        average_rate = 1/current_step*(self.rate + self.average_rate*(current_step-1))

        return average_rate


    def compute_packet_loss_rate(self, num_received_packet:np.ndarray, num_send_packet:np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        '''
        Returns devices packet loss rate on each interfaces, devices packet loss rate on the whole, and system packet loss rate
        '''
        packet_loss_rate = np.zeros(shape=(self._num_devices, 2))
        global_packet_loss_rate = np.zeros(shape=(self._num_devices))
        for k in range(self._num_devices):
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

        return packet_loss_rate, global_packet_loss_rate, sum_packet_loss_rate    
    

    def get_feedback(self, allocation:np.ndarray, num_sent_packet:np.ndarray, power:np.ndarray) -> np.ndarray:
        """
        Compute the number of received packet at device side

        Parameters
        ----------
        allocation : np.ndarray
            Array of shape (_num_devices, 2) representing the allocated subchannels and beams for each device.
        num_sent_packet : np.ndarray
            Array of shape (num_devices, 2)
            representing the number of packet AP sent to devices
        power : np.ndarray
            Array of shape (num_devices, 2) representing the power levels for each device.
        

        Returns
        -------
        instant_rate : np.ndarray
            Array of shape (num_devices, 2) representing the instantaneous rate for each device.
        """
        self.rate = self.compute_instant_rate(allocation, power)
        l_max = np.floor(np.multiply(self.rate, self.T/self.D))

        num_received_packet = np.minimum(num_sent_packet, l_max)
        
        self.packet_loss_rate, self.global_packet_loss_rate, self.sum_packet_loss_rate = self.compute_packet_loss_rate(num_received_packet, num_sent_packet)

        return num_received_packet