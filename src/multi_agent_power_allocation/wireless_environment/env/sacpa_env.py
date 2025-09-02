from .base_env import WirelessEnvironmentBase
from ..wireless_communication_cluster import WirelessCommunicationCluster
import torch
from torch.nn.functional import softmax
import numpy as np
import gymnasium as gym
from typing import Dict, List
import attrs


class WirelessEnvironmentSACPA(WirelessEnvironmentBase):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    def reward_qos(self, agent:str) -> float:
        if not hasattr(self, "_reward_qos"):
            self._reward_qos : Dict[int, float] = {
                agent: 0.0 for agent in self.agents
            }

        return self._reward_qos[agent]


    def set_reward_qos(self, agent:str, value:float) -> None:
        self._reward_qos[agent] = value 
    
 
    def observation_space(self, agent:str):
        wcc_agent = self.wc_clusters[agent]
        num_devices = wcc_agent.num_devices
        L_max = wcc_agent.L_max

        return gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(num_devices)), # Quality of Service Satisfaction of each device on Sub6GHz
                np.zeros(shape=(num_devices)), # Quality of Service Satisfaction of each device on mmWave,
                np.zeros(shape=(num_devices)), # Number of received packets of each device on Sub6GHz of previous time step
                np.zeros(shape=(num_devices)), # Number of received packets of each device on mmWave of previous time step
                np.zeros(shape=(num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.zeros(shape=(num_devices)), # Average Rate of each device on mmWave of previous time step
                np.zeros(shape=(num_devices)), # Power of each device on Sub6GHz on previous time step
                np.zeros(shape=(num_devices)), # Power of each device on mmWave on previous time step
            ], dtype=np.float32).transpose().flatten(),
            high=np.array([
                np.ones(shape=(num_devices)), # Quality of Service Satisfaction of each device on Sub6GHz
                np.ones(shape=(num_devices)), # Quality of Service Satisfaction of each device on mmWave,
                np.full(shape=(num_devices), fill_value=L_max), # Number of received packets of each device on Sub6GHz of previous time step
                np.full(shape=(num_devices), fill_value=L_max), # Number of received packets of each device on mmWave of previous time step,
                np.ones(shape=(num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.ones(shape=(num_devices)), # Average Rate of each device on mmWave of previous time step,
                np.ones(shape=(num_devices)), # Power of each device on Sub6GHz of previous time step
                np.ones(shape=(num_devices)), # Power of each device on mmWave of previous time step,
            ], dtype=np.float32).transpose().flatten(),
        )
    

    def action_space(self, agent:str):
        wcc_agent = self.wc_clusters[agent]
        num_devices = wcc_agent.num_devices

        return gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.zeros(shape=(num_devices)), # Number of packets to send of each device on mmWave,
                np.zeros(shape=(num_devices)), # Power of each device on Sub6GHz,
                np.zeros(shape=(num_devices)), # Power of each device on mmWave,
            ], dtype=np.float32).flatten(),
            high=np.array([
                np.ones(shape=(num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.ones(shape=(num_devices)), # Number of packets to send of each device on mmWave,
                np.ones(shape=(num_devices)), # Power of each device on Sub6GHz,
                np.ones(shape=(num_devices)), # Power of each device on mmWave,
            ], dtype=np.float32).flatten(),
        )
    

    def reset(self, *, seed=None, options=None):
        for wcc_agent in self.wc_clusters:
            wcc_agent: str
            self.wc_clusters[wcc_agent].reset()

        observations = self.get_observations()
        infos = {}
        return observations, infos


    def compute_number_send_packet_and_power(self, wc_cluster:WirelessCommunicationCluster, policy_network_output:torch.Tensor) -> None:
        """
        Compute the number of packets to send and power for one cluster based on the its agent's policy network output.
        
        Parameters
        ----------
        policy_network_output : toch.Tensor
            Tensor containing the output from the policy network of its agent.
        
        Returns
        -------
        None
        """
        power_start_index = 2*wc_cluster.num_devices
        interface_score = policy_network_output[:power_start_index].reshape(wc_cluster.num_devices, 2)
        interface_score = torch.softmax(torch.tensor(interface_score), dim=1).numpy()

        number_of_send_packet = np.minimum(np.minimum(
            interface_score*wc_cluster.L_max,
            wc_cluster.l_max_estimate,
        ).astype(int), wc_cluster.L_max)

        power = policy_network_output[power_start_index:]
        power = torch.softmax(torch.tensor(power), dim=-1).numpy()
        power = power.reshape(wc_cluster.num_devices, 2)

        for k in range(wc_cluster.num_devices):
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] == 0: # Force to send at least one packet on more reliable channel
                if wc_cluster.packet_loss_rate[k,0] < wc_cluster.packet_loss_rate[k,1]:
                    number_of_send_packet[k,0] = 1
                else:
                    number_of_send_packet[k,1] = 1
            
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] > wc_cluster.L_max:
                # If the number of packets to send exceeds the maximum number of packets that can be sent
                # then send on both channels by the proportion of the packet success rate
                if np.sum(wc_cluster.packet_loss_rate[k]) == 0:
                    psr_proportion = 0.5
                else:
                    psr_proportion = 1 - wc_cluster.packet_loss_rate[k,0]/np.sum(wc_cluster.packet_loss_rate[k])
                number_of_send_packet[k,0] = np.floor(psr_proportion*wc_cluster.L_max)
                number_of_send_packet[k,1] = wc_cluster.L_max - number_of_send_packet[k,0]
            
            # Send the remaining power to the other channel
            if number_of_send_packet[k,0] == 0:
                power[k,1] += power[k,0]
                power[k,0] = 0
            if number_of_send_packet[k,1] == 0:
                power[k,0] += power[k,1]
                power[k,1] = 0

        wc_cluster.set_num_send_packet(number_of_send_packet)
        wc_cluster.set_transmit_power(power)


    def _compute_action(self, agent:str ,policy_network_output):
        """
        Compute action for one agent
        """
        wc_cluster = self.wc_clusters[agent]
        wc_cluster.estimate_l_max()
        self.compute_number_send_packet_and_power(wc_cluster, policy_network_output)
        wc_cluster.update_allocation()
        wc_cluster.update_signal_power()


    def compute_actions(self, policy_network_outputs):
        """
        Compute actions accross all agents
        """
        actions = {}
        for agent in self.agents:
            self._compute_action(agent, policy_network_outputs[agent])


    def _update_feedback(self, agent:str):
        """
        Compute number of received packet at devices side of one agent (wireless communication cluster)
        """
        wc_cluster = self.wc_clusters[agent]

        interference = np.zeros_like(wc_cluster.signal_power)

        for other_agent in self.agents:
            other_agent : str
            if other_agent != agent:
                interference += self.wc_clusters[other_agent].signal_power

        wc_cluster.update_feedback(interference=interference)
        wc_cluster.update_average_rate()
        wc_cluster.update_packet_loss_rate()


    def get_feedbacks(self):
        """
        Compute number of received packet at devices side across all wireless communication cluster.
        This function updates the feedback and average rate for each cluster.
        """
        for agent in self.agents:
            self._update_feedback(agent)


    def _compute_rewards(self, agent:str) -> Dict[str,float]:
        """
        Compute reward for one agent

        Parameters
        ----------
        agent : str
            Agent id
        
        Returns
        -------
        reward : Dict[str, float]
            Dictionary contains reward components and value of reward sum
        """
        wc_cluster = self.wc_clusters[agent]

        def estimate_ideal_power(num_send_packet, channel_power, W):
            if channel_power==0:
                return wc_cluster.P_sum
            
            ideal_power = (2**((num_send_packet*wc_cluster.D)/(W*wc_cluster.T)) - 1) * \
                        W*wc_cluster.Sigma_sqr/channel_power
            return min(ideal_power, wc_cluster.P_sum)
        
        reward_qos = 0
        reward_power = 0
        target_power = []
        predicted_power = []

        for k in range(wc_cluster.num_devices):
            # Unit: percentage
            transmit_power = \
                wc_cluster.transmit_power[k, 0], \
                wc_cluster.transmit_power[k, 1]
            
            channel_power_gain = \
                wc_cluster.channel_power_gain[k, 0] , \
                wc_cluster.channel_power_gain[k, 1]

            qos_satisfaction = \
                wc_cluster.packet_loss_rate[k, 0] < wc_cluster.qos_threshold , \
                wc_cluster.packet_loss_rate[k, 1] < wc_cluster.qos_threshold

            num_received_packet = \
                wc_cluster.num_received_packet[k, 0] , \
                wc_cluster.num_received_packet[k, 1]
            
            num_send_packet = \
                wc_cluster.num_send_packet[k, 0] , \
                wc_cluster.num_send_packet[k, 1]
            
            reward_qos += (num_received_packet[0] + num_received_packet[1])/(num_send_packet[0] + num_send_packet[1]) - (1 - qos_satisfaction[0]) - (1 - qos_satisfaction[1])

            if num_send_packet[0] > 0:
                wc_cluster.estimated_ideal_power[k, 0] = estimate_ideal_power(num_send_packet[0], channel_power_gain[0], wc_cluster._W_sub)
                target_power.append(wc_cluster.estimated_ideal_power[k, 0])
                predicted_power.append(transmit_power[0])

            if num_send_packet[1] > 0:
                wc_cluster.estimated_ideal_power[k, 1] = estimate_ideal_power(num_send_packet[1], channel_power_gain[1], wc_cluster._W_mw)
                target_power.append(wc_cluster.estimated_ideal_power[k, 1])
                predicted_power.append(transmit_power[1])

        target_power = torch.tensor(target_power)
        predicted_power = torch.tensor(predicted_power)

        target_power = softmax(target_power, dim=-1)
        reward_power = -wc_cluster.num_devices*(target_power*(target_power.log()-predicted_power.log())).sum().numpy()
        reward_qos = ((self.current_step-1)*self.reward_qos(agent) + reward_qos)/self.current_step

        self.set_reward_qos(agent, reward_qos)
        instance_reward = self.reward_coef['reward_qos']*reward_qos + self.reward_coef['reward_power']*reward_power
        
        return {
            "reward_qos": reward_qos, 
            "reward_power": reward_power, 
            "instant_reward": instance_reward
        }
        

    def get_rewards(self) -> Dict[int, Dict[str, float]]:
        rewards = {} 
        for agent in self.agents:
            agent : str
            reward = self._compute_rewards(agent)
            rewards.update({agent: reward})

        return rewards

            
    def _get_state(self, agent:str) -> np.ndarray:
        """
        """
        wc_cluster = self.wc_clusters[agent]

        _state = np.zeros(shape=(wc_cluster.num_devices, 8))
        # QoS satisfaction
        _state[:, 0] = (wc_cluster.packet_loss_rate[:, 0] <= wc_cluster.qos_threshold).astype(float)
        _state[:, 1] = (wc_cluster.packet_loss_rate[:, 1] <= wc_cluster.qos_threshold).astype(float)
        _state[:, 2] = wc_cluster.num_received_packet[:, 0].copy()
        _state[:, 3] = wc_cluster.num_received_packet[:, 1].copy()
        _state[:, 4] = wc_cluster.average_rate[:, 0]/wc_cluster.maximum_rate[0]
        _state[:, 5] = wc_cluster.average_rate[:, 1]/wc_cluster.maximum_rate[1]
        _state[:, 6] = wc_cluster.transmit_power[:, 0].copy()*10.0 # Scale up
        _state[:, 7] = wc_cluster.transmit_power[:, 1].copy()*10.0
        
        return _state
    

    def get_observations(self) -> Dict[int, np.ndarray]:
        observations = {}
        
        for agent in self.agents:
            agent : str
            
            observations.update({agent: self._get_state(agent).flatten()})

        return observations
    

    def get_infos(self, rewards:Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        infos = {}

        for agent in self.agents:
            agent : str

            wc_cluster = self.wc_clusters[agent]
            agent_reward = rewards.get(agent)
            infos.update({agent: wc_cluster.get_info(agent_reward)})

        return infos


    def step(self, actions):
        observations = {}
        terminations = {
            agent: False for agent in self.agents
        }
        truncations = {
            agent: False for agent in self.agents
        }
        infos = {}
        
        policy_network_outputs = {
            agent: actions[agent] for agent in self.agents
        }

        self.compute_actions(policy_network_outputs=actions)
        self.get_feedbacks()

        for wc_cluster in self.wc_clusters:
            self.wc_clusters[wc_cluster].step()

        _rewards = self.get_rewards()
        rewards = {
            agent: _rewards.get(agent).get("instant_reward") for agent in _rewards.keys()
        }

        observations = self.get_observations()
        
        infos = self.get_infos(_rewards)

        self.current_step += 1
        if self.current_step > self.max_num_step + 1:
            truncations = {
                agent: True for agent in self.agents
            }
        
        return observations, rewards, terminations, truncations, infos

