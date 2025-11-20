from typing import Dict, Any, List
import random
import attrs

from pettingzoo import ParallelEnv

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import pygame
from pygame import Surface

from multi_agent_power_allocation.wireless_environment.wireless_communication_cluster import (
    WirelessCommunicationCluster,
    compute_h_sub,
)
from multi_agent_power_allocation.utils.plot import plot_positions
from multi_agent_power_allocation.algorithms.high_level import Algorithm, Reward
from multi_agent_power_allocation.algorithms.algorithm_register import Algorithms


@attrs.define
class WirelessEnvironment(ParallelEnv):
    """
    Base class for wireless environments in PettingZoo API.
    This class is designed to be extended by specific wireless environment implementations.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render.fps": 24,
        "name": "wireless_environment_base",
        "is_parallelizable": True,
    }

    algorithm_mapping: Dict[str, Algorithm]  # {cluster_id: Algorithm}
    reward_coef: Dict[str, float]
    wc_clusters_configs: List[Dict[str, Any]]
    n_warm_up_step: int = attrs.field()
    num_cluster: int = attrs.field(default=2, kw_only=True)
    max_num_step: int = attrs.field(default=10_000)
    current_step: int = attrs.field(default=1)
    seed: int = attrs.field(default=None)
    render_mode: str = attrs.field(default=None)
    window: Surface = attrs.field(default=None, init=False)
    clock: pygame.time.Clock = attrs.field(default=None, init=False)
    closed: bool = attrs.field(default=False, init=False)
    wc_clusters: Dict[str, WirelessCommunicationCluster] = attrs.field(
        default={}, init=False
    )
    reward_qos: Dict[str, float] = attrs.field(init=False)

    def __attrs_post_init__(self):
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        self.agents = list(self.algorithm_mapping.keys())
        self.possible_agents = self.agents[:]
        self.reward_qos = {agent: 0.0 for agent in self.agents}

        for i in range(self.num_cluster):
            if not (
                self.wc_clusters_configs[i].get("LOS_PATH_LOSS")
                and self.wc_clusters_configs[i].get("NLOS_PATH_LOSS")
            ):
                num_devices = self.wc_clusters_configs[i].get("num_devices")
                self.wc_clusters_configs[i].update(
                    {
                        "LOS_PATH_LOSS": np.random.normal(
                            0, 5.8, size=(self.max_num_step + 1, num_devices)
                        )  # TODO: different seed for different cluster
                    }
                )
                self.wc_clusters_configs[i].update(
                    {
                        "NLOS_PATH_LOSS": np.random.normal(
                            0, 8.7, size=(self.max_num_step + 1, num_devices)
                        )
                    }
                )

            self.wc_clusters.update(
                {
                    self.agents[i]: WirelessCommunicationCluster(
                        cluster_id=i, **self.wc_clusters_configs[i]
                    )
                }
            )

    def reset(self, seed=None, options=None):
        for wc_cluster in self.wc_clusters.values():
            wc_cluster.reset()

        observations = self.get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def compute_number_send_packet_and_power(
        self,
        agent: str,
        policy_network_output: torch.Tensor,
    ) -> None:
        wc_cluster = self.wc_clusters[agent]
        algorithm = self.algorithm_mapping[agent]

        algorithm.compute_number_send_packet_and_power(
            wc_cluster, policy_network_output
        )

    def _compute_action(self, agent: str, policy_network_output):
        """
        Compute action for one agent
        """
        wc_cluster = self.wc_clusters[agent]
        algorithm = self.algorithm_mapping[agent]
        wc_cluster.estimate_l_max(algorithm)
        self.compute_number_send_packet_and_power(agent, policy_network_output)
        wc_cluster.update_allocation()
        wc_cluster.update_signal_power()  # Must be updated after allocation

    def compute_actions(self, policy_outputs):
        """
        Compute actions accross all agents
        """
        for agent in self.agents:
            self._compute_action(agent, policy_outputs[agent])

    def _update_feedback(self, agent: str):
        """
        Compute number of received packet at devices side of one agent (wireless communication cluster)
        """
        wc_cluster = self.wc_clusters[agent]

        interference = np.zeros_like(wc_cluster.signal_power)

        for other_agent in self.agents:
            other_agent: str

            if other_agent != agent:
                other_wcc = self.wc_clusters[other_agent]

                for other_device, allocation in enumerate(other_wcc.allocation):
                    subchannel, _ = allocation

                    if subchannel != -1:
                        device_indice = np.where(
                            wc_cluster.allocation[:, 0] == subchannel
                        )  # find which device of wc_cluster use this sub channel
                        if device_indice[0].size > 0:  # found
                            assert (
                                device_indice[0].size <= 1
                            ), f"There are more than one device using sub-channel {subchannel}! Devices: {device_indice}."
                            device = device_indice[0][0]

                            interference_h = compute_h_sub(
                                distance_to_AP=np.linalg.norm(
                                    wc_cluster.device_positions[device]
                                    - other_wcc.AP_position
                                ),
                                h_tilde=other_wcc.h_tilde[
                                    wc_cluster.cluster_id,
                                    self.current_step,
                                    0,
                                    device,
                                    subchannel,
                                ],
                            )

                            interference_transmit_power = (
                                other_wcc.transmit_power[other_device, 0]
                                * other_wcc.P_sum
                            )

                            interference[0, subchannel] += (
                                interference_h * interference_transmit_power
                            )

        wc_cluster.update_feedback(interference=interference)
        wc_cluster.update_packet_loss_rate()
        wc_cluster.update_packet_loss_rate_stacked()
        wc_cluster.update_average_rate()
        wc_cluster.update_average_rate_stacked()

    def get_feedbacks(self):
        """
        Compute number of received packet at devices side across all wireless communication cluster.
        This function updates the feedback and average rate for each cluster.
        """
        for agent in self.agents:
            self._update_feedback(agent)

    def _compute_rewards(self, agent: str) -> Reward:
        algorithm = self.algorithm_mapping[agent]
        wc_cluster = self.wc_clusters[agent]

        rewards = algorithm.compute_reward(
            wc_cluster, self.reward_qos[agent], self.reward_coef
        )

        self.reward_qos[agent] = rewards.reward_components["reward_qos"]

        return rewards

    def get_rewards(self) -> Dict[int, Reward]:
        rewards = {}
        for agent in self.agents:
            agent: str
            reward = self._compute_rewards(agent)
            rewards.update({agent: reward})

        return rewards

    def _get_state(self, agent: str) -> np.ndarray:
        algorithm = self.algorithm_mapping[agent]
        wc_cluster = self.wc_clusters[agent]

        state = algorithm.get_state(wc_cluster)

        if np.isnan(state).any():
            raise ValueError(f"Env produced NaN state: {state}")

        return state

    def get_observations(self) -> Dict[int, np.ndarray]:
        observations = {}

        for agent in self.agents:
            agent: str

            observations.update({agent: self._get_state(agent).flatten()})

        return observations

    def get_infos(
        self, rewards: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        infos = {}

        for agent in self.agents:
            agent: str

            wc_cluster = self.wc_clusters[agent]
            agent_reward = rewards.get(agent)
            infos.update({agent: wc_cluster.get_info(agent_reward)})

        return infos

    def state(self):
        states = []
        for agent in self.agents:
            states.append(self._get_state(agent))

        return np.array(states)

    def estimate_CGINR(self):
        for agent in self.agents:
            algorithm = self.algorithm_mapping[agent]
            if isinstance(algorithm, Algorithms.SACPA.value):
                self.wc_clusters[agent].estimate_CGINR()

    def step(self, actions: torch.Tensor | np.ndarray):
        """
        Parameters
        ----------
            actions: torch.Tensor | np.ndarray
                Comes directly from BaseAlgorithm.get_actions()
        """
        observations = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {}

        self.compute_actions(policy_outputs=actions)
        self.get_feedbacks()
        self.estimate_CGINR()

        for wc_cluster in self.wc_clusters.values():
            wc_cluster.step()

        _rewards = self.get_rewards()
        rewards = {agent: _rewards.get(agent).reward_sum for agent in _rewards}

        observations = self.get_observations()

        infos = self.get_infos(_rewards)

        self.current_step += 1
        if self.current_step > self.max_num_step + 1:
            truncations = {agent: True for agent in self.agents}

        if self.current_step == 2000:
            AP_positions = [
                wc_cluster.AP_position for wc_cluster in self.wc_clusters.values()
            ]
            for wc_cluster in self.wc_clusters.values():
                wc_cluster.change_obstacle_positions(AP_positions)

        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        algorithm = self.algorithm_mapping[agent]
        num_devices = self.wc_clusters[agent].num_devices
        L_max = self.wc_clusters[agent].L_max
        return algorithm.observation_space(num_devices, L_max)

    def action_space(self, agent):
        algorithm = self.algorithm_mapping[agent]
        num_devices = self.wc_clusters[agent].num_devices
        return algorithm.action_space(num_devices)

    def render(self):
        mode = self.render_mode

        if self.closed:
            return

        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # left half, right half

        # === LEFT HALF: Positions ===
        ax_pos = fig.add_subplot(gs[0, 0])

        colormap = plt.get_cmap("tab10")
        colors = [colormap(i) for i in range(self.num_cluster)]

        positions: List[Dict] = []
        for idx, (cid, cluster) in enumerate(self.wc_clusters.items()):
            positions.append(
                {
                    "ID": cluster.cluster_id,
                    "AP": cluster.AP_position,
                    "devices": cluster.device_positions,
                    "obstacles": cluster.obstacle_positions,
                }
            )
        plot_positions(ax_pos, positions, colors)

        # === RIGHT HALF: Stats per cluster ===
        outer_gs = gs[0, 1].subgridspec(self.num_cluster, 1)  # vertical split

        for idx, (cid, cluster) in enumerate(self.wc_clusters.items()):
            inner_gs = outer_gs[idx].subgridspec(1, 2)  # split into 2 halves

            # LEFT: Bar chart (packets per device)
            ax_bar = fig.add_subplot(inner_gs[0, 0])
            packets = cluster.num_send_packet.sum(axis=1)
            ax_bar.bar(
                range(len(packets)),
                packets,
                color=colors[idx],
                tick_label=[f"Device {i+1}" for i in range(cluster.num_devices)],
            )
            ax_bar.set_title(f"Cluster {cid} Num. Sent Packets")
            ax_bar.set_xlabel("Device ID")
            ax_bar.set_ylabel("Packets")

            # RIGHT: Pie chart (transmit power)
            ax_pie = fig.add_subplot(inner_gs[0, 1])
            power_alloc = cluster.transmit_power.sum(axis=1)
            ax_pie.pie(
                power_alloc,
                labels=[f"D{i}" for i in range(len(power_alloc))],
                autopct="%1.1f%%",
                colors=[colors[idx]] * len(power_alloc),
            )
            ax_pie.set_title(f"Cluster {cid} Power")

        plt.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close()

        if mode == "human":
            img = canvas.buffer_rgba()
            size = canvas.get_width_height()

            if self.window is None:
                pygame.init()  # pylint:disable=no-member
                self.clock = pygame.time.Clock()

                window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
                self.window = pygame.display.set_mode(window_size)
                pygame.display.set_icon(Surface((0, 0)))
                pygame.display.set_caption("WirelessEnvironment")

            self.window.fill("white")
            screen = pygame.display.get_surface()
            plot = pygame.image.frombuffer(img, size, "RGBA")
            screen.blit(plot, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # pylint:disable=no-member
                    self.close()

        elif mode == "rgb_array":
            img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            return img

    def close(self):
        """Closes the environment and terminates its visualization."""
        pygame.quit()  # pylint:disable=no-member
        self.window = None
        self.closed = True
