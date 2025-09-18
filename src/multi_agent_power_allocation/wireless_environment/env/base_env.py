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
)
from multi_agent_power_allocation.wireless_environment.constants import MAP_SIZE


@attrs.define
class WirelessEnvironmentBase(ParallelEnv):
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

    def __attrs_post_init__(self):
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        self.agents: List[str] = [str(i) for i in range(self.num_cluster)]
        self.possible_agents = self.agents[:]

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
                        )
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
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_number_send_packet_and_power(
        self,
        wc_cluster: WirelessCommunicationCluster,
        policy_network_output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _compute_action(self, agent: str, policy_network_output):
        """
        Compute action for one agent
        """
        wc_cluster = self.wc_clusters[agent]
        wc_cluster.estimate_l_max()
        self.compute_number_send_packet_and_power(wc_cluster, policy_network_output)
        wc_cluster.update_allocation()
        wc_cluster.update_signal_power()  # Must be updated after allocation

    def compute_actions(self, policy_network_outputs):
        """
        Compute actions accross all agents
        """
        for agent in self.agents:
            self._compute_action(agent, policy_network_outputs[agent])

    def _update_feedback(self, agent: str):
        """
        Compute number of received packet at devices side of one agent (wireless communication cluster)
        """
        wc_cluster = self.wc_clusters[agent]

        interference = np.zeros_like(wc_cluster.signal_power)

        for other_agent in self.agents:
            other_agent: str
            if other_agent != agent:
                interference += self.wc_clusters[other_agent].signal_power

        wc_cluster.update_feedback(interference=interference)
        wc_cluster.update_packet_loss_rate()
        wc_cluster.update_average_rate()

    def get_feedbacks(self):
        """
        Compute number of received packet at devices side across all wireless communication cluster.
        This function updates the feedback and average rate for each cluster.
        """
        for agent in self.agents:
            self._update_feedback(agent)

    def _compute_rewards(self, agent: str) -> Dict[str, float]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_rewards(self) -> Dict[int, Dict[str, float]]:
        rewards = {}
        for agent in self.agents:
            agent: str
            reward = self._compute_rewards(agent)
            rewards.update({agent: reward})

        return rewards

    def _get_state(self, agent: str) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_observations(self) -> Dict[int, np.ndarray]:
        observations = {}

        for agent in self.agents:
            agent: str

            observations.update({agent: self._get_state(agent).flatten()})

        return observations

    def get_infos(
        self, rewards: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, float]]:
        infos = {}

        for agent in self.agents:
            agent: str

            wc_cluster = self.wc_clusters[agent]
            agent_reward = rewards.get(agent)
            infos.update({agent: wc_cluster.get_info(agent_reward)})

        return infos

    def step(self, actions):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def observation_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def action_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def render(self):
        mode = self.render_mode

        if self.closed:
            return

        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # left half, right half

        # === LEFT HALF: Positions ===
        ax_pos = fig.add_subplot(gs[0, 0])
        ax_pos.set_title("Access Points and IoT Devices")
        ax_pos.set_xlabel("X")
        ax_pos.set_ylabel("Y")
        ax_pos.set_xlim(-MAP_SIZE[0] / 2, MAP_SIZE[0] / 2)
        ax_pos.set_ylim(-MAP_SIZE[1] / 2, MAP_SIZE[1] / 2)

        # Plot each cluster
        colors = plt.get_cmap("tab10", self.num_cluster)
        for idx, (cid, cluster) in enumerate(self.wc_clusters.items()):
            ap_x, ap_y = cluster.AP_position
            dev_positions = np.array(cluster.device_positions)

            # Plot AP
            ax_pos.scatter(
                ap_x, ap_y, c=[colors(idx)], marker="^", s=200, label=f"AP {cid}"
            )

            # Plot devices
            ax_pos.scatter(
                dev_positions[:, 0],
                dev_positions[:, 1],
                c=[colors(idx)],
                marker="o",
                alpha=0.7,
                label=f"Cluster {cid} Devices",
            )

            # Plot obstacles
            for pos in cluster.obstacle_positions:
                start_point, end_point = pos

                # Extract x and y coordinates
                x_coords = [start_point[0], end_point[0]]
                y_coords = [start_point[1], end_point[1]]

                # Plot the line segment
                ax_pos.plot(x_coords, y_coords, "k-", linewidth=4)

        ax_pos.plot([], [], "k-", linewidth=2, label="Obstacles")
        ax_pos.legend(loc="upper left")
        ax_pos.grid(True, linestyle="--", alpha=0.5)

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
                color=colors(idx),
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
                colors=[colors(idx)] * len(power_alloc),
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
