import argparse
import os
from typing import Any, Callable, Dict, Optional, Tuple, List

import attrs

import numpy as np

import plotly.graph_objects as go

import wandb
from wandb.sdk.wandb_run import Run


@attrs.define
class Logger:
    """Weights and Biases logger that sends data to https://wandb.ai/."""

    project: Optional[str] = None
    name: Optional[str] = None
    entity: Optional[str] = None
    run_id: Optional[str] = None
    config: Optional[argparse.Namespace] = None
    monitor_gym: bool = True
    wandb_run: Run = attrs.field(init=False)

    @wandb_run.default
    def _wandb_run_factory(self):
        return (
            wandb.init(
                project=self.project,
                name=self.name,
                id=self.run_id,
                resume="allow",
                entity=self.entity,
                monitor_gym=self.monitor_gym,
                config=self.config,  # type: ignore,
            )
            if not wandb.run
            else wandb.run
        )

    def write(self, step: int, data: Dict[str, Any]) -> None:
        log_data: Dict = data.get("update_data", {})
        for _, cluster_data in data.get("clusters_data").items():
            cluster_data: List[Dict]
            cluster_data: Dict = cluster_data[0]  # Unpack batch
            prefix = next(iter(cluster_data.keys())).split("/")[0]

            num_sent_packet_acc = cluster_data.pop(
                f"{prefix}/ Accumulate/ Num. Sent packet"
            )
            num_received_packet_acc = cluster_data.pop(
                f"{prefix}/ Accumulate/ Num. Received packet"
            )
            fig = self.plot_interface_usage(
                num_sent_packet_acc,
                num_received_packet_acc,
            )

            cluster_data.update(
                {f"{prefix}/ Overall/ Interface Usage": wandb.Plotly(fig)}
            )

            log_data.update(cluster_data)

        self.wandb_run.log(log_data, step=step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

            checkpoint_artifact = wandb.Artifact(
                "run_" + self.wandb_run.id + "_checkpoint",  # type: ignore
                type="model",
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> Tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(  # type: ignore
            f"run_{self.wandb_run.id}_checkpoint:latest"  # type: ignore
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata["checkpoint_path"])
        )

        try:  # epoch / gradient_step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step

    def plot_interface_usage(
        self,
        num_sent_packet: np.ndarray,
        num_received_packet: np.ndarray,
        title: str = None,
    ):
        """
        Plot the interface usage as a Plotly stacked bar chart.
        Args:
            num_sent_packet (np.ndarray): Array of sent packets [num_devices, 2].
            num_received_packet (np.ndarray): Array of received packets [num_devices, 2].
            title (str, optional): Title of the plot.
        Returns:
            plotly.graph_objects.Figure
        """
        num_dropped_packet = num_sent_packet - num_received_packet
        num_devices = num_sent_packet.shape[0]
        x = [f"D{k+1}" for k in range(num_devices)]

        fig = go.Figure()

        # Sub-6GHz bars
        fig.add_trace(
            go.Bar(
                name="Successfully received on Sub-6GHz",
                x=x,
                y=num_received_packet[:, 0],
                marker_color="blue",
                offsetgroup="Sub-6GHz",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Dropped packet on Sub-6GHz",
                x=x,
                y=num_dropped_packet[:, 0],
                marker_color="red",
                offsetgroup="Sub-6GHz",
                base=num_received_packet[:, 0],
            )
        )

        # mmWave bars
        fig.add_trace(
            go.Bar(
                name="Successfully received on mmWave",
                x=x,
                y=num_received_packet[:, 1],
                marker_color="green",
                offsetgroup="mmWave",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Dropped packet (mmWave)",
                x=x,
                y=num_dropped_packet[:, 1],
                marker_color="red",
                offsetgroup="mmWave",
                base=num_received_packet[:, 1],
            )
        )

        # Layout
        fig.update_layout(
            barmode="relative",
            xaxis_title="Device",
            yaxis_title="Number of packets",
            bargap=0.3,  # spacing between groups
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            height=600,
            width=1000,
        )

        return fig
