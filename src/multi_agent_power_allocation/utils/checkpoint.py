"""
Model checkpointing: periodic/best/final local saves, optional upload to WandB artifacts
and to the Hugging Face Hub.
"""

import os
import re
import json
import shutil
from typing import Any, Dict, Optional

import attrs
import yaml

import numpy as np

import torch

from multi_agent_power_allocation.algorithms.high_level import Algorithm


DEFAULT_CHECKPOINT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    # Root folder, checkpoints of a run go to `<save_dir>/<run name>/`
    "save_dir": "checkpoints",
    # Save every `save_freq` environment steps (0 or null disables periodic saves)
    "save_freq": 5000,
    # Number of periodic checkpoints to keep on disk (older ones are deleted)
    "keep_last": 2,
    # Keep a copy of the checkpoint with the highest mean reward over the last `save_freq` steps
    "save_best": True,
    # Upload the final checkpoint folder as a WandB artifact
    "upload_to_wandb": False,
    # Push the final checkpoint folder to the Hugging Face Hub
    "push_to_hub": False,
    "hf_repo_id": None,  # e.g. "<user>/sacra-power-allocation"
    "hf_private": True,
    # Also push every periodic checkpoint (useful on preemptible machines such as Kaggle)
    "hf_push_every_save": False,
    # Name of the env var holding the HF token (falls back to the cached `huggingface-cli login`)
    "hf_token_env": "HF_TOKEN",
}


def resolve_checkpoint_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    resolved = dict(DEFAULT_CHECKPOINT_CONFIG)
    resolved.update(config or {})
    if resolved["push_to_hub"] and not resolved["hf_repo_id"]:
        raise ValueError("`checkpoint_config.hf_repo_id` is required when `push_to_hub` is true")
    return resolved


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def save_policies(
    path: str,
    policies: Dict[str, Algorithm],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(path, exist_ok=True)
    for agent_id, policy in policies.items():
        torch.save(
            {
                "algorithm": type(policy).__name__,
                "low_level_algorithm": type(policy.low_level_algorithm).__name__,
                "state_dict": policy.low_level_algorithm.state_dict(),
            },
            os.path.join(path, f"agent_{agent_id}.pt"),
        )
    with open(os.path.join(path, "metadata.json"), "wt", encoding="utf-8") as f:
        json.dump(metadata or {}, f, indent=2, default=str)


def load_policies(path: str, policies: Dict[str, Algorithm]) -> Dict[str, Any]:
    """
    Restore policies (built from the same config) in place from a checkpoint folder.
    Returns the checkpoint metadata.
    """
    for agent_id, policy in policies.items():
        checkpoint = torch.load(
            os.path.join(path, f"agent_{agent_id}.pt"), weights_only=False
        )
        if checkpoint["algorithm"] != type(policy).__name__:
            raise ValueError(
                f"Agent {agent_id}: checkpoint holds {checkpoint['algorithm']}, "
                f"but the policy is {type(policy).__name__}"
            )
        policy.low_level_algorithm.load_state_dict(checkpoint["state_dict"])

    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "rt", encoding="utf-8") as f:
        return json.load(f)


@attrs.define
class Checkpointer:
    policies: Dict[str, Algorithm]
    run_name: str
    config: Dict[str, Any] = attrs.field(converter=resolve_checkpoint_config)
    run_config: Optional[Dict[str, Any]] = None
    wandb_run: Any = None
    run_dir: str = attrs.field(init=False)
    _rewards: list = attrs.field(init=False, factory=list)
    _saved: list = attrs.field(init=False, factory=list)
    _best_score: float = attrs.field(init=False, default=-np.inf)

    def __attrs_post_init__(self):
        self.run_dir = os.path.join(self.config["save_dir"], slugify(self.run_name))
        if self.enabled:
            os.makedirs(self.run_dir, exist_ok=True)
            if self.run_config is not None:
                with open(
                    os.path.join(self.run_dir, "config.yaml"), "wt", encoding="utf-8"
                ) as f:
                    yaml.safe_dump(self.run_config, f, sort_keys=False)

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    def _metadata(self, step: int, **extra) -> Dict[str, Any]:
        return {"run_name": self.run_name, "step": step, **extra}

    def step(self, step: int, rewards: Dict[str, float]) -> None:
        """
        Call once per environment step with the reward of each agent.
        """
        if not self.enabled:
            return
        self._rewards.append(float(np.mean([np.mean(r) for r in rewards.values()])))

        save_freq = self.config["save_freq"]
        if not save_freq or (step + 1) % save_freq != 0:
            return

        score = float(np.mean(self._rewards))
        self._rewards.clear()

        path = os.path.join(self.run_dir, f"step_{step + 1}")
        save_policies(path, self.policies, self._metadata(step + 1, mean_reward=score))
        self._saved.append(path)
        while len(self._saved) > max(1, self.config["keep_last"]):
            shutil.rmtree(self._saved.pop(0), ignore_errors=True)

        if self.config["save_best"] and score > self._best_score:
            self._best_score = score
            best = os.path.join(self.run_dir, "best")
            shutil.rmtree(best, ignore_errors=True)
            shutil.copytree(path, best)

        if self.config["push_to_hub"] and self.config["hf_push_every_save"]:
            self.push_to_hub(f"Checkpoint at step {step + 1}")

    def finalize(self, step: int) -> str:
        if not self.enabled:
            return ""
        path = os.path.join(self.run_dir, "final")
        save_policies(path, self.policies, self._metadata(step))

        if self.config["upload_to_wandb"] and self.wandb_run is not None:
            import wandb  # pylint: disable=import-outside-toplevel

            artifact = wandb.Artifact(
                f"{slugify(self.run_name)}-checkpoint",
                type="model",
                metadata=self._metadata(step),
            )
            artifact.add_dir(self.run_dir)
            self.wandb_run.log_artifact(artifact)

        if self.config["push_to_hub"]:
            self.push_to_hub(f"Final checkpoint at step {step}")

        return path

    def push_to_hub(self, commit_message: str) -> None:
        try:
            from huggingface_hub import HfApi  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Pushing to the Hugging Face Hub requires `pip install huggingface_hub`"
            ) from e

        api = HfApi(token=os.environ.get(self.config["hf_token_env"]) or None)
        repo_id = self.config["hf_repo_id"]
        api.create_repo(repo_id, private=self.config["hf_private"], exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=self.run_dir,
            path_in_repo=slugify(self.run_name),
            commit_message=commit_message,
            # Mirror the local rotation of periodic checkpoints
            delete_patterns=["step_*/*"],
        )
