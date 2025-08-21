import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.core.rl_module.apis.q_net_api import QNetAPI
from ray.rllib.core.rl_module.apis.target_network_api import TargetNetworkAPI
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models import ModelCatalog
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from stable_baselines3.sac.policies import Actor
from typing import Dict


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class BackBone(nn.Module):
    def __init__(self, state_dim, latent_dim, num_devices, *args, **kwargs):
        super(BackBone, self).__init__(*args, **kwargs)
        self.num_devices = num_devices
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(state_dim, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                256, 4, 512, batch_first=True
            ), 
            num_layers=1
        )
        self.project = nn.Linear(256*self.num_devices, latent_dim)

    def forward(self, batch:Dict[str, torch.Tensor])->torch.Tensor:
        obs = batch['obs']
        batch_size = obs.shape[0]
        out = obs.reshape(batch_size, self.num_devices, self.state_dim)
        
        out = self.embed(out)
        out = F.relu(out)

        out = self.transformer(out)
        out = out.reshape(batch_size, -1) # Flatten the output
        out = self.project(out)
        out = F.relu(out)
        
        return {
            ENCODER_OUT: out
        }


class SACPAModule(DefaultSACTorchRLModule):
    @override(TorchRLModule)
    def setup(self):
        # Pull custom params from model_config["custom_model_config"]
        if not self.model_config:
            raise ValueError("Model config is empty. Please provide a valid model config.")
        try:
            state_dim = self.model_config.get("state_dim")
            latent_dim = self.model_config.get("latent_dim", 256)
            num_devices = self.model_config.get("num_devices")
        except KeyError as e:
            raise KeyError(f"Missing required model config parameter: {e}")
        
        self.twin_q = self.model_config["twin_q"]

        # Build the encoder for the policy.
        self.pi_encoder = BackBone(state_dim, latent_dim, num_devices)

        if not self.inference_only or self.framework != "torch":
            # SAC needs a separate Q network encoder (besides the pi network).
            # This is because the Q network also takes the action as input
            # (concatenated with the observations).
            self.qf_encoder = BackBone(state_dim, latent_dim, num_devices)

            # If necessary, build also a twin Q encoders.
            if self.twin_q:
                self.qf_twin_encoder = BackBone(state_dim, latent_dim, num_devices)

        # Build heads.
        self.pi = self.catalog.build_pi_head(framework=self.framework)

        if not self.inference_only or self.framework != "torch":
            self.qf = self.catalog.build_qf_head(framework=self.framework)
            # If necessary build also a twin Q heads.
            if self.twin_q:
                self.qf_twin = self.catalog.build_qf_head(framework=self.framework)