import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns
from stable_baselines3.sac.policies import Actor


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

    def forward(self, state:torch.Tensor)->torch.Tensor:
        batch_size = state.shape[0]
        out = state.reshape(batch_size, self.num_devices, self.state_dim)
        
        out = self.embed(out)
        out = F.relu(out)

        out = self.transformer(out)
        out = out.reshape(batch_size, -1) # Flatten the output
        out = self.project(out)
        out = F.relu(out)
        
        return out


class FeatureExtractor(TorchRLModule):
    @override(TorchRLModule)
    def setup(self):
        # Pull custom params from model_config["custom_model_config"]
        if not self.model_config:
            raise ValueError("Model config is empty. Please provide a valid model config.")
        try:
            state_dim = self.model_config.get("state_dim")
            latent_dim = self.model_config.get("latent_dim", 256)
            num_devices = self.model_config.get("num_devices")
            action_dim = int(np.prod(self.action_space.shape))
        except KeyError as e:
            raise KeyError(f"Missing required model config parameter: {e}")
        
        self.backbone = BackBone(state_dim, latent_dim, num_devices)

        self.mu = nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, action_dim),
        )

        self.log = nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, action_dim),
        )

    def _forward(self, batch, **kwargs):
        latent = self.backbone(batch[Columns.OBS])
        mu = self.mu(latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return {Columns.ACTION_DIST_INPUTS: (mu, log_std)}