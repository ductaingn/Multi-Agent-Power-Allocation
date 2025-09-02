from tianshou.data import Batch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.optim import Adam
from gymnasium.spaces import Space


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class BackBone(nn.Module):
    def __init__(self, iot_device_state_dim, latent_dim, num_devices, *args, **kwargs):
        super(BackBone, self).__init__(*args, **kwargs)
        self.num_devices = num_devices
        self.state_dim = iot_device_state_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(iot_device_state_dim, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                256, 4, 512, batch_first=True
            ), 
            num_layers=1
        )
        self.project = nn.Linear(256*self.num_devices, latent_dim)

    def forward(self, obs:torch.Tensor)->torch.Tensor:
        batch_size = obs.shape[0]
        out = obs.reshape(batch_size, self.num_devices, self.state_dim)
        
        out = self.embed(out)
        out = F.relu(out)

        out = self.transformer(out)
        out = out.reshape(batch_size, -1) # Flatten the output
        out = self.project(out)
        out = F.relu(out)
        
        return out
        

class SACPAACtor(nn.Module):
    def __init__(self, observation_space:Space, action_space:Space, latent_dim, num_devices, *args, **kwargs):
        super().__init__(*args, **kwargs)

        iot_device_state_dim, action_dim = observation_space.shape[-1]//num_devices, action_space.shape[-1]

        self.features_extractor = BackBone(iot_device_state_dim, latent_dim, num_devices)

        self.latent_pi = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU6(inplace=True),
            nn.Linear(256, 256), nn.ReLU6(inplace=True),
        )

        # Build heads.
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        

    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if isinstance(obs, Batch):
            obs = torch.tensor(obs.obs, dtype=torch.float32)
        batch = obs.shape[0]
        features = self.features_extractor(obs.view(batch, -1))
        latent = self.latent_pi(features)
        mu = self.mu(latent)
        log_std = self.log_std(latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        logits = (mu ,log_std)

        return logits, state

        
class SACPACritic(nn.Module):
    def __init__(self, observation_space:Space, action_space:Space, latent_dim, num_devices, *args, **kwargs):
        super().__init__(*args, **kwargs)

        iot_device_state_dim, action_dim = observation_space.shape[-1]//num_devices, action_space.shape[-1]

        self.features_extractor = BackBone(iot_device_state_dim, latent_dim, num_devices)

        self.latent_q = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU6(inplace=True),
            nn.Linear(256, 256), nn.ReLU6(inplace=True),
        )

        self.qf = nn.Linear(256, 1)

    def forward(self, obs, act, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if isinstance(obs, Batch):
            obs = torch.tensor(obs.obs, dtype=torch.float32)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32)

        batch = obs.shape[0]
        features = self.features_extractor(obs.view(batch, -1))
        latent = self.latent_q(torch.cat([features, act], dim=1))
        q_value = self.qf(latent)
        
        return q_value