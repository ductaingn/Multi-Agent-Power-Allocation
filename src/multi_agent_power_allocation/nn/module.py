import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from gymnasium.spaces import Space, Discrete


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class BackBone(nn.Module):
    def __init__(self, iot_device_state_dim, latent_dim, num_devices, *args, **kwargs):
        super(BackBone, self).__init__(*args, **kwargs)
        self.num_devices = num_devices
        self.state_dim = iot_device_state_dim
        self.latent_dim = latent_dim

        self.embed = nn.Sequential(nn.Linear(iot_device_state_dim, 256), nn.GELU())
        self.input_norm = nn.RMSNorm(256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                256, 4, 512, batch_first=True, activation=F.gelu
            ),
            norm=nn.RMSNorm(256),
            num_layers=1,
        )
        self.project = nn.Sequential(nn.Linear(256, latent_dim), nn.GELU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        out = obs.reshape(batch_size, self.num_devices, self.state_dim)

        out = self.embed(out)

        out = self.input_norm(out)
        out: torch.Tensor = self.transformer(out)
        out = out.mean(dim=1)

        out = self.project(out)

        return out


class SACPAACtor(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        latent_dim,
        num_devices,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        iot_device_state_dim, action_dim = (
            observation_space.shape[-1] // num_devices,
            action_space.shape[-1],
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.features_extractor = BackBone(
            iot_device_state_dim, latent_dim, num_devices
        )

        self.latent_pi = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        # Build heads.
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.device = device
        self.to(self.device)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        batch = obs.shape[0]
        features = self.features_extractor(obs.view(batch, -1))
        latent = self.latent_pi(features)
        mu = self.mu(latent)
        log_std = self.log_std(latent)
        std = torch.clamp(log_std, self.log_std_min, self.log_std_max).exp()
        logits = (mu, std)

        return logits


class SACPACritic(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        latent_dim,
        num_devices,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        iot_device_state_dim, action_dim = (
            observation_space.shape[-1] // num_devices,
            action_space.shape[-1],
        )

        self.features_extractor = BackBone(
            iot_device_state_dim, latent_dim, num_devices
        )

        self.latent_q = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        self.qf = nn.Linear(256, 1)

        self.device = device
        self.to(self.device)

    def forward(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(act, np.ndarray):
            act = torch.as_tensor(act, dtype=torch.float32, device=self.device)

        batch = obs.shape[0]
        features = self.features_extractor(obs.view(batch, -1))
        latent = self.latent_q(torch.cat([features, act], dim=1))
        q_value = self.qf(latent)

        return q_value


class DQNQNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Discrete,
        latent_dim,
        num_devices,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        iot_device_state_dim, action_dim = (
            observation_space.shape[-1],
            action_space.n,
        )

        self.network = nn.Sequential(
            nn.Linear(iot_device_state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

        self.device = device
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.float().to(self.device)
        action = self.network(obs)

        return action
