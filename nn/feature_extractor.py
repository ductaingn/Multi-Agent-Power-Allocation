from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


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


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.spaces.Box, state_dim:int, latent_dim:int, num_devices:int, features_dim = 256, *args, **kwargs):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.feature_extractor_net = BackBone(state_dim, latent_dim, num_devices)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor_net(observations)

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override


class FeatureExtractor(TorchModelV2, nn.Module):
    def __init__(
        self, 
        obs_space:gym.spaces.Space,
        action_space:gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *args, **kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Pull custom params from model_config["custom_model_config"]
        cfg:dict = model_config.get("model_config", {})
        if cfg is {}:
            raise ValueError("Model config is empty. Please provide a valid model config.")
        try:
            state_dim = cfg.get("state_dim")
            latent_dim = cfg.get("latent_dim", 256)
            num_devices = cfg.get("num_devices")
        except KeyError as e:
            raise KeyError(f"Missing required model config parameter: {e}")
        
        self.backbone = BackBone(state_dim, latent_dim, num_devices)

        # Policy & value heads
        self.policy_head = nn.Linear(latent_dim, num_outputs)
        self.value_branch = nn.Linear(latent_dim, 1)

        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_flat = input_dict["obs_flat"]
        features = self.backbone(obs_flat)
        self._value_out = self.value_branch(features)
        model_out = self.policy_head(features)

        return model_out, state
    
    def value_function(self):
        if self._value_out is None:
            raise RuntimeError("Value function called before forward pass.")
        return self._value_out.view(-1)
    
ModelCatalog.register_custom_model(
    "FeatureExtractor",
    FeatureExtractor
)