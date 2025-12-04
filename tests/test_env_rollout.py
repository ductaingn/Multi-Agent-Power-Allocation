import os
import pytest
from tqdm import tqdm

from multi_agent_power_allocation.wireless_environment.env.env import (
    WirelessEnvironment,
)
from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.utils.train_config import TrainConfig


def test_env_rollout():
    config_path = os.path.join(BASE_DIR, "run", "default_config.yaml")
    config = TrainConfig(config_path)
    config.env_config["render_mode"] = "human"
    env = WirelessEnvironment(**config.env_config)
    obs, infos = env.reset()
    for frame in tqdm(range(1, 1 + 10000)):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
