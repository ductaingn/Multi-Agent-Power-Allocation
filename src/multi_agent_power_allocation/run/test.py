import os
from tqdm import tqdm

from multi_agent_power_allocation.wireless_environment.env.env import (
    WirelessEnvironment,
)
from multi_agent_power_allocation.utils.trainer import parse_config
from multi_agent_power_allocation import BASE_DIR

if __name__ == "__main__":
    config_path = os.path.join(BASE_DIR, "run", "default_config.yaml")
    config: dict = parse_config(config_path)
    env = WirelessEnvironment(**config["env_config"])
    obs, infos = env.reset()
    for _ in tqdm(range(10000)):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        env.render()  # should pop up your matplotlib figure
