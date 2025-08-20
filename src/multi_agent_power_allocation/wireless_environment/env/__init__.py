from .base_env import WirelessEnvironmentBase
from .sacpa_env import WirelessEnvironmentSACPA
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from functools import partial
from gymnasium.envs.registration import register

def env_creater(cls, config):
    return ParallelPettingZooEnv(cls(**config))

register_env("WirelessEnvironmentSACPA-v2", env_creator=partial(env_creater, WirelessEnvironmentSACPA))