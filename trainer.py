from wireless_environment.env.base_env import WirelessEnvironmentBase
from ray import tune
from ray.rllib.algorithms.sac import SACConfig

config = (
    SACConfig()
    .environment(
        WirelessEnvironmentBase,
        env_config={

        }
    )
    .framework("torch")
).training()
