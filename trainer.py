from wireless_environment.env import WirelessEnvironmentSACPA
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import pickle
import numpy as np

with open("data/scenario_1/h_tilde.pickle", "rb") as file:
    h_tilde = np.array(pickle.load(file))

with open("data/scenario_1/device_positions.pickle", "rb") as file:
    device_positions = np.array(pickle.load(file))

def policy_mapping_fn(agent_id, episode, **kwargs):
    return "learning_policy"

config = (
    SACConfig()
    .environment(
        "WirelessEnvironmentSACPA-v2",
        env_config=dict(
            num_cluster=3,
            reward_coef = dict(
                reward_qos = 1.0,
                reward_power = 10.0
            ),

            wc_cluster_config = dict(
                h_tilde = h_tilde,
                num_devices = 10,
                device_positions = device_positions,
                num_sub_channel = 16,
                num_beam = 16,
                device_blockages = np.array([0,1,0,0,0,1,0,0,0,0], dtype=bool)
            )
        )
    )
    .framework("torch")
    .multi_agent(
        policies=["learning_policy"],
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["learning_policy"]
    )
    # .rl_module(
    #     model_config=dict(
    #         state_dim=8,
    #         latent_dim=256,
    #         num_devices=10,
    #     )
    # )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            "learning_policy": RLModuleSpec(),
        }),
    )
).training()

algorithm = config.build_algo()
algorithm.train()
