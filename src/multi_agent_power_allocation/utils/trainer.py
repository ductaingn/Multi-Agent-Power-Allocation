from multi_agent_power_allocation.nn.feature_extractor import FeatureExtractor
from multi_agent_power_allocation.wireless_environment.env import *
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm import Algorithm
import pickle
import yaml
import numpy as np
import attrs
from typing import Dict, Literal
from tqdm import tqdm
from multi_agent_power_allocation import BASE_DIR
import os


def process_default_config(path:str) -> Dict:
    """
    Process the default yaml config file into keyword arguments to parse Trainer class

    Parameters:
    ----------
    path : str
        Path to the config yaml file
    
    Returns:
    -------
    kwargs : Dict
        Keyword arguments
    """
    try:
        with open(path, "rb") as file:
            config:Dict = yaml.safe_load(file)
    except Exception as e:
        print("Error occured when trying to open default config file!")
        print(e)

    algorithm:dict = config.get("algorithm")
    env_config:dict = config.get("env_config")
    wc_cluster_config:dict = env_config.get("wc_cluster_config")
    model_config:dict = config.get("model_config")

    h_tilde_path = os.path.join(
        BASE_DIR,
        "data",
        wc_cluster_config["h_tilde"]
    )
    if os.path.isfile(h_tilde_path):
        wc_cluster_config.update({
            "h_tilde": np.array(
                pickle.load(
                    open(h_tilde_path, "rb")
                )
            ) 
        })
    else:
        raise FileNotFoundError("`h_tilde` path is not valid!")

    device_positions_path = os.path.join(
        BASE_DIR,
        "data",
        wc_cluster_config["device_positions"]
    )
    if os.path.isfile(h_tilde_path):
        wc_cluster_config.update({
            "device_positions": np.array(
                pickle.load(
                    open(device_positions_path, "rb")
                )
            ) 
        })
    else:
        raise FileNotFoundError("`device_positions` path is not valid!")

    device_blockages = np.array(wc_cluster_config["device_blockages"])
    one_hot = np.zeros(wc_cluster_config["num_devices"], dtype=bool)
    one_hot[device_blockages] = True

    wc_cluster_config.update({
        "device_blockages": one_hot 
    })

    return config


@attrs.define
class Trainer:
    algorithm: Literal["SACPA, SACPF, RAQL, Random"]
    env: str = attrs.field(init=False)
    env_config: Dict = attrs.field()
    num_agent: int = attrs.field(init=False)
    model_config: Dict = attrs.field()
    max_num_step: int = attrs.field(init=False)
    policies: Dict = attrs.field(init=False)


    @env_config.validator
    def _check_env_config(self, attribute, value:Dict):
        must_have_keys = ["num_cluster", "wc_cluster_config", "max_num_step"]
        
        for key in must_have_keys:
            if key not in value:
                raise ValueError(f"env_config must contain {key}!")
        
        wc_cluster_config:Dict = value.get("wc_cluster_config")
        if not isinstance(wc_cluster_config, Dict):
            raise ValueError("wc_cluster_config must be a dictionary!")

        must_have_wccc_keys = ["h_tilde", "num_devices", "device_positions", "num_sub_channel", "num_beam", "device_blockages"]

        for key in must_have_wccc_keys:
            if key not in wc_cluster_config:
                raise ValueError(f"wc_cluster_config must contain {key}!")


    @model_config.validator
    def _check_model_config(self, attribute, value:Dict):
        must_have_keys = ["state_dim", "latent_dim", "num_devices"]

        for key in must_have_keys:
            if key not in value:
                raise ValueError(f"model_config must contain {key}!")


    def __attrs_post_init__(self):
        self.env = f"WirelessEnvironment{self.algorithm}-v2"
        self.max_num_step = self.env_config["max_num_step"]
        self.num_agent = self.env_config["num_cluster"]
        self.policies = [f"agent_{i}_policy" for i in range(self.num_agent)]


    def policy_mapping_fn(self, agent_id:int, episode, **kwargs):
        return f"agent_{agent_id}_policy"


    def build(self) -> Algorithm:
        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                policy: RLModuleSpec(
                    module_class = FeatureExtractor,
                    model_config = self.model_config
                ) for policy in self.policies
            }
        )
        config = (
            SACConfig()
            .environment(
                env=self.env,
                env_config=self.env_config
            )
            .framework("torch")
            .multi_agent(
                policies=self.policies,
                policy_mapping_fn=self.policy_mapping_fn,
                policies_to_train=self.policies
            )
            .rl_module(
                rl_module_spec=rl_module_spec
            )
        )

        config.training()

        return config.build_algo()
    

    def train(self) -> Algorithm:
        algorithm = self.build()

        for _ in range(tqdm(self.max_num_step)):
            algorithm.train()

        return algorithm