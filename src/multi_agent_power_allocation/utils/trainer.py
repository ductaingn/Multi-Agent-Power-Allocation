from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic, Adam
from multi_agent_power_allocation.wireless_environment.env import *
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, RayVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import SACPolicy, MultiAgentPolicyManager, BasePolicy
from tianshou.trainer import OffpolicyTrainer, BaseTrainer
from tianshou.utils import WandbLogger
from pettingzoo.utils.conversions import parallel_to_aec
from torch.utils.tensorboard import SummaryWriter
import torch
import gymnasium as gym
import pickle
import yaml
import numpy as np
import attrs
from typing import Dict, Literal, Tuple, List
from tqdm import tqdm
import os
from copy import deepcopy

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
    wandb_config: Dict = attrs.field()
    num_env: int = attrs.field(default=1)


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
        must_have_keys = ["latent_dim", "num_devices"]

        for key in must_have_keys:
            if key not in value:
                raise ValueError(f"model_config must contain {key}!")
            
        value.update({"observation_space": self.get_env().observation_space})
        value.update({"action_space": self.get_env().action_space})


    def __attrs_post_init__(self):
        self.env = f"WirelessEnvironment{self.algorithm}-v2"
        self.max_num_step = self.env_config["max_num_step"]
        self.num_agent = self.env_config["num_cluster"]
        self.policies = [f"agent_{i}_policy" for i in range(self.num_agent)]


    def get_env(self):
        if self.algorithm == "SACPA":
            env_parallel = WirelessEnvironmentSACPA(**deepcopy(self.env_config))
        env_aec = parallel_to_aec(env_parallel)
        return PettingZooEnv(env_aec)


    def get_agents(self) -> Tuple[BasePolicy, List]:
        env = self.get_env()

        observation_space = env.observation_space['observation'] if isinstance(
            env.observation_space, gym.spaces.Dict
        ) else env.observation_space

        agents = []
        for _ in range(self.num_agent):
            actor = SACPAACtor(**self.model_config)
            actor_optim = Adam(actor.parameters())
            critic1 = SACPACritic(**self.model_config)
            critic1_optim = Adam(critic1.parameters())
            critic2 = SACPACritic(**self.model_config)
            critic2_optim = Adam(critic2.parameters())

            agent = SACPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim
            )

            agents.append(agent)
        
        policy = MultiAgentPolicyManager(agents, env)

        return policy, env.agents

    def build(self, run_name:str) -> BaseTrainer:
        # ======== environment setup =========
        train_envs = DummyVectorEnv([lambda: self.get_env() for _ in range(self.num_env)])

        # ======== agent setup =========
        policy, agents = self.get_agents()

        # ======== collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(100_000*self.num_env, buffer_num=self.num_env),
            exploration_noise=True
        )

        # ======== wandb logging setup =========
        log_path = os.path.join(BASE_DIR, 'SACPA')
        writer = SummaryWriter(log_path)
        logger = WandbLogger(
            project=self.wandb_config["project"],
            config={
                "algorithm": self.algorithm,
                "env_config": self.env_config
            },
            name=run_name
            )
        logger.load(writer)

        # ======== trainer setup ========
        trainer = OffpolicyTrainer(
            policy,
            train_collector,
            None,
            1,
            10_000,
            1,
            1,
            256,
            logger=logger,
            test_in_train=False,
        )

        return trainer

    def train(self, run_name:str) -> Dict[str, float | str]:
        trainer = self.build(run_name)

        result = trainer.run()

        return result