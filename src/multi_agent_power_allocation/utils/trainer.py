from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.nn.module import SACPAACtor, SACPACritic, Adam, ActorTraceWrapper, CriticTraceWrapper
from multi_agent_power_allocation.wireless_environment.env import *
from multi_agent_power_allocation.utils.collector import Collector
from multi_agent_power_allocation.utils.logger import Logger

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv, RayVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import SACPolicy, MultiAgentPolicyManager, BasePolicy
from tianshou.trainer import OffpolicyTrainer, BaseTrainer
from tianshou.utils import MultipleLRSchedulers

from pettingzoo.utils.conversions import parallel_to_aec
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import gymnasium as gym
import numpy as np

import wandb
import pickle
import yaml
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

    device_blockages = np.array(wc_cluster_config["device_blockages"]) - 1
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
    SAC_config: Dict = attrs.field()
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


    def get_agents(self) -> Tuple[MultiAgentPolicyManager, List]:
        env = self.get_env()

        policies = []
        for _ in range(self.num_agent):
            actor = SACPAACtor(**self.model_config)
            actor_optim = Adam(actor.parameters(), lr=self.SAC_config["lr"])
            critic1 = SACPACritic(**self.model_config)
            critic1_optim = Adam(critic1.parameters(), lr=self.SAC_config["lr"])
            critic2 = SACPACritic(**self.model_config)
            critic2_optim = Adam(critic2.parameters(), lr=self.SAC_config["lr"])

            # auto entropy tuning setup
            target_entropy = float(-np.prod(env.action_space.shape))  
            log_alpha = torch.log(torch.ones(1) * 1.0).requires_grad_(True)
            alpha_optim = Adam([log_alpha], lr=self.SAC_config["lr"])

            scheduler = MultipleLRSchedulers(
                CosineAnnealingLR(actor_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(critic1_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(critic2_optim, T_max=self.env_config["max_num_step"]),
                CosineAnnealingLR(alpha_optim, T_max=self.env_config["max_num_step"])
            )

            policy = SACPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim,
                alpha=(target_entropy, log_alpha, alpha_optim),
                lr_scheduler=scheduler
            )

            policies.append(policy)
        
        policy = MultiAgentPolicyManager(policies, env)

        return policy, env.agents

    def build(self, run_name:str) -> BaseTrainer:
        # ======== environment setup =========
        train_envs = DummyVectorEnv([lambda: self.get_env() for _ in range(self.num_env)])

        # ======== agent setup =========
        policy, agents = self.get_agents()

        # ======== logging setup =========
        logger = Logger(
            train_interval=1,
            test_interval=1,
            update_interval=1,
            project=self.wandb_config["project"],
            config={
                "algorithm": self.algorithm,
                "env_config": self.env_config
            },
            name=run_name
        )
        writer = logger.writer
        
        dummy_env = self.get_env()
        dummy_obs = torch.tensor([dummy_env.observation_space.sample()])
        dummy_act = torch.tensor([dummy_env.action_space.sample()])

        actor_script = torch.jit.script(ActorTraceWrapper(policy.policies[agents[0]].actor))
        critic1_script = torch.jit.script(CriticTraceWrapper(policy.policies[agents[0]].critic1))
        writer.add_graph(actor_script, dummy_obs)
        writer.add_graph(critic1_script, (dummy_obs, dummy_act))

        del dummy_env, dummy_obs, dummy_act

        def log_params(step):
            for agent, pol in policy.policies.items():
                pol: SACPolicy

                lr = pol.actor_optim.param_groups[0]['lr']
                writer.add_scalar(f"Agent {agent}/ learning rate", lr, step)

                for name, param in pol.actor.named_parameters():
                    writer.add_histogram(f"Agent {agent}/ actor module/ {name}", param, step)
                    writer.add_histogram(f"Agent {agent}/ actor module/ {name}.grad", torch.zeros_like(param) if param.grad is None else param.grad, step)

        
        # ======== collector setup =========
        train_collector = Collector(
            policy=policy,
            env=train_envs,
            buffer=VectorReplayBuffer(100_000*self.num_env, buffer_num=self.num_env),
            exploration_noise=True
        )
        train_collector.load_writer(writer)

        # ======== callback setup ========
        def save_best_fn(policy):
            model_save_path = os.path.join(writer.log_dir, "model", "policy.pth")
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

        def train_fn(epoch, env_step):
            log_params(env_step)

        # ======== trainer setup ========
        trainer = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=1,
            step_per_epoch=10_000,
            step_per_collect=1,
            episode_per_test=1,
            batch_size=256,
            update_per_step=1,
            save_best_fn=save_best_fn,
            train_fn=train_fn,
            logger=logger,
            test_in_train=False,
        )

        return trainer

    def train(self, run_name:str) -> Dict[str, float | str]:
        trainer = self.build(run_name)

        result = trainer.run()

        trainer.logger.writer.close()
        trainer.logger.wandb_run.finish(exit_code=0)

        return result