from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from copy import deepcopy

import numpy as np

from pettingzoo import ParallelEnv

# Type aliases
AgentID = Union[str, int]
EnvFn = Callable[[], ParallelEnv]
Obs = Dict[AgentID, np.ndarray]  # Observation of one environment
Actions = Dict[AgentID, np.ndarray]
Rewards = Dict[AgentID, float]
Terminations = Dict[AgentID, bool]
Truncations = Dict[AgentID, bool]
Infos = Dict[AgentID, Dict[str, float]]

BatchedObs = Dict[AgentID, np.ndarray]
BatchedActions = Dict[AgentID, np.ndarray]
BatchedRewards = Dict[AgentID, np.ndarray]
BatchedTerminations = Dict[AgentID, np.ndarray]
BatchedTruncations = Dict[AgentID, np.ndarray]
BatchedInfos = Dict[AgentID, List[Dict[str, float]]]


class SyncVecEnv:
    """
    Synchronous (serial) vectorized wrapper for PettingZoo ParallelEnv.

    - env_fns: iterable of callables that create ParallelEnv instances.
    - copy: whether to deepcopy returned observations (like gym's SyncVectorEnv).
    """

    def __init__(self, env_fns: Sequence[EnvFn], copy: bool = True):
        self.envs: List[ParallelEnv] = [fn() for fn in env_fns]
        if len(self.envs) == 0:
            raise ValueError("env_fns must contain at least one env factory")
        self.num_envs = len(self.envs)
        self.copy = copy

        # Validate agent sets and spaces
        self.possible_agents = list(self.envs[0].possible_agents)
        for e in self.envs[1:]:
            if list(e.possible_agents) != self.possible_agents:
                raise ValueError(
                    "All envs must have identical possible_agents in the same order"
                )

        # Per-agent spaces (assume identical across envs)
        self.observation_spaces = {
            ag: self.envs[0].observation_space(ag) for ag in self.possible_agents
        }
        self.action_spaces = {
            ag: self.envs[0].action_space(ag) for ag in self.possible_agents
        }

        # internal buffers
        # _env_obs[i] is the obs dict returned by env i
        self._env_obs: List[Obs | None] = [None] * self.num_envs
        self._env_actions: List[Actions | None] = [None] * self.num_envs
        self._env_rewards: List[Rewards | None] = [None] * self.num_envs
        self._env_infos: List[Infos | None] = [None] * self.num_envs
        self._env_terminations: List[Terminations | None] = [None] * self.num_envs
        self._env_truncations: List[Truncations | None] = [None] * self.num_envs

    # -------------------------
    # Reset
    # -------------------------
    def reset(
        self, seed: int | Sequence[int] | None = None, options: dict | None = None
    ) -> Tuple[BatchedObs, List[dict]]:
        """
        Reset all envs.

        Returns:
            batched_obs: dict agent -> np.ndarray(shape=(num_envs, *obs_shape))
            infos_list: list of infos per env (length num_envs)
        """
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            if len(seed) != self.num_envs:
                raise ValueError("seed sequence length must equal num_envs")
            seeds = list(seed)

        for i, (env, s) in enumerate(zip(self.envs, seeds)):
            obs_dict, info = env.reset(seed=s)
            self._env_obs[i] = obs_dict
            self._env_infos[i] = info

        # Build per-agent batched obs arrays
        batched_obs: BatchedObs = {}
        batched_infos: BatchedInfos = {}
        for ag in self.possible_agents:
            # collect observations for this agent across all envs
            obs_list = [self._env_obs[i][ag] for i in range(self.num_envs)]
            infos_list = [self._env_infos[i][ag] for i in range(self.num_envs)]
            batched_obs[ag] = np.stack(obs_list, axis=0)
            batched_infos[ag] = infos_list

        return (deepcopy(batched_obs) if self.copy else batched_obs, batched_infos)

    def step(self, actions: BatchedActions) -> Tuple[
        BatchedObs,
        BatchedRewards,
        BatchedTerminations,
        BatchedTruncations,
        BatchedInfos,
    ]:
        """
        Step all envs.

        Args:
            actions: BatchedActions

        Returns:
            batched_obs: dict agent -> np.ndarray(shape=(num_envs, ...))
            rewards: dict agent -> np.ndarray(shape=(num_envs,))
            dones: np.ndarray shape (num_envs,)  -- True if any agent done in that env
            infos_list: list of info dicts (len=num_envs)
        """
        # Normalize actions into a list of per-env action dicts
        first_agent = next(iter(actions))
        batched_len = len(actions[first_agent])
        if batched_len != self.num_envs:
            raise ValueError("Batched actions must have first dim == num_envs")
        for i in range(self.num_envs):
            self._env_actions[i] = {
                ag: np.asarray(actions[ag])[i] for ag in self.possible_agents
            }

        # Step
        for i, (env, act) in enumerate(zip(self.envs, self._env_actions)):
            obs_i, rew_i, term_i, trunc_i, info_i = env.step(act)
            self._env_obs[i] = obs_i
            self._env_rewards[i] = rew_i
            self._env_terminations[i] = term_i
            self._env_truncations[i] = trunc_i
            self._env_infos[i] = info_i

        # Build batched obs and rewards per agent
        batched_obs: BatchedObs = {}
        batched_rewards: BatchedRewards = {}
        batched_terminations: BatchedTerminations = {}
        batched_truncations: BatchedTruncations = {}
        batched_infos: BatchedInfos = {}
        for ag in self.possible_agents:
            obs_list = [self._env_obs[i][ag] for i in range(self.num_envs)]
            rew_list = [self._env_rewards[i][ag] for i in range(self.num_envs)]
            term_list = [self._env_terminations[i][ag] for i in range(self.num_envs)]
            trunc_list = [self._env_truncations[i][ag] for i in range(self.num_envs)]
            info_list = [self._env_infos[i][ag] for i in range(self.num_envs)]

            batched_obs[ag] = np.stack(obs_list, axis=0)
            batched_rewards[ag] = np.stack(rew_list, axis=0)
            batched_terminations[ag] = np.stack(term_list, axis=0)
            batched_truncations[ag] = np.stack(trunc_list, axis=0)
            batched_infos[ag] = info_list

        return (
            deepcopy(batched_obs) if self.copy else batched_obs,
            batched_rewards,
            batched_terminations,
            batched_truncations,
            batched_infos,
        )

    def get_attr(self, name: str) -> Tuple[Any, ...]:
        return tuple(getattr(env, name) for env in self.envs)

    def set_attr(self, name: str, values: Any):
        if not isinstance(values, (list, tuple)):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError("Values length must equal num_envs")
        for env, v in zip(self.envs, values):
            setattr(env, name, v)

    def call(self, method_name: str, *args, **kwargs):
        results = []
        for env in self.envs:
            fn = getattr(env, method_name)
            results.append(fn(*args, **kwargs))
        return tuple(results)

    def close(self):
        for env in self.envs:
            env.close()
