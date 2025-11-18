from typing import Any, Tuple, Union, List, Dict
from collections import defaultdict
import pickle

import attrs

import torch

import numpy as np

import gymnasium as gym

from multi_agent_power_allocation.algorithms.low_level.utils.replay_buffer import (
    ReplayBufferSamples,
)
from . import LowLevelAlgorithm, DummyActor


ln2 = np.log(2)


@attrs.define
class Table:
    default_value: float = 0.0
    table: Dict[tuple, Dict[tuple, float]] = attrs.field(init=False)

    @table.default
    def _table_factory(self):
        return defaultdict(lambda: defaultdict(lambda: self.default_value))

    def get(self, state: Any, action: Tuple[int, ...]) -> float:
        return self.table[state][action]

    def update(self, state: Any, action: Tuple[int, ...], value: float):
        self.table[state][action] = value

    def all_state_actions(self):
        for state, actions in self.table.items():
            for action, value in actions.items():
                yield (state, action, value)

    def save(self, filename):
        """
        Save the Q-table to a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def load(self, filename):
        """
        Load the Q-table from a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def __add__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(
                state, action, self.get(state, action) + other.get(state, action)
            )
        return result

    def __sub__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(
                state, action, self.get(state, action) - other.get(state, action)
            )
        return result

    def __mul__(self, scalar: Union[int, float]) -> "Table":
        result = Table(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result

    def __truediv__(self, scalar: Union[int, float]) -> "Table":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "Table":
        return self.__mul__(scalar)

    def __iadd__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self

    def __pow__(self, exponent: float) -> "Table":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")

        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value**exponent)
        return result

    def copy(self) -> "Table":
        new_q = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        return new_q

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Table):
            return False
        return dict(self.table) == dict(other.table)

    def __repr__(self) -> str:
        entries = list(self.all_state_actions())
        preview = entries[:5]
        repr_str = "\n".join(f"{s} | {a} → {q:.2f}" for s, a, q in preview)
        if len(entries) > 5:
            repr_str += f"\n... and {len(entries)-5} more entries"
        return repr_str or "Table(empty)"


@attrs.define
class QTable(Table):
    best_action_cache: Dict = attrs.field(
        init=False, default=dict()
    )  # Cache: state -> best_action

    def update(self, state, action, value):
        super().update(state, action, value)

        # Update best action cache
        current_best = self.best_action_cache.get(state)
        if current_best is None or value > self.get(state, current_best):
            self.best_action_cache[state] = action

    def best_action(self, state: Any):
        return self.best_action_cache.get(state, None)

    def max_q_value(self, state: Any) -> float:
        best = self.best_action(state)
        if best is None:
            return self.default_value
        return self.get(state, best)

    def __add__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(
                state, action, self.get(state, action) + other.get(state, action)
            )
        return result

    def __sub__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(
                state, action, self.get(state, action) - other.get(state, action)
            )
        return result

    def __mul__(self, scalar: Union[int, float]) -> "QTable":
        result = QTable(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result

    def __truediv__(self, scalar: Union[int, float]) -> "QTable":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "QTable":
        return self.__mul__(scalar)

    def __iadd__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self

    def __pow__(self, exponent: float) -> "QTable":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")

        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value**exponent)
        return result

    def copy(self):
        new_q = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        new_q.best_action_cache = self.best_action_cache.copy()
        return new_q

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "table": dict(self.table),
                    "best_action_cache": self.best_action_cache,
                    "default_value": self.default_value,
                },
                f,
            )

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data["default_value"]
        self.table = defaultdict(
            lambda: defaultdict(lambda: self.default_value), data["table"]
        )
        self.best_action_cache = data["best_action_cache"]


class VTable(Table):
    def update(self, state, action, value=1):
        if state not in self.table:
            self.table[state][action] = value
        else:
            self.table[state][action] += value

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "table": dict(self.table),
                    "default_value": self.default_value,
                },
                f,
            )

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data["default_value"]
        self.table = defaultdict(
            lambda: defaultdict(lambda: self.default_value), data["table"]
        )


class AlphaTable(Table):
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "table": dict(self.table),
                    "default_value": self.default_value,
                },
                f,
            )

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data["default_value"]
        self.table = defaultdict(
            lambda: defaultdict(lambda: self.default_value), data["table"]
        )


@attrs.define
class RAQL(LowLevelAlgorithm):
    action_space: gym.spaces.Space
    num_q_table: int = 4
    epsilon: float = 0.5
    gamma: float = 0.9
    lambda_p: float = 0.5
    beta: float = -0.5
    lambda_: float = 0.995
    x0: float = -1
    actor: DummyActor = attrs.field(init=False, default=DummyActor())
    Q_tables: List[QTable] = attrs.field(init=False)
    V_tables: List[VTable] = attrs.field(init=False)
    Alpha_tables: List[AlphaTable] = attrs.field(init=False)

    @Q_tables.default
    def _Q_tables_factory(self):
        return [QTable() for _ in range(self.num_q_table)]

    @V_tables.default
    def _V_tables_factory(self):
        return [VTable() for _ in range(self.num_q_table)]

    @Alpha_tables.default
    def _Alpha_tables_factory(self):
        return [AlphaTable() for _ in range(self.num_q_table)]

    def unbatch_obs(self, obs) -> List[np.ndarray]:
        return [o for o in obs]

    def unbatch_data(
        self, data: ReplayBufferSamples
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        unbatched_data = []
        for obs, next_obs, act, rew in zip(
            data.observations, data.next_observations, data.actions, data.rewards
        ):
            unbatched_data.append([obs, next_obs, act, rew])

        return unbatched_data

    def inference(self, obs, **kwargs):
        unbatched_obs = self.unbatch_obs(obs)
        batched_actions = []
        for o in unbatched_obs:
            state = tuple(o.flatten().tolist())
            H = np.random.randint(0, self.num_q_table)
            Q_random = self.Q_tables[H]
            self.epsilon = self.epsilon * self.lambda_

            p = np.random.rand()
            if p < self.epsilon:
                action = self.action_space.sample()
                action = tuple(action.flatten().tolist())

            else:
                average_q_table = (
                    sum(self.Q_tables, start=QTable(default_value=0)) / self.num_q_table
                )
                risk_averse_Q: QTable = Q_random - self.lambda_p / (
                    self.num_q_table - 1
                ) * (
                    sum(
                        (
                            (self.Q_tables[i] - average_q_table) ** 2
                            for i in range(self.num_q_table)
                        ),
                        start=QTable(default_value=0),
                    )
                )
                action = risk_averse_Q.best_action(state=state)
                if action is None:
                    # If no action is found, sample a random action
                    action = self.action_space.sample()
                    action = tuple(action.flatten().tolist())

                if action not in state:
                    risk_averse_Q.update(state, action, risk_averse_Q.default_value)

                if risk_averse_Q.table[state][action] < risk_averse_Q.default_value:
                    action = self.action_space.sample()

            batched_actions.append(action)

        return torch.tensor(batched_actions)

    def u(self, x):
        u = -np.exp(self.beta * x)
        return u

    def learn(self, data):
        unbached_data = self.unbatch_data(data)
        for obs, next_obs, act, rew in unbached_data:
            obs = tuple(obs.flatten().tolist())
            next_obs = tuple(next_obs.flatten().tolist())
            act = tuple(act.flatten().tolist())

            J = np.random.poisson(1, self.num_q_table)
            for i in range(self.num_q_table):
                if J[i] == 1:
                    self.V_tables[i].update(obs, act)
                    self.Alpha_tables[i].update(
                        obs, act, 1 / (self.V_tables[i].get(obs, act))
                    )

                    q_update_value = self.Q_tables[i].get(obs, act) + self.Alpha_tables[
                        i
                    ].get(obs, act) * (
                        self.u(
                            rew
                            + self.gamma * self.Q_tables[i].max_q_value(next_obs)
                            - self.Q_tables[i].get(obs, act)
                        )
                        - self.x0
                    )

                    self.Q_tables[i].update(obs, act, q_update_value)

        return [0.0] * 5
