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
            repr_str += f"\n... and {len(entries) - 5} more entries"
        return repr_str or "Table(empty)"


@attrs.define
class QTable(Table):
    best_action_cache: Dict = attrs.field(
        init=False, factory=dict
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
    rng: np.random.Generator | None = attrs.field(default=None, kw_only=True)
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

    def state_dict(self) -> Dict[str, Any]:
        def dump(table: Table) -> Dict:
            return {
                "default_value": table.default_value,
                "table": {s: dict(a) for s, a in table.table.items()},
            }

        state = super().state_dict()
        state["Q_tables"] = [
            {**dump(q), "best_action_cache": dict(q.best_action_cache)}
            for q in self.Q_tables
        ]
        state["V_tables"] = [dump(v) for v in self.V_tables]
        state["Alpha_tables"] = [dump(a) for a in self.Alpha_tables]
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        def restore(table: Table, saved: Dict):
            table.default_value = saved["default_value"]
            table.table = defaultdict(
                lambda: defaultdict(lambda: table.default_value),
                {
                    s: defaultdict(lambda: table.default_value, a)
                    for s, a in saved["table"].items()
                },
            )

        state = dict(state)
        for name, cls in (
            ("Q_tables", QTable),
            ("V_tables", VTable),
            ("Alpha_tables", AlphaTable),
        ):
            tables = []
            for saved in state.pop(name):
                table = cls()
                restore(table, saved)
                if "best_action_cache" in saved:
                    table.best_action_cache = dict(saved["best_action_cache"])
                tables.append(table)
            setattr(self, name, tables)
        super().load_state_dict(state)

    def unbatch_obs(self, obs) -> List[np.ndarray]:
        return [o for o in obs]

    def unbatch_data(
        self, data: ReplayBufferSamples
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        unbatched_data = []
        for obs, next_obs, act, rew in zip(
            data.observations, data.next_observations, data.actions, data.rewards
        ):
            unbatched_data.append(
                [
                    obs.detach().cpu().numpy(),
                    next_obs.detach().cpu().numpy(),
                    act.detach().cpu().numpy(),
                    rew.detach().cpu().numpy(),
                ]
            )

        return unbatched_data

    @property
    def num_actions(self) -> int:
        return int(np.prod(self.action_space.high - self.action_space.low + 1))

    def max_q_value(self, q_table: QTable, state: tuple) -> float:
        """
        max_a Q(s, a) over *all* actions, as in (23): Q-tables are initialized to 0 (Algorithm 1,
        line 2), so actions never tried in `s` count with value 0. (The cached best action of
        QTable is not used: it is not updated when the value of the cached action decreases.)
        """
        row = q_table.table.get(state, {})
        values = [float(np.squeeze(v)) for v in row.values()]
        if len(row) < self.num_actions:
            values.append(q_table.default_value)
        return max(values)

    def sample_untried_action(self, tried) -> tuple:
        while True:
            action = tuple(self.action_space.sample().flatten().tolist())
            if action not in tried:
                return action

    def risk_averse_best_action(self, Q_random: QTable, state: tuple):
        """
        argmax_a Q_hat(s, a), with (22)
            Q_hat(s, a) = Q_H(s, a) - lambda_p / (I - 1) * sum_i (Q_i(s, a) - mean_j Q_j(s, a))^2
        over all actions. Actions never tried in `s` have Q_i(s, a) = 0 for every i, hence
        Q_hat(s, a) = 0; if no tried action is better, an untried one is taken (uniformly).
        Only the row of the current state is evaluated (building the full risk-averse table
        every step is O(table size) and becomes the bottleneck once many states are visited).
        """
        actions = {}
        for q_table in self.Q_tables:
            if state in q_table.table:
                actions.update(dict.fromkeys(q_table.table[state]))

        def value(q_table: QTable, action) -> float:
            row = q_table.table.get(state, {})
            return float(np.squeeze(row.get(action, q_table.default_value)))

        best_action, best_value = None, -np.inf
        for action in actions:
            q_values = np.array([value(q_table, action) for q_table in self.Q_tables])
            risk = ((q_values - q_values.mean()) ** 2).sum()
            risk_averse_value = (
                value(Q_random, action) - self.lambda_p / (self.num_q_table - 1) * risk
            )
            if risk_averse_value > best_value:
                best_action, best_value = action, risk_averse_value

        untried_value = 0.0  # Q-tables are initialized to 0
        if len(actions) < self.num_actions and best_value < untried_value:
            return self.sample_untried_action(actions)
        return best_action

    def inference(self, obs, **kwargs):
        unbatched_obs = self.unbatch_obs(obs)
        batched_actions = []
        
        for o in unbatched_obs:
            state = tuple(o.flatten().tolist())
            
            # Use Generator if available, otherwise fall back to global numpy RNG
            if self.rng is None:
                H = np.random.randint(0, self.num_q_table)
                p = np.random.rand()
            else:
                H = self.rng.integers(0, self.num_q_table)
                p = self.rng.random()
            
            Q_random = self.Q_tables[H]
            self.epsilon = self.epsilon * self.lambda_

            if p < self.epsilon:
                action = self.action_space.sample()
                action = tuple(action.flatten().tolist())

            else:
                action = self.risk_averse_best_action(Q_random, state)

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
            rew = float(np.squeeze(rew))  # (1,) array from the buffer -> scalar Q-values

            # Use Generator if available, otherwise fall back to global numpy RNG
            if self.rng is None:
                J = np.random.poisson(1, self.num_q_table)
            else:
                J = self.rng.poisson(1, self.num_q_table)
            
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
                            + self.gamma * self.max_q_value(self.Q_tables[i], next_obs)
                            - self.Q_tables[i].get(obs, act)
                        )
                        - self.x0
                    )

                    self.Q_tables[i].update(obs, act, q_update_value)

        return (0.0, 0.0, 0.0, 0.0, 0.0)
