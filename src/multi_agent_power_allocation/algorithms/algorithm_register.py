from enum import Enum

from multi_agent_power_allocation.algorithms.high_level import (
    DQN,
    RAQL,
    SACPA,
    SACPF,
    Random,
)


class Algorithms(Enum):
    """
    High level algorithms
    """

    SACPA = SACPA
    SACPF = SACPF
    DQN = DQN
    RAQL = RAQL
    RANDOM = Random
