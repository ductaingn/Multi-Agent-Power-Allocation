from pettingzoo import ParallelEnv
import attrs


@attrs.define
class WirelessEnvironmentBase(ParallelEnv):
    """
    Base class for wireless environments in PettingZoo.
    This class is designed to be extended by specific wireless environment implementations.
    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "wireless_environment_base",
        "is_parallelizable": True,
    }
    
    def __attrs_post_init__(self):
        ...
        
    def reset(self, seed = None, options = None):
        ...

    def step(self, actions):
        ...
        
    def render(self):
        pass

    def observation_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def action_space(self, agent):
        raise NotImplementedError("This method should be implemented by subclasses.")