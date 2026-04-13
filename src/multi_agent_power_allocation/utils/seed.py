"""
Seed utilities for reproducible training using numpy.random.Generator.

This module provides a centralized way to create reproducible random generators
and set seeds across Python's random module and PyTorch (CPU and CUDA).
Using numpy.random.Generator provides better control over randomness in 
algorithm instances compared to the deprecated global np.random.seed().
"""

import numpy as np
import random
import torch


def create_generator(seed: int) -> np.random.Generator:
    """
    Create a numpy.random.Generator with the given seed.
    
    This should be used instead of np.random.seed() because it provides
    instance-level RNG control, allowing different components to have
    independent but reproducible random streams.
    
    Parameters
    ----------
    seed : int
        The random seed value
    
    Returns
    -------
    np.random.Generator
        A generator instance that can be passed to algorithms and other components
    """
    return np.random.default_rng(seed)


def set_seed(seed: int) -> None:
    """
    Set all random seeds for complete reproducibility (non-Generator components).
    
    This function ensures reproducibility across:
    - Python's built-in random module
    - PyTorch CPU operations
    - PyTorch CUDA operations
    - CuDNN operations
    
    Note: For NumPy operations, use create_generator() and pass the Generator
    instance to your algorithms instead of relying on global state.
    
    Parameters
    ----------
    seed : int
        The random seed value to set
    
    Notes
    -----
    - This should be called early in the training script
    - For algorithm-specific RNG, use create_generator() instead of this function
    """
    # Set Python's random module seed
    random.seed(seed)

    # Set NumPy seeds
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
