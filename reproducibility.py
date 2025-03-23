"""
Reproducibility utilities for ensuring deterministic behavior across runs.
This module centralizes random seed management and deterministic settings.
"""

import os
import random
import numpy as np
import torch
from config import get_config


def set_seed(seed=None, deterministic=None):
    """
    Set seed for all random number generators for reproducibility.
    
    If seed or deterministic flag is not provided, reads from config.
    
    Args:
        seed: Integer seed for random number generators. 
              If None, reads from config.
        deterministic: Whether to use deterministic algorithms in PyTorch.
                      If None, reads from config.
                      
    Returns:
        dict: Settings applied for reproducibility
    """
    # Get configuration
    config = get_config()
    
    # Use provided seed or get from config
    if seed is None:
        seed = config.get_model_param("random_seed", 42)
    
    # Use provided deterministic flag or get from config
    if deterministic is None:
        deterministic = config.get_model_param("deterministic_mode", True)
    
    # Set Python random module seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set deterministic behavior (may impact performance)
    if deterministic:
        # Ensure CuDNN uses deterministic algorithms
        # (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for PyTorch deterministic operations
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA >= 10.2
    else:
        # Allow CuDNN to benchmark multiple algorithms and pick fastest
        # (faster but not reproducible)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Return applied settings
    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def get_reproducibility_info():
    """
    Get information about current reproducibility settings.
    
    Returns:
        dict: Current reproducibility settings
    """
    config = get_config()
    seed = config.get_model_param("random_seed", 42)
    deterministic = config.get_model_param("deterministic_mode", True)
    
    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "python_hashseed": os.environ.get("PYTHONHASHSEED", "Not set"),
    }


def is_deterministic():
    """
    Check if deterministic mode is enabled.
    
    Returns:
        bool: True if using deterministic settings
    """
    return torch.backends.cudnn.deterministic