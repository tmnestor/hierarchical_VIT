"""
Device utility functions for PyTorch models.
Centralizes device selection logic across the codebase.
"""

import os
import torch


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    The order of preference is:
    1. CUDA (if available)
    2. MPS (if available, for Apple Silicon)
    3. CPU (as fallback)
    
    If using MPS, automatically enables fallback for unsupported operations.
    
    Returns:
        torch.device: The appropriate device for model operations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable MPS fallback for operations not supported by MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using Apple MPS (Metal Performance Shaders) with fallback enabled")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device


def get_device_info():
    """
    Get detailed information about the current device setup.
    
    Returns:
        dict: Dictionary containing device information
    """
    device = get_device()
    
    info = {
        "device": str(device),
        "device_type": device.type,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        })
    
    info["pytorch_version"] = torch.__version__
    
    return info


def move_to_device(model, data, device=None):
    """
    Move a model and data to the specified device, or the best available device.
    
    Args:
        model: PyTorch model to move
        data: Data to move (can be tensor, list of tensors, or dict of tensors)
        device: Target device (if None, uses get_device())
        
    Returns:
        tuple: (model on device, data on device)
    """
    if device is None:
        device = get_device()
    
    # Move model to device
    model = model.to(device)
    
    # Move data to device based on its type
    if isinstance(data, torch.Tensor):
        return model, data.to(device)
    elif isinstance(data, list):
        return model, [x.to(device) if isinstance(x, torch.Tensor) else x for x in data]
    elif isinstance(data, dict):
        return model, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    else:
        # If data is not a tensor or collection of tensors, return as is
        return model, data