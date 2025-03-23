import torch
import torch.nn as nn
from config import get_config

def create_swin_transformer(pretrained=True, num_classes=None, verbose=True):
    """Create a Swin Transformer model for receipt counting using HuggingFace transformers.
    
    Args:
        pretrained: Whether to load pretrained weights from Hugging Face
        num_classes: Number of output classes. If None, will be determined from config
        verbose: Whether to show warnings about weight initialization
        
    Returns:
        Configured Swin model
    """
    from transformers import SwinForImageClassification
    import logging
    import transformers
    
    # Get number of classes from config if not provided
    if num_classes is None:
        config = get_config()
        num_classes = len(config.class_distribution)
    
    # Temporarily disable HuggingFace warnings if not verbose
    if not verbose:
        # Save previous verbosity level
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    
    # Load model from Hugging Face
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", 
        num_labels=num_classes,  # Classification task based on config
        ignore_mismatched_sizes=True
    )
    
    # Restore previous verbosity if changed
    if not verbose:
        transformers.logging.set_verbosity(prev_verbosity)
    
    # Replace classification head with stronger regularization and increased capacity
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 768),  # Larger first hidden layer
        nn.BatchNorm1d(768),
        nn.GELU(),
        nn.Dropout(0.4),  # Increased dropout
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.4),  # Increased dropout
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model

# Function to save a model state dict
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load a saved model
def load_swin_model(path, num_classes=None, strict=True):
    """Load a saved Swin model with option for strict parameter matching.
    
    Args:
        path: Path to the saved model weights
        num_classes: Number of output classes. If None, will be determined from config
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        
    Returns:
        Loaded model
    """
    # We use pretrained=False to create an empty model structure without HuggingFace pretrained weights
    # This avoids downloading unnecessary weights that would be immediately overwritten
    # Our saved weights will contain everything needed for the model
    model = create_swin_transformer(pretrained=False, num_classes=num_classes, verbose=False)
    
    # Load our own pretrained weights from the saved file
    model.load_state_dict(torch.load(path), strict=strict)
    return model
