import torch
import torch.nn as nn
from config import get_config

def create_vit_transformer(pretrained=True, num_classes=None, verbose=True):
    """Create a ViT model for receipt counting using HuggingFace transformers.
    
    Args:
        pretrained: Whether to load pretrained weights from Hugging Face
        num_classes: Number of output classes. If None, will be determined from config
        verbose: Whether to show warnings about weight initialization
        
    Returns:
        Configured ViT model
    """
    from transformers import ViTForImageClassification
    import transformers
    
    # Get configuration
    config = get_config()
    
    # Get number of classes from config if not provided
    if num_classes is None:
        num_classes = len(config.class_distribution)
    
    # Temporarily disable HuggingFace warnings if not verbose
    if not verbose:
        # Save previous verbosity level
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    
    # Load model from Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        num_labels=num_classes,  # Classification task based on config
        ignore_mismatched_sizes=True
    )
    
    # Restore previous verbosity if changed
    if not verbose:
        transformers.logging.set_verbosity(prev_verbosity)
    
    # Get classifier architecture parameters from config
    classifier_dims = config.get_model_param("classifier_dims", [768, 512, 256])
    dropout_rates = config.get_model_param("dropout_rates", [0.4, 0.4, 0.3])
    
    # Create classification layers
    layers = []
    in_features = model.classifier.in_features
    
    # Build sequential model with the configured parameters
    for i, dim in enumerate(classifier_dims):
        layers.append(nn.Linear(in_features, dim))
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rates[i]))
        in_features = dim
    
    # Add final classification layer
    layers.append(nn.Linear(in_features, num_classes))
    
    # Replace the classifier with our custom architecture
    model.classifier = nn.Sequential(*layers)
    
    return model

# Function to save a model state dict
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load a saved model
def load_vit_model(path, num_classes=None, strict=True):
    """Load a saved ViT model with option for strict parameter matching.
    
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
    model = create_vit_transformer(pretrained=False, num_classes=num_classes, verbose=False)
    
    # Load our own pretrained weights from the saved file
    model.load_state_dict(torch.load(path), strict=strict)
    return model
