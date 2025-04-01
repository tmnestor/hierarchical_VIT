"""
Unified model factory for creating and managing transformer models.
This module centralizes model creation, saving, and loading for both ViT and Swin models.
"""

import torch
import torch.nn as nn
from config import get_config

# SafeBatchNorm1d class to handle single-sample inference
class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that doesn't fail with batch size 1."""
    def forward(self, x):
        if x.size(0) == 1:
            # For single sample, skip normalization
            return x
        return super().forward(x)

class ModelFactory:
    """Factory class for creating and managing transformer models."""
    
    # Model type to HuggingFace model path mapping
    MODEL_PATHS = {
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swinv2-tiny-patch4-window8-256"
    }
    
    @classmethod
    def create_transformer(cls, model_type="vit", pretrained=True, num_classes=None, verbose=True, mode="train", offline=False):
        """Create a transformer model for receipt counting.
        
        Args:
            model_type: Type of model to create ("vit" or "swin")
            pretrained: Whether to load pretrained weights from Hugging Face
            num_classes: Number of output classes. If None, will be determined from config
            verbose: Whether to show warnings about weight initialization
            mode: "train" for training mode, "eval" for evaluation mode
            
        Returns:
            Configured transformer model
        """
        # Import transformers with environment variable to disable CV2
        import os
        import sys
        
        # Configure environment based on online/offline mode
        if offline:
            # Prevent any online access attempts
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print("Running in offline mode - model must be pre-downloaded")
        else:
            # Allow downloads if the model isn't available locally
            os.environ.pop("TRANSFORMERS_OFFLINE", None)  # Remove if set
            print("Running in online mode - will download model if needed")
        
        # Set environment variables to prefer PIL over OpenCV
        os.environ["OPENCV_NOT_AVAILABLE"] = "1"
        os.environ["VISION_PIPELINE_BACKEND"] = "PIL"
        
        import transformers
        from transformers import logging as transformers_logging
        
        # Validate model type
        model_type = model_type.lower()
        if model_type not in cls.MODEL_PATHS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_PATHS.keys())}")
        
        # Get configuration
        config = get_config()
        
        # Get number of classes from config if not provided
        if num_classes is None:
            num_classes = len(config.class_distribution)
        
        # Temporarily disable HuggingFace warnings if not verbose
        if not verbose:
            # Save previous verbosity level
            prev_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
        
        # Get our custom configuration parameters
        project_config = get_config()
        classifier_dims = project_config.get_model_param("classifier_dims", [768, 512, 256])
        dropout_rates = project_config.get_model_param("dropout_rates", [0.4, 0.4, 0.3])
        
        # Create appropriate model type
        try:
            if model_type == "vit":
                from transformers import ViTForImageClassification, ViTConfig
                if pretrained:
                    model = ViTForImageClassification.from_pretrained(
                        cls.MODEL_PATHS["vit"], 
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                else:
                    # Create from config only, no pretrained weights
                    config = ViTConfig(num_labels=num_classes)
                    model = ViTForImageClassification(config)
            elif model_type == "swin":
                # Import model without the image processing components that require OpenCV
                from transformers import AutoConfig, AutoModelForImageClassification
                
                if pretrained:
                    # Use AutoModelForImageClassification which handles model differences internally
                    try:
                        model = AutoModelForImageClassification.from_pretrained(
                            cls.MODEL_PATHS["swin"], 
                            num_labels=num_classes,
                            ignore_mismatched_sizes=True,
                            local_files_only=offline
                        )
                    except Exception as e:
                        if offline:
                            raise ValueError(
                                f"Failed to load model in offline mode. Please download the model first: {e}"
                            )
                        else:
                            # Re-raise the exception for online mode
                            raise
                else:
                    # Create from config only, no pretrained weights
                    try:
                        config = AutoConfig.from_pretrained(
                            cls.MODEL_PATHS["swin"], 
                            num_labels=num_classes,
                            local_files_only=offline
                        )
                        model = AutoModelForImageClassification.from_config(config)
                    except Exception as e:
                        if offline:
                            raise ValueError(
                                f"Failed to load model config in offline mode. Please download the model first: {e}"
                            )
                        else:
                            # Re-raise the exception for online mode
                            raise
        finally:
            # Restore previous verbosity if changed
            if not verbose:
                transformers_logging.set_verbosity(prev_verbosity)
        
        # Create unified classifier architecture
        cls._build_classifier(model, classifier_dims, dropout_rates, num_classes)
        
        # Set model mode
        if mode.lower() == "eval":
            model.eval()
        else:
            model.train()
            
        return model
    
    @staticmethod
    def _build_classifier(model, classifier_dims, dropout_rates, num_classes):
        """Build a custom classifier head for the model.
        
        Args:
            model: The transformer model
            classifier_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each layer
            num_classes: Number of output classes
        """
        # Create classification layers
        layers = []
        in_features = model.classifier.in_features
        
        # Build sequential model with the configured parameters
        for i, dim in enumerate(classifier_dims):
            layers.append(nn.Linear(in_features, dim))
            # Use SafeBatchNorm1d to handle single-sample inference
            layers.append(SafeBatchNorm1d(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rates[i]))
            in_features = dim
        
        # Add final classification layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Replace the classifier with our custom architecture
        model.classifier = nn.Sequential(*layers)
    
    @staticmethod
    def save_model(model, path):
        """Save a model's state dictionary to disk.
        
        Args:
            model: The model to save
            path: File path to save to
        """
        torch.save(model.state_dict(), path)
    
    @classmethod
    def load_model(cls, path, model_type="vit", num_classes=None, strict=True, mode="eval"):
        """Load a saved transformer model.
        
        Args:
            path: Path to the saved model weights
            model_type: Type of model to load ("vit" or "swin")
            num_classes: Number of output classes. If None, will be determined from config
            strict: Whether to enforce strict parameter matching
            mode: "train" for training mode, "eval" for evaluation mode
            
        Returns:
            Loaded model
        """
        import transformers
        # Disable warnings
        transformers.logging.set_verbosity_error()
        
        # Create empty model structure WITHOUT pretrained weights
        # (this is critical to ensure we don't mix pretrained weights with our own)
        model = cls.create_transformer(
            model_type=model_type,
            pretrained=False,  # Never use pretrained weights when loading our models
            num_classes=num_classes,
            verbose=False,
            mode=mode
        )
        
        print(f"Loading model from {path}")
        
        # Try loading with different options to handle different PyTorch versions
        try:
            # Try loading as a checkpoint dictionary first (more common for training checkpoints)
            checkpoint = torch.load(path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Found model_state_dict in checkpoint")
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                print("Found state_dict in checkpoint")
                model.load_state_dict(checkpoint['state_dict'], strict=strict)
            else:
                # Otherwise treat it as a direct state dict
                print("Loading direct state dict")
                model.load_state_dict(checkpoint, strict=strict)
                
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("WARNING: Using untrained model!")
        
        # Set model mode
        if mode.lower() == "eval":
            model.eval()
        else:
            model.train()
            
        return model


# End of ModelFactory class