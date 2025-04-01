"""
Pruning utilities for transformer models using the Lottery Ticket Hypothesis approach.

This module provides functions to implement the Lottery Ticket Hypothesis (LTH) pruning method
for Vision Transformer models. The LTH approach (Frankle & Carbin, 2019) creates sparse 
subnetworks that can be trained to similar performance as the original dense network.

Key steps in LTH:
1. Train the full model and save weights
2. Prune a percentage of the smallest-magnitude weights
3. Reset remaining weights to their ORIGINAL initialization values (crucial for LTH)
4. Retrain the pruned network
5. Optionally iterate for multiple rounds of pruning

Reference: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
           by Jonathan Frankle and Michael Carbin (ICLR 2019)
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model_factory import ModelFactory
from config import get_config


class LotteryTicketPruner:
    """
    Implements the Lottery Ticket Hypothesis pruning approach for transformer models.
    
    This class manages the process of finding "winning tickets" (sparse subnetworks)
    according to the original LTH paper. It handles weight initialization tracking,
    iterative pruning, and resetting weights to initial values.
    """
    
    def __init__(self, model_type="swin", num_classes=None, device=None, save_dir="models/pruned"):
        """
        Initialize the pruner with a new model and track its initial weights.
        
        Args:
            model_type: Type of model ("vit" or "swin")
            num_classes: Number of output classes
            device: Device to use for model (if None, will be determined automatically)
            save_dir: Directory to save pruned models
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create initial model - this is crucial for LTH as we need the initial weights
        print(f"Creating initial {model_type} model for Lottery Ticket pruning...")
        self.initial_model = ModelFactory.create_transformer(
            model_type=model_type,
            pretrained=True,  # Start with pretrained weights
            num_classes=num_classes,
            mode="train"
        )
        
        # Store a deep copy of the initial weights - critical for LTH
        self.initial_weights = self._get_weight_copy(self.initial_model)
        print(f"Stored initial weights for {len(self.initial_weights)} layers")
        
        # Initialize the current model to the initial state
        self.model = copy.deepcopy(self.initial_model).to(self.device)
        
        # Dictionary to store pruning masks for each layer
        self.masks = {}
        
        # Tracking metrics during pruning process
        self.pruning_history = {
            'pruning_percentages': [],
            'model_sizes': [],
            'param_counts': [],
            'metrics': []
        }
    
    def _get_weight_copy(self, model):
        """
        Get a deep copy of all trainable weights from the model.
        
        Args:
            model: The model to copy weights from
            
        Returns:
            dict: Dictionary mapping parameter names to their tensor copies
        """
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.clone()
        return weights
    
    def create_pruned_model(self, pruning_percentage, trained_model_path=None):
        """
        Create a pruned model according to the Lottery Ticket Hypothesis.
        
        Args:
            pruning_percentage: Percentage of weights to prune (0-100)
            trained_model_path: Path to a trained model to use for magnitude-based pruning
                               (if None, uses the current model)
                               
        Returns:
            The pruned model with weights reset to their initial values
        """
        # If a trained model is provided, load it for pruning
        if trained_model_path:
            print(f"Loading trained model from {trained_model_path} for pruning...")
            trained_model = ModelFactory.load_model(
                trained_model_path,
                model_type=self.model_type,
                num_classes=self.num_classes,
                mode="train"
            ).to(self.device)
        else:
            trained_model = self.model
        
        # Get a fresh copy of the initial model for pruning
        self.model = copy.deepcopy(self.initial_model).to(self.device)
        
        # Apply global pruning based on magnitude across the entire model
        self._apply_global_pruning(trained_model, pruning_percentage)
        
        # Reset weights to initialization values (KEY STEP for Lottery Ticket Hypothesis)
        self._reset_to_initial_weights()
        
        # Record pruning history
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        unpruned_params = 0
        masks = self._get_masks()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in masks:
                    # Count unpruned parameters (where mask is 1)
                    unpruned_params += torch.sum(masks[name]).item()
                else:
                    # If no mask, all parameters are unpruned
                    unpruned_params += param.numel()
        sparsity = 1.0 - (unpruned_params / total_params)
        
        # Update pruning history
        self.pruning_history['pruning_percentages'].append(pruning_percentage)
        self.pruning_history['model_sizes'].append(self._calc_model_size_mb())
        self.pruning_history['param_counts'].append((total_params, unpruned_params))
        
        # Print pruning summary
        print(f"Pruned {pruning_percentage:.1f}% of weights")
        print(f"Model sparsity: {sparsity:.2%}")
        print(f"Parameters: {unpruned_params:,} / {total_params:,}")
        print(f"Model size: {self._calc_model_size_mb():.2f} MB")
        
        return self.model
    
    def _get_masks(self):
        """Get the current pruning masks for all layers."""
        # Simply return our stored masks dictionary - we no longer rely on hooks
        return self.masks
    
    def _apply_global_pruning(self, trained_model, pruning_percentage):
        """
        Apply global magnitude-based pruning to the model.
        
        This prunes the bottom X% of weights across the entire model based on
        their absolute magnitude in the trained model.
        
        Args:
            trained_model: The trained model to use for magnitude-based pruning
            pruning_percentage: Percentage of weights to prune (0-100)
        """
        if pruning_percentage <= 0:
            print("No pruning requested (0% pruning rate)")
            return
        
        if pruning_percentage >= 100:
            raise ValueError("Pruning percentage must be less than 100%")
        
        # Collect all weights and their corresponding parameter names/modules
        all_weights = []
        params_to_prune = []
        
        # First, we need to find all the module and parameter names we want to prune
        for (name, param), (trained_name, trained_param) in zip(
            self.model.named_parameters(), trained_model.named_parameters()):
            
            # Only prune trainable weights, not biases or LayerNorm parameters
            if param.requires_grad and 'weight' in name and 'norm' not in name and 'layernorm' not in name:
                # For transformers, we also want to exclude certain projection layers
                if 'classifier' in name or 'head' in name:
                    # Parse the name to find the module and parameter name
                    module_name, param_name = name.rsplit('.', 1)
                    module = self.model
                    for component in module_name.split('.'):
                        if hasattr(module, component):
                            module = getattr(module, component)
                        else:
                            print(f"Warning: Could not find module {component} in {module_name}")
                            break
                    else:
                        # If we didn't break, we found the module
                        params_to_prune.append((module, param_name))
                        all_weights.append(trained_param.data.abs().flatten().cpu().numpy())
        
        if not params_to_prune:
            raise ValueError("No parameters found to prune!")
            
        # Flatten all weights into a single array for global pruning threshold computation
        all_weights_flat = np.concatenate([weights.flatten() for weights in all_weights])
        threshold_index = int(len(all_weights_flat) * (pruning_percentage / 100))
        threshold_value = np.sort(all_weights_flat)[threshold_index]
        
        print(f"Global pruning threshold: {threshold_value}")
        print(f"Applying pruning to {len(params_to_prune)} layers...")
        
        # Now apply custom pruning to each module based on the global threshold
        for (module, param_name), weights in zip(params_to_prune, all_weights):
            # For using trained model weights to determine pruning threshold
            # Find the equivalent module in the trained model
            module_path = None
            for name, mod in self.model.named_modules():
                if mod is module:
                    module_path = name
                    break
                    
            if module_path:
                trained_module = trained_model
                for component in module_path.split('.'):
                    if component and hasattr(trained_module, component):
                        trained_module = getattr(trained_module, component)
                    else:
                        if component:  # Skip empty component
                            print(f"Warning: Could not find component '{component}' in trained model")
                        break
            else:
                print(f"Warning: Could not find module path for parameter {param_name}")
                trained_module = module  # Fall back to current module
            
            # Get parameters from both model being pruned and trained model
            param = getattr(module, param_name)
            trained_param = getattr(trained_module, param_name)
            
            # Create mask based on trained weights
            mask = (trained_param.data.abs() > threshold_value).float()
            
            # Use PyTorch's CustomFromMask pruning method
            prune.CustomFromMask.apply(module, param_name, mask)
            
            # Store the full parameter name and mask for later
            full_name = None
            for name, p in self.model.named_parameters():
                if p is param:
                    full_name = name
                    break
                    
            if full_name:
                # Get the mask that PyTorch has created
                self.masks[full_name] = getattr(module, f"{param_name}_mask").data
    
    def _reset_to_initial_weights(self):
        """
        Reset the weights of unpruned connections to their initial values.
        This is a critical step in the Lottery Ticket Hypothesis.
        """
        print("Resetting unpruned weights to their initial values...")
        
        with torch.no_grad():
            # Get all modules and their parameter names
            for name, module in self.model.named_modules():
                # Skip the top-level module (model itself)
                if name == '':
                    continue
                    
                # Look for parameters in this module that have a mask
                for param_name in [n for n, _ in module.named_parameters(recurse=False)]:
                    mask_name = f"{param_name}_mask"
                    orig_name = f"{param_name}_orig"
                    
                    # Check if this parameter has been pruned
                    if hasattr(module, mask_name) and hasattr(module, orig_name):
                        # Get the mask
                        mask = getattr(module, mask_name)
                        
                        # Construct full parameter name
                        full_name = f"{name}.{param_name}"
                        
                        # Check if we have the initial weights for this parameter
                        if full_name in self.initial_weights:
                            # Reset to initial weights while maintaining the pruning
                            orig_param = getattr(module, orig_name)
                            orig_param.data = self.initial_weights[full_name].to(orig_param.device)
                        else:
                            print(f"Warning: No initial weights found for {full_name}")
                
            # Also reset parameters that weren't pruned
            for name, param in self.model.named_parameters():
                # Skip parameters that we've already handled
                if any(name.endswith(suffix) for suffix in ['_orig', '_mask']):
                    continue
                
                # Reset this parameter if we have initial weights for it
                if name in self.initial_weights:
                    param.data = self.initial_weights[name].to(param.device)
    
    def save_pruned_model(self, pruning_percentage, metrics=None):
        """
        Save the pruned model with its pruning metadata.
        
        Args:
            pruning_percentage: Percentage of weights pruned
            metrics: Optional dictionary of evaluation metrics
        """
        # Create a directory for this pruning percentage
        pruned_dir = self.save_dir / f"pruned_{pruning_percentage:.1f}pct"
        pruned_dir.mkdir(exist_ok=True)
        
        # Save the model weights
        model_path = pruned_dir / f"pruned_{self.model_type}_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pruning_percentage': pruning_percentage,
            'model_type': self.model_type,
            'metrics': metrics,
            'masks': {name: mask.cpu() for name, mask in self.masks.items()}
        }, model_path)
        
        # Also save just the model state dict for compatibility
        compat_path = pruned_dir / f"pruned_{self.model_type}_model_compat.pth"
        torch.save(self.model.state_dict(), compat_path)
        
        # Save a pruning report
        report_path = pruned_dir / "pruning_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Pruning Report for {self.model_type.upper()} Model\n")
            f.write(f"Pruning percentage: {pruning_percentage:.1f}%\n")
            f.write(f"Model size: {self._calc_model_size_mb():.2f} MB\n")
            
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            unpruned_params = 0
            masks = self._get_masks()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name in masks:
                        # Count unpruned parameters (where mask is 1)
                        unpruned_params += torch.sum(masks[name]).item()
                    else:
                        # If no mask, all parameters are unpruned
                        unpruned_params += param.numel()
            sparsity = 1.0 - (unpruned_params / total_params)
            
            f.write(f"Model sparsity: {sparsity:.2%}\n")
            f.write(f"Parameters: {unpruned_params:,} / {total_params:,}\n\n")
            
            if metrics:
                f.write("Evaluation Metrics:\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v}\n")
                    
        print(f"Saved pruned model to {model_path}")
        return model_path
    
    def _calc_model_size_mb(self):
        """Calculate the size of the model in MB."""
        torch.save(self.model.state_dict(), "temp_model_size.pth")
        size_bytes = os.path.getsize("temp_model_size.pth")
        os.remove("temp_model_size.pth")
        return size_bytes / (1024 * 1024)  # Convert bytes to MB


def iterative_lottery_ticket_pruning(
    model_type="swin",
    initial_model_path=None, 
    trained_model_path=None,
    pruning_percentages=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    training_func=None,
    evaluation_func=None,
    num_classes=None,
    save_dir="models/lottery_tickets",
    device=None
):
    """
    Perform iterative Lottery Ticket Hypothesis pruning on a model.
    
    Args:
        model_type: Type of model ("vit" or "swin")
        initial_model_path: Path to the initial model weights (optional)
        trained_model_path: Path to a trained model to use for pruning
        pruning_percentages: List of cumulative pruning percentages to try
        training_func: Function to train a pruned model (signature: model -> trained_model_path)
        evaluation_func: Function to evaluate a model (signature: model -> metrics_dict)
        num_classes: Number of output classes
        save_dir: Directory to save pruned models
        device: Device to use for model
        
    Returns:
        Dictionary of pruning results for each percentage
    """
    # Initialize results dictionary
    results = {
        'pruning_percentages': [],
        'model_sizes': [],
        'sparsities': [],
        'metrics': [],
        'model_paths': []
    }
    
    # Initialize pruner
    pruner = LotteryTicketPruner(
        model_type=model_type,
        num_classes=num_classes,
        device=device,
        save_dir=save_dir
    )
    
    if initial_model_path:
        # If initial model provided, use it to set initial weights
        initial_model = ModelFactory.load_model(
            initial_model_path, 
            model_type=model_type, 
            num_classes=num_classes
        )
        pruner.initial_weights = pruner._get_weight_copy(initial_model)
        print(f"Loaded initial weights from {initial_model_path}")
    
    # If no trained model provided, but training function is available
    if not trained_model_path and training_func:
        print("No trained model provided. Training the initial model...")
        # Train the initial model
        trained_model_path = training_func(pruner.model)
        print(f"Trained initial model saved to {trained_model_path}")
    
    if not trained_model_path:
        raise ValueError("Either trained_model_path or training_func must be provided")
    
    # Iterate through pruning percentages
    for pruning_pct in pruning_percentages:
        print(f"\n=== Pruning {pruning_pct:.1f}% of weights ===")
        
        # Create pruned model with current pruning percentage
        pruned_model = pruner.create_pruned_model(pruning_pct, trained_model_path)
        
        # Train the pruned model if training function provided
        if training_func:
            print(f"Training pruned model ({pruning_pct:.1f}% pruned)...")
            pruned_trained_path = training_func(pruned_model)
            # Load the trained pruned model
            pruned_model = ModelFactory.load_model(
                pruned_trained_path,
                model_type=model_type,
                num_classes=num_classes
            ).to(device or pruner.device)
        
        # Evaluate the pruned model if evaluation function provided
        metrics = None
        if evaluation_func:
            print(f"Evaluating pruned model ({pruning_pct:.1f}% pruned)...")
            metrics = evaluation_func(pruned_model)
            print(f"Metrics: {metrics}")
        
        # Save the pruned model with its metrics
        model_path = pruner.save_pruned_model(pruning_pct, metrics)
        
        # Calculate sparsity
        total_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        masks = pruner._get_masks()
        unpruned_params = 0
        
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                if name in masks:
                    # Count unpruned parameters (where mask is 1)
                    unpruned_params += torch.sum(masks[name]).item()
                else:
                    # If no mask, all parameters are unpruned
                    unpruned_params += param.numel()
        sparsity = 1.0 - (unpruned_params / total_params)
        
        # Update results
        results['pruning_percentages'].append(pruning_pct)
        results['model_sizes'].append(pruner._calc_model_size_mb())
        results['sparsities'].append(sparsity)
        results['metrics'].append(metrics)
        results['model_paths'].append(str(model_path))
    
    return results


def load_pruned_model(model_path, device=None):
    """
    Load a pruned model with its pruning masks.
    
    Args:
        model_path: Path to the saved pruned model
        device: Device to load the model to
        
    Returns:
        The loaded pruned model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if this is a pruned model checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        pruning_percentage = checkpoint.get('pruning_percentage', 'unknown')
        model_type = checkpoint.get('model_type', 'swin')
        masks = checkpoint.get('masks', {})
        
        print(f"Loading pruned {model_type} model ({pruning_percentage}% pruned)")
        
        # Create a new model
        num_classes = None  # Will be inferred from the state dict
        model = ModelFactory.create_transformer(
            model_type=model_type,
            pretrained=False,
            num_classes=num_classes,
            mode="eval"
        ).to(device)
        
        # Load the state dictionary
        model.load_state_dict(model_state_dict)
        
        # Apply pruning masks if available
        if masks:
            print(f"Applying {len(masks)} pruning masks")
            for name, mask in masks.items():
                try:
                    # Parse the full parameter name to find the module and parameter
                    if "." in name:
                        module_name, param_name = name.rsplit('.', 1)
                        module = model
                        for component in module_name.split('.'):
                            if hasattr(module, component):
                                module = getattr(module, component)
                            else:
                                print(f"Warning: Could not find module {component} in {module_name}")
                                break
                        else:
                            # Apply the mask using PyTorch's pruning API
                            prune.CustomFromMask.apply(module, param_name, mask.to(device))
                    else:
                        print(f"Warning: Skipping mask for {name} - unable to parse module name")
                except Exception as e:
                    print(f"Error applying mask to {name}: {e}")
        
        return model
    else:
        # If it's just a state dict
        print("Loading model from state dictionary (no pruning metadata found)")
        model = ModelFactory.load_model(model_path, mode="eval")
        return model


def create_lottery_ticket_train_func(base_train_func, model_type="swin", **kwargs):
    """
    Create a training function for lottery ticket pruning.
    
    Args:
        base_train_func: Base training function with signature: 
                        (train_csv, train_dir, val_csv, val_dir, **kwargs) -> trained_model_path
        model_type: Type of model ("vit" or "swin")
        **kwargs: Additional arguments to pass to the base_train_func
        
    Returns:
        Function with signature: model -> trained_model_path
    """
    def train_func(model):
        # Create a temporary directory to save the model
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the model to a temporary file
            temp_model_path = os.path.join(temp_dir, f"temp_{model_type}_model.pth")
            torch.save(model.state_dict(), temp_model_path)
            
            # Call the base training function with the resume parameter
            trained_model_path = base_train_func(
                resume_checkpoint=temp_model_path,
                **kwargs
            )
            
            return trained_model_path
    
    return train_func


def create_lottery_ticket_eval_func(base_eval_func, **kwargs):
    """
    Create an evaluation function for lottery ticket pruning.
    
    Args:
        base_eval_func: Base evaluation function with signature:
                       (model, **kwargs) -> metrics_dict
        **kwargs: Additional arguments to pass to the base_eval_func
        
    Returns:
        Function with signature: model -> metrics_dict
    """
    def eval_func(model):
        # Call the base evaluation function
        return base_eval_func(model, **kwargs)
    
    return eval_func


def convert_pruned_model_to_standard(model):
    """
    Convert a pruned model back to standard parameter format for compatibility
    with standard training procedures.
    
    Args:
        model: A model with pruned parameters (_orig and _mask suffixes)
        
    Returns:
        A model with standard parameter names
    """
    # Create a completely new model with the same architecture
    # This is the safest way to ensure clean, unpruned parameters
    model_type = None
    
    # Determine model type
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type == 'swinv2':
            model_type = 'swin'
        elif model.config.model_type.startswith('vit'):
            model_type = 'vit'
    
    if model_type is None:
        print("Warning: Could not determine model type, using 'swin' as default")
        model_type = 'swin'
    
    # Determine number of classes from the final layer
    num_classes = None
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        num_classes = model.classifier.out_features
    elif isinstance(model.classifier, nn.Sequential) and hasattr(model.classifier[-1], 'out_features'):
        num_classes = model.classifier[-1].out_features
    
    if num_classes is None:
        print("Warning: Could not determine number of classes, using default")
    
    # Create a new model
    from model_factory import ModelFactory
    new_model = ModelFactory.create_transformer(
        model_type=model_type,
        pretrained=True,  # Start with pretrained weights (these will be overwritten)
        num_classes=num_classes,
        mode="train"
    )
    
    # Copy all non-pruned parameters and apply masks to pruned ones
    with torch.no_grad():
        # Get state dict of the pruned model
        state_dict = {}
        for name, param in model.state_dict().items():
            # Skip mask parameters
            if '_mask' in name:
                continue
                
            # For original parameters, copy the value directly
            if '_orig' not in name:
                state_dict[name] = param.clone()
                continue
                
            # For pruned parameters, get the original name and apply mask
            orig_name = name.replace('_orig', '')
            mask_name = name.replace('_orig', '_mask')
            
            if mask_name in model.state_dict():
                # Apply mask to the original parameter
                mask = model.state_dict()[mask_name]
                pruned_param = param * mask
                state_dict[orig_name] = pruned_param
            else:
                # No mask found, just copy the original parameter
                state_dict[orig_name] = param.clone()
    
    # Load the state dict into the new model
    try:
        new_model.load_state_dict(state_dict, strict=False)
        print("Successfully converted pruned model to standard format")
    except Exception as e:
        print(f"Warning: Error loading converted state dict: {e}")
        print("Using the new model with default weights")
    
    return new_model