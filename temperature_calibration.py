#!/usr/bin/env python3
"""
Temperature calibration for model confidence scores.
This implementation is based on the paper:
"On Calibration of Modern Neural Networks" by Guo et al. (ICML 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
from tqdm import tqdm

from datasets import ReceiptDataset
from device_utils import get_device
from train_simplified_hierarchical import create_simplified_model

class ModelWithTemperature(nn.Module):
    """
    A wrapper for a classification model that applies temperature scaling
    to the model's predictions.
    """
    def __init__(self, model, temperature=1.0):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, input_):
        """
        Forward pass with temperature scaling.
        
        Args:
            input_: Input tensor
            
        Returns:
            Scaled softmax probabilities
        """
        # Pass input through the model
        output = self.model(input_)
        
        # Handle different model output formats
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
            
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Return scaled probabilities
        return F.softmax(scaled_logits, dim=1)
    
    def get_temp(self):
        """Get the current temperature value."""
        return self.temperature.item()
    
def calibrate_model(model, valid_loader, device, max_iter=50, lr=0.01, init_temp=1.0):
    """
    Calibrate a model's confidence using temperature scaling.
    
    Args:
        model: The model to calibrate
        valid_loader: DataLoader with validation data
        device: Device to run the calibration on
        max_iter: Maximum number of optimization iterations
        lr: Learning rate for the optimizer
        init_temp: Initial temperature value
        
    Returns:
        calibrated_model: Model with calibrated temperature
        temperature: Learned temperature value
        nll_history: History of negative log-likelihood values
    """
    # Create a wrapper model with temperature scaling
    temperature_model = ModelWithTemperature(model, init_temp)
    temperature_model.to(device)
    
    # Only optimize the temperature parameter
    optimizer = optim.LBFGS([temperature_model.temperature], lr=lr, max_iter=max_iter)
    
    # Keep track of NLL values
    nll_history = []
    
    def eval_nll():
        """Evaluate negative log likelihood loss on validation data."""
        nll_criterion = nn.CrossEntropyLoss().to(device)
        temperature_model.eval()
        total_nll = 0
        total_samples = 0
        
        # We need to keep gradients for the temperature parameter
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Get logits from the base model
            with torch.no_grad():  # Don't track gradients for model parameters
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            
            # Apply temperature scaling - keep gradients for temperature
            scaled_logits = logits / temperature_model.temperature
            
            # Calculate loss
            loss = nll_criterion(scaled_logits, targets)
            total_nll += loss.item() * batch_size
        
        # We need to recompute the final loss with gradients
        # For efficiency, we'll use a subset of the data for the backward pass
        sample_inputs, sample_targets = next(iter(valid_loader))
        sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
        
        with torch.no_grad():  # Don't track gradients for model parameters
            outputs = model(sample_inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
        # Apply temperature scaling with gradients
        scaled_logits = logits / temperature_model.temperature
        loss_with_grad = nll_criterion(scaled_logits, sample_targets)
        
        # Store history
        avg_nll = total_nll / total_samples
        nll_history.append(avg_nll)
        
        # Print current values
        temp_val = temperature_model.temperature.item()
        print(f"Temperature: {temp_val:.4f}, NLL: {avg_nll:.6f}")
        
        return loss_with_grad
    
    # Define the optimization step
    def step_fn():
        optimizer.zero_grad()
        loss = eval_nll()
        loss.backward()
        return loss
    
    # Run the optimization
    print("Optimizing temperature parameter...")
    optimizer.step(step_fn)
    
    # Final temperature value
    final_temp = temperature_model.temperature.item()
    print(f"Final temperature: {final_temp:.4f}")
    
    return temperature_model, final_temp, nll_history

def visualize_calibration(model, temp_model, data_loader, device, num_bins=10, output_path=None):
    """
    Visualize the calibration of a model before and after temperature scaling.
    
    Args:
        model: Original model
        temp_model: Temperature-calibrated model
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        num_bins: Number of bins for reliability diagram
        output_path: Path to save visualization
    """
    # Collect predictions and targets
    originals = {"confidences": [], "predictions": [], "targets": []}
    calibrated = {"confidences": [], "predictions": [], "targets": []}
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Original model
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
            else:
                logits = outputs
                probs = F.softmax(logits, dim=1)
                
            # Get predictions and confidences
            _, preds = torch.max(probs, 1)
            confs = torch.max(probs, 1)[0]
            
            # Store original model results
            originals["confidences"].extend(confs.cpu().numpy())
            originals["predictions"].extend(preds.cpu().numpy())
            originals["targets"].extend(targets.cpu().numpy())
            
            # Calibrated model
            cal_probs = temp_model(inputs)
            _, cal_preds = torch.max(cal_probs, 1)
            cal_confs = torch.max(cal_probs, 1)[0]
            
            # Store calibrated model results
            calibrated["confidences"].extend(cal_confs.cpu().numpy())
            calibrated["predictions"].extend(cal_preds.cpu().numpy())
            calibrated["targets"].extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    for key in ["confidences", "predictions", "targets"]:
        originals[key] = np.array(originals[key])
        calibrated[key] = np.array(calibrated[key])
    
    # Calculate ECE (Expected Calibration Error)
    def compute_ece(confidences, predictions, targets, num_bins):
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_sizes = []
        bin_accs = []
        bin_confs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            bin_indices = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            if not np.any(bin_indices):
                bin_sizes.append(0)
                bin_accs.append(0)
                bin_confs.append(0)
                continue
                
            # Calculate accuracy and confidence for this bin
            bin_size = np.sum(bin_indices)
            bin_acc = np.mean(predictions[bin_indices] == targets[bin_indices])
            bin_conf = np.mean(confidences[bin_indices])
            
            # Store values
            bin_sizes.append(bin_size)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            
            # Add to ECE
            ece += (bin_size / len(confidences)) * np.abs(bin_acc - bin_conf)
        
        return ece, bin_accs, bin_confs, bin_sizes
    
    # Calculate ECE for original and calibrated models
    orig_ece, orig_accs, orig_confs, orig_sizes = compute_ece(
        originals["confidences"], 
        originals["predictions"], 
        originals["targets"], 
        num_bins
    )
    
    cal_ece, cal_accs, cal_confs, cal_sizes = compute_ece(
        calibrated["confidences"], 
        calibrated["predictions"], 
        calibrated["targets"], 
        num_bins
    )
    
    # Calculate accuracies
    orig_acc = np.mean(originals["predictions"] == originals["targets"])
    cal_acc = np.mean(calibrated["predictions"] == calibrated["targets"])
    
    # Create reliability diagrams
    plt.figure(figsize=(15, 8))
    
    # Original model reliability diagram
    plt.subplot(1, 2, 1)
    bin_centers = np.linspace(0.05, 0.95, num_bins)
    plt.bar(bin_centers, orig_accs, width=0.1, alpha=0.8, label='Accuracy', color='b')
    plt.bar(bin_centers, orig_confs, width=0.1, alpha=0.2, label='Confidence', color='r')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Original Model\nECE: {orig_ece:.4f}, Accuracy: {orig_acc:.4f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Calibrated model reliability diagram
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, cal_accs, width=0.1, alpha=0.8, label='Accuracy', color='b')
    plt.bar(bin_centers, cal_confs, width=0.1, alpha=0.2, label='Confidence', color='r')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibrated Model (T={temp_model.get_temp():.2f})\nECE: {cal_ece:.4f}, Accuracy: {cal_acc:.4f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()
    
    # Return calibration metrics
    return {
        "original_ece": orig_ece,
        "calibrated_ece": cal_ece,
        "original_accuracy": orig_acc,
        "calibrated_accuracy": cal_acc,
        "temperature": temp_model.get_temp()
    }

def save_calibrated_model(model, temperature, output_path):
    """
    Save a calibrated model.
    
    Args:
        model: Original model
        temperature: Calibrated temperature value
        output_path: Path to save the calibrated model
    """
    # Save the model state dict
    torch.save({
        "model_state_dict": model.state_dict(),
        "temperature": temperature
    }, output_path)
    print(f"Calibrated model saved to {output_path}")

def load_calibrated_model(model_path, model_type, device=None):
    """
    Load a calibrated model.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('vit' or 'swin')
        device: Device to load the model to
        
    Returns:
        calibrated_model: Model with calibrated temperature
    """
    if device is None:
        device = get_device()
        
    # Load model state dict and temperature
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create base model
    base_model = create_simplified_model(model_type, num_classes=2)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model = base_model.to(device)
    
    # Create temperature model
    temperature = checkpoint.get("temperature", 1.0)
    temp_model = ModelWithTemperature(base_model, temperature)
    temp_model = temp_model.to(device)
    
    return temp_model

def calibrate_hierarchical_models(
    level1_model_path,
    level2_model_path,
    val_csv,
    val_dir,
    model_type,
    output_dir,
    batch_size=32
):
    """
    Calibrate hierarchical models for receipt counting.
    
    Args:
        level1_model_path: Path to Level 1 model
        level2_model_path: Path to Level 2 model
        val_csv: Path to validation CSV
        val_dir: Path to validation directory
        model_type: Type of model ('vit' or 'swin')
        output_dir: Directory to save calibrated models
        batch_size: Batch size for calibration
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    level1_model = create_simplified_model(model_type, num_classes=2)
    level1_model.load_state_dict(torch.load(level1_model_path, map_location=device))
    level1_model = level1_model.to(device)
    level1_model.eval()
    
    level2_model = create_simplified_model(model_type, num_classes=2)
    level2_model.load_state_dict(torch.load(level2_model_path, map_location=device))
    level2_model = level2_model.to(device)
    level2_model.eval()
    
    # Create validation datasets for each level
    print("Creating validation datasets...")
    level1_dataset = ReceiptDataset(
        csv_file=val_csv,
        img_dir=val_dir,
        augment=False,
        hierarchical_level="level1"
    )
    
    level1_loader = DataLoader(
        level1_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    level2_dataset = ReceiptDataset(
        csv_file=val_csv,
        img_dir=val_dir,
        augment=False,
        hierarchical_level="level2"
    )
    
    level2_loader = DataLoader(
        level2_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Calibrate Level 1 model
    print("\nCalibrating Level 1 model...")
    level1_temp_model, level1_temp, level1_nll_history = calibrate_model(
        model=level1_model,
        valid_loader=level1_loader,
        device=device
    )
    
    # Visualize Level 1 calibration
    level1_metrics = visualize_calibration(
        model=level1_model,
        temp_model=level1_temp_model,
        data_loader=level1_loader,
        device=device,
        output_path=output_path / f"{model_type}_level1_calibration.png"
    )
    
    # Save Level 1 calibrated model
    level1_output_path = output_path / f"{model_type}_level1_calibrated.pth"
    save_calibrated_model(level1_model, level1_temp, level1_output_path)
    
    # Calibrate Level 2 model
    print("\nCalibrating Level 2 model...")
    level2_temp_model, level2_temp, level2_nll_history = calibrate_model(
        model=level2_model,
        valid_loader=level2_loader,
        device=device
    )
    
    # Visualize Level 2 calibration
    level2_metrics = visualize_calibration(
        model=level2_model,
        temp_model=level2_temp_model,
        data_loader=level2_loader,
        device=device,
        output_path=output_path / f"{model_type}_level2_calibration.png"
    )
    
    # Save Level 2 calibrated model
    level2_output_path = output_path / f"{model_type}_level2_calibrated.pth"
    save_calibrated_model(level2_model, level2_temp, level2_output_path)
    
    # Save calibration metrics
    import json
    metrics = {
        "level1": level1_metrics,
        "level2": level2_metrics
    }
    
    with open(output_path / f"{model_type}_calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\nCalibration complete!")
    print(f"Level 1 temperature: {level1_temp:.4f}")
    print(f"Level 2 temperature: {level2_temp:.4f}")
    print(f"Models saved to {output_path}")
    
    return {
        "level1_temperature": level1_temp,
        "level2_temperature": level2_temp,
        "level1_metrics": level1_metrics,
        "level2_metrics": level2_metrics
    }

def main():
    """Parse arguments and run calibration."""
    parser = argparse.ArgumentParser(description="Calibrate hierarchical models")
    
    # Model options
    parser.add_argument("--level1_model", required=True,
                      help="Path to Level 1 model")
    parser.add_argument("--level2_model", required=True,
                      help="Path to Level 2 model")
    parser.add_argument("--model_type", choices=["vit", "swin"], default="swin",
                      help="Model type (vit or swin)")
    
    # Data options
    parser.add_argument("--val_csv", required=True,
                      help="Path to validation CSV file")
    parser.add_argument("--val_dir", required=True,
                      help="Directory containing validation images")
    
    # Output options
    parser.add_argument("--output_dir", default="models/calibrated",
                      help="Directory to save calibrated models")
    
    # Batch size
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for calibration")
    
    args = parser.parse_args()
    
    calibrate_hierarchical_models(
        level1_model_path=args.level1_model,
        level2_model_path=args.level2_model,
        val_csv=args.val_csv,
        val_dir=args.val_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()