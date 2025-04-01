#!/usr/bin/env python
"""
Lottery Ticket Hypothesis demonstration for Vision Transformers.

This script demonstrates the Lottery Ticket Hypothesis on a small example,
showing how pruned networks can maintain or improve performance when properly initialized.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from prune_utils import LotteryTicketPruner, convert_pruned_model_to_standard
from model_factory import ModelFactory
from device_utils import get_device
from config import get_config


def train_model_with_swin_trainer(
    model,
    train_csv,
    train_dir,
    val_csv,
    val_dir,
    epochs=5,
    batch_size=16,
    lr=1e-4,
    binary=True,
    output_dir="temp_outputs"
):
    """
    Train a model using the train_swin_classification.py module.
    
    Args:
        model: PyTorch model to train (initial model)
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file
        val_dir: Directory containing validation images
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        binary: Whether to use binary classification
        output_dir: Directory to save trained model
        
    Returns:
        Tuple of (trained model, training history)
    """
    from train_swin_classification import train_model
    import os
    import pandas as pd
    import tempfile
    
    # Create temporary directory to save the model
    temp_dir = tempfile.mkdtemp(prefix="lottery_ticket_")
    temp_model_path = os.path.join(temp_dir, "initial_model.pth")
    
    # Convert pruned model to standard format
    print("Converting pruned model to standard format for training compatibility...")
    model = convert_pruned_model_to_standard(model)
    
    # Save the converted model state
    torch.save(model.state_dict(), temp_model_path)
    
    # Create a unique output directory to avoid overwriting files
    temp_output_dir = os.path.join(output_dir, f"temp_training_{os.path.basename(temp_dir)}")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Train the model using train_swin_classification's train_model function
    trained_model = train_model(
        train_csv=train_csv,
        train_dir=train_dir,
        val_csv=val_csv,
        val_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=temp_output_dir,
        binary=binary,
        augment=True,
        resume_checkpoint=temp_model_path
    )
    
    # Load training history if available
    history_path = os.path.join(temp_output_dir, "swin_classification_history.csv")
    history = {}
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history = {
            'train_loss': history_df['train_loss'].tolist(),
            'val_loss': history_df['val_loss'].tolist(),
            'val_accuracy': history_df['val_acc'].tolist(),
            'val_balanced_accuracy': history_df['val_balanced_acc'].tolist(),
            'val_f1_macro': history_df['val_f1_macro'].tolist()
        }
    else:
        # Create empty history if file doesn't exist
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_balanced_accuracy': [],
            'val_f1_macro': []
        }
    
    # Clean up temporary initial model
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    return trained_model, history


def lottery_ticket_demo(
    train_csv,
    train_dir,
    val_csv,
    val_dir,
    output_dir="demo_outputs",
    model_type="swin",
    pruning_percentages=[0, 20, 50, 80, 90, 95],
    epochs=5,
    batch_size=8,
    binary=False,  # Change default to False for multi-class classification
    seed=42,
    workers=1,
    lr=5e-5  # Add learning rate parameter aligned with train_swin_classification.py
):
    """
    Run a demonstration of the Lottery Ticket Hypothesis.
    
    Args:
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file
        val_dir: Directory containing validation images
        output_dir: Directory to save results
        model_type: Type of model ("vit" or "swin")
        pruning_percentages: List of pruning percentages to try
        epochs: Number of training epochs
        batch_size: Batch size for training
        binary: Whether to use binary classification
        seed: Random seed for reproducibility
        workers: Number of dataloader workers
        
    Returns:
        Dictionary with results
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set binary mode for config if needed
    config = get_config()
    if binary:
        config.set_binary_mode(True)
        print("Using binary classification (0 vs 1+ receipts)")
    else:
        config.set_binary_mode(False)
        print(f"Using multi-class classification (0-{len(config.class_distribution)-1} receipts)")
    
    # We'll use train_swin_classification.py directly instead of creating data loaders here
    print(f"Training using train_swin_classification.py with data from:")
    print(f"  Train: {train_csv} ({train_dir})")
    print(f"  Validation: {val_csv} ({val_dir})")
    
    # Get the number of classes based on configuration
    num_classes = 2 if binary else len(config.class_distribution)
    print(f"Using {'binary' if binary else 'multi-class'} classification with {num_classes} classes")
    
    # Create pruner
    pruner = LotteryTicketPruner(
        model_type=model_type,
        num_classes=num_classes,
        device=device,
        save_dir=output_dir
    )
    
    # Training results
    results = {
        'pruning_percentages': pruning_percentages,
        'final_metrics': [],
        'histories': []
    }
    
    # Train models with different pruning rates
    for pruning_pct in pruning_percentages:
        print(f"\n{'='*80}")
        print(f"TRAINING WITH {pruning_pct}% PRUNING")
        print(f"{'='*80}")
        
        # Create pruned model
        if pruning_pct == 0:
            # Use the original model without pruning
            model = pruner.model
        else:
            # Create pruned model with current percentage
            model = pruner.create_pruned_model(pruning_pct)
        
        # Train model using the Swin trainer
        trained_model, history = train_model_with_swin_trainer(
            model=model,
            train_csv=train_csv,
            train_dir=train_dir,
            val_csv=val_csv,
            val_dir=val_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            binary=binary,
            output_dir=os.path.join(output_dir, f"train_{pruning_pct}pct")
        )
        
        # Save model
        model_dir = output_path / f"pruned_{pruning_pct}pct"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{model_type}_model.pth"
        torch.save(trained_model.state_dict(), model_path)
        
        # Save final metrics
        final_metrics = {
            'val_loss': history['val_loss'][-1],
            'val_accuracy': history['val_accuracy'][-1],
            'val_balanced_accuracy': history['val_balanced_accuracy'][-1],
            'val_f1_macro': history['val_f1_macro'][-1]
        }
        
        # Update results
        results['final_metrics'].append(final_metrics)
        results['histories'].append(history)
        
        print(f"Model saved to {model_path}")
        print(f"Final metrics: {final_metrics}")
    
    # Plot learning curves
    plot_learning_curves(results, output_path)
    
    # Plot final metrics vs pruning percentage
    plot_metrics_vs_pruning(results, output_path)
    
    return results


def plot_learning_curves(results, output_dir):
    """
    Plot learning curves for each pruning percentage.
    
    Args:
        results: Dictionary with training results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    
    # Plot learning curves for each pruning percentage
    for i, pruning_pct in enumerate(results['pruning_percentages']):
        history = results['histories'][i]
        
        plt.figure(figsize=(12, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Loss Curves - {pruning_pct}% Pruning')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['val_accuracy'], label='Accuracy')
        plt.title(f'Validation Accuracy - {pruning_pct}% Pruning')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot validation balanced accuracy
        plt.subplot(2, 2, 3)
        plt.plot(history['val_balanced_accuracy'], label='Balanced Accuracy')
        plt.title(f'Validation Balanced Accuracy - {pruning_pct}% Pruning')
        plt.xlabel('Epoch')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot validation F1 score
        plt.subplot(2, 2, 4)
        plt.plot(history['val_f1_macro'], label='F1 Macro')
        plt.title(f'Validation F1 Score - {pruning_pct}% Pruning')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / f"learning_curves_{pruning_pct}pct.png")
        plt.close()


def plot_metrics_vs_pruning(results, output_dir):
    """
    Plot final metrics vs pruning percentage.
    
    Args:
        results: Dictionary with training results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    
    # Extract final metrics
    pruning_pcts = results['pruning_percentages']
    val_acc = [metrics['val_accuracy'] for metrics in results['final_metrics']]
    val_bal_acc = [metrics['val_balanced_accuracy'] for metrics in results['final_metrics']]
    val_f1 = [metrics['val_f1_macro'] for metrics in results['final_metrics']]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot all metrics on one plot
    plt.plot(pruning_pcts, val_acc, 'o-', label='Accuracy')
    plt.plot(pruning_pcts, val_bal_acc, 's-', label='Balanced Accuracy')
    plt.plot(pruning_pcts, val_f1, '^-', label='F1 Macro')
    
    plt.title('Model Performance vs. Pruning Percentage')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.legend()
    
    # Mark the best model
    best_idx = np.argmax(val_f1)
    best_pct = pruning_pcts[best_idx]
    best_f1 = val_f1[best_idx]
    
    plt.annotate(f'Best: {best_pct}% pruning\nF1={best_f1:.4f}',
                xy=(best_pct, best_f1), xytext=(best_pct+5, best_f1-0.05),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig(output_path / "metrics_vs_pruning.png")
    plt.close()
    
    # Also save as CSV
    import pandas as pd
    df = pd.DataFrame({
        'Pruning_Percentage': pruning_pcts,
        'Accuracy': val_acc,
        'Balanced_Accuracy': val_bal_acc,
        'F1_Macro': val_f1
    })
    df.to_csv(output_path / "metrics_vs_pruning.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Lottery Ticket Hypothesis demonstration for Vision Transformers"
    )
    
    # Data options
    parser.add_argument(
        "--train_csv", "-tc", required=True,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--train_dir", "-td", required=True,
        help="Directory containing training images"
    )
    parser.add_argument(
        "--val_csv", "-vc", required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--val_dir", "-vd", required=True,
        help="Directory containing validation images"
    )
    
    # Model and training options
    parser.add_argument(
        "--output_dir", "-o", default="demo_outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_type", choices=["vit", "swin"], default="swin",
        help="Type of model (vit or swin)"
    )
    parser.add_argument(
        "--pruning_percentages", "-p", type=float, nargs="+",
        default=[0, 20, 50, 80, 90, 95],
        help="Pruning percentages to try"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Use binary classification (0 vs 1+ receipts) instead of multi-class (default: multi-class)"
    )
    parser.add_argument(
        "--multi-class", action="store_true", 
        help="Use multi-class classification (0-4 receipts) - this is the default"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=5e-5,
        help="Learning rate for training"
    )
    
    args = parser.parse_args()
    
    # Determine classification mode
    binary_mode = args.binary and not args.multi_class
    if args.binary and args.multi_class:
        print("Warning: Both --binary and --multi-class flags provided. Using multi-class mode.")
        binary_mode = False
    
    # Run demonstration
    results = lottery_ticket_demo(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        val_csv=args.val_csv,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        pruning_percentages=args.pruning_percentages,
        epochs=args.epochs,
        batch_size=args.batch_size,
        binary=binary_mode,
        seed=args.seed,
        workers=args.workers,
        lr=args.lr
    )
    
    # Print summary
    print("\nLottery Ticket Hypothesis Demonstration Summary:")
    print("-" * 50)
    
    # Find the best model
    f1_scores = [metrics['val_f1_macro'] for metrics in results['final_metrics']]
    best_idx = np.argmax(f1_scores)
    best_pct = results['pruning_percentages'][best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Best model: {best_pct}% pruning with F1={best_f1:.4f}")
    
    # Check if we found a winning ticket
    baseline_f1 = results['final_metrics'][0]['val_f1_macro']  # 0% pruning (original model)
    
    print(f"Original model (0% pruning): F1={baseline_f1:.4f}")
    
    winning_tickets = []
    for i, pct in enumerate(results['pruning_percentages']):
        if i == 0:  # Skip the original model
            continue
            
        f1 = results['final_metrics'][i]['val_f1_macro']
        if f1 >= baseline_f1:
            winning_tickets.append((pct, f1))
    
    if winning_tickets:
        print("\nWinning tickets found!")
        for pct, f1 in winning_tickets:
            improvement = ((f1 - baseline_f1) / baseline_f1) * 100
            print(f"  {pct}% pruning: F1={f1:.4f} ({improvement:.2f}% improvement)")
    else:
        print("\nNo winning tickets found in this demonstration")
        print("Try increasing the number of epochs or using a different random seed")


if __name__ == "__main__":
    main()