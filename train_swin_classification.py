import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os  # Keep for environment variables and some legacy functionality
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_factory import ModelFactory
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from config import get_config
from training_utils import validate, plot_confusion_matrix, plot_training_curves, ModelCheckpoint, EarlyStopping
from device_utils import get_device
from reproducibility import set_seed, get_reproducibility_info


# Import dataset classes from the unified module
from datasets import ReceiptDataset, create_data_loaders


# Using unified validation and plotting functions from training_utils.py


def train_model(
    train_csv,
    train_dir,
    val_csv=None,
    val_dir=None,
    epochs=15,
    batch_size=16,
    lr=1e-4,
    output_dir="models",
    binary=False,
    augment=True,
    resume_checkpoint=None,
):
    """
    Train the Swin Transformer model for receipt counting as a classification task.
    
    Args:
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file (optional)
        val_dir: Directory containing validation images (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        output_dir: Directory to save trained model and results
        binary: If True, use binary classification (0 vs 1+ receipts)
        augment: If True, apply data augmentation during training
        resume_checkpoint: Path to checkpoint to resume training from (optional)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure for binary classification if requested
    config = get_config()
    if binary:
        config.set_binary_mode(True)
        print("Using binary classification (multiple receipts or not)")
    else:
        config.set_binary_mode(False)
        print(f"Using multi-class classification (0-{len(config.class_distribution)-1} receipts)")

    # Get number of workers from config
    num_workers = config.get_model_param("num_workers", 4)
    
    # Create data loaders using the unified function
    train_loader, val_loader, num_train_samples, num_val_samples = create_data_loaders(
        train_csv=Path(train_csv),
        train_dir=Path(train_dir),
        val_csv=Path(val_csv) if val_csv else None,
        val_dir=Path(val_dir) if val_dir else None,
        batch_size=batch_size,
        augment_train=augment,
        binary=binary
    )

    # Set random seed for reproducibility
    seed_info = set_seed()
    print(f"Using random seed: {seed_info['seed']}, deterministic mode: {seed_info['deterministic']}")
    
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model or load from checkpoint
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        print(f"Loading model checkpoint from {checkpoint_path}")
        model = ModelFactory.load_model(checkpoint_path, model_type="swin").to(device)
        print("Resumed Swin Transformer model from checkpoint")
    else:
        model = ModelFactory.create_transformer(model_type="swin", pretrained=True).to(device)
        print("Initialized new Swin Transformer model using Hugging Face transformers")

    # Loss and optimizer with more robust learning rate control
    # Get class weights from configuration system
    print(f"Using class distribution: {config.class_distribution}")
    print(f"Using calibration factors: {config.calibration_factors}")
    
    # Get the normalized and scaled weights tensor for loss function
    normalized_weights = config.get_class_weights_tensor(device)
    print(f"Using class weights: {normalized_weights}")
    
    # Get optimizer parameters from config
    label_smoothing = config.get_model_param("label_smoothing", 0.1)
    weight_decay = config.get_model_param("weight_decay", 0.01)
    lr_scheduler_factor = config.get_model_param("lr_scheduler_factor", 0.5)
    lr_scheduler_patience = config.get_model_param("lr_scheduler_patience", 2)
    min_lr = config.get_model_param("min_lr", 1e-6)
    
    # Use label smoothing along with class weights to improve generalization
    criterion = nn.CrossEntropyLoss(
        weight=normalized_weights,
        label_smoothing=label_smoothing  # Add label smoothing to help with overfitting
    )  # Weighted classification loss with smoothing
    
    # Use different learning rates for backbone and classification head
    # Typically, backbone needs smaller learning rate since it's pretrained
    backbone_lr = lr * config.get_model_param("backbone_lr_multiplier", 0.1)
    
    # Create parameter groups with different learning rates
    parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': lr}
    ]
    
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    print(f"Using learning rates - Backbone: {backbone_lr}, Classifier: {lr}")
    
    # Use ReduceLROnPlateau scheduler to reduce LR when validation metrics plateau
    # This helps prevent erratic bouncing around the optimum
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Monitor balanced accuracy which we want to maximize
        factor=lr_scheduler_factor,  # Multiply LR by this factor on plateau
        patience=lr_scheduler_patience,  # Number of epochs with no improvement before reducing LR
        verbose=True,        # Print message when LR is reduced
        min_lr=min_lr        # Don't reduce LR below this value
    )

    # Training metrics
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "val_f1_macro": []}

    # For early stopping - get patience from config
    patience = config.get_model_param("early_stopping_patience", 8)
    patience_counter = 0
    best_balanced_acc = 0
    best_f1_macro = 0  # Also track F1 macro improvement

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # HuggingFace models return an object with logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                loss = criterion(logits, targets)
            else:
                loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping to prevent large updates
            gradient_clip_value = config.get_model_param("gradient_clip_value", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Learning rate scheduler will be updated after validation

        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validate using the unified validation function
        metrics = validate(model, val_loader, criterion, device)
        
        # Extract metrics and update history
        val_loss = metrics['loss']
        val_acc = metrics['accuracy']
        val_balanced_acc = metrics['balanced_accuracy']
        val_f1_macro = metrics['f1_macro']
        predictions = metrics['predictions']
        ground_truth = metrics['targets']
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_f1_macro"].append(val_f1_macro)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Balanced Accuracy: {val_balanced_acc:.2%}, "
            f"F1 Macro: {val_f1_macro:.2%}"
        )
        
        # Update learning rate scheduler based on balanced accuracy
        scheduler.step(val_f1_macro)

        # Use the ModelCheckpoint utility to save models based on metrics
        checkpoint = ModelCheckpoint(
            output_dir=output_dir,
            metrics=["balanced_accuracy", "f1_macro"],
            mode="max",
            verbose=True
        )
        
        # Check if any metric has improved and save the model if needed
        improved = checkpoint.check_improvement(
            metrics_dict=metrics,
            model=model,
            model_type="swin"
        )
        
        # Use the EarlyStopping utility to decide whether to stop training
        # Monitor F1 macro instead of balanced accuracy
        early_stopping = EarlyStopping(patience=patience, mode="max", verbose=True)
        if early_stopping.check_improvement(val_f1_macro):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    ModelFactory.save_model(model, output_path / "receipt_counter_swin_final.pth")

    # Generate validation plots using the unified functions
    metrics = validate(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    accuracy, balanced_accuracy = plot_confusion_matrix(
        metrics['predictions'],
        metrics['targets'],
        output_path=output_path / "swin_classification_results.png",
    )

    # Save training history
    pd.DataFrame(history).to_csv(
        output_path / "swin_classification_history.csv", index=False
    )

    # Plot training curves using the unified function
    plot_training_curves(
        history,
        output_path=output_path / "swin_classification_curves.png"
    )

    print("\nFinal Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")

    print(f"\nTraining complete! Models saved to {output_path}/")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Swin-Tiny model for receipt counting as classification"
    )
    
    # Data input options
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        "--train_csv", "-tc",
        default="receipt_dataset/train.csv",
        help="Path to CSV file containing training data",
    )
    data_group.add_argument(
        "--train_dir", "-td",
        default="receipt_dataset/train",
        help="Directory containing training images",
    )
    data_group.add_argument(
        "--val_csv", "-vc",
        default="receipt_dataset/val.csv",
        help="Path to CSV file containing validation data",
    )
    data_group.add_argument(
        "--val_dir", "-vd",
        default="receipt_dataset/val",
        help="Directory containing validation images",
    )
    data_group.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation during training"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        "--epochs", "-e", type=int, default=30, 
        help="Number of training epochs (default: 30)"
    )
    training_group.add_argument(
        "--batch_size", "-b", type=int, default=16, 
        help="Batch size for training (default: 16)"
    )
    training_group.add_argument(
        "--lr", "-l", type=float, default=5e-5, 
        help="Learning rate (default: 5e-5)"
    )
    training_group.add_argument(
        "--backbone_lr_multiplier", "-blrm", type=float, default=0.1,
        help="Multiplier for backbone learning rate relative to classifier (default: 0.1)"
    )
    training_group.add_argument(
        "--weight_decay", "-wd", type=float,
        help="Weight decay for optimizer (default: from config)"
    )
    training_group.add_argument(
        "--label_smoothing", "-ls", type=float,
        help="Label smoothing factor (default: from config)"
    )
    training_group.add_argument(
        "--grad_clip", "-gc", type=float,
        help="Gradient clipping max norm (default: from config)"
    )
    training_group.add_argument(
        "--output_dir", "-o",
        default="models",
        help="Directory to save trained model and results",
    )
    training_group.add_argument(
        "--config", "-c",
        help="Path to configuration JSON file with class distribution and calibration factors",
    )
    training_group.add_argument(
        "--resume", "-r",
        help="Resume training from checkpoint file"
    )
    training_group.add_argument(
        "--binary", "-bin", action="store_true",
        help="Train as binary classification (multiple receipts or not)"
    )
    training_group.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration without actual training"
    )
    
    # Class distribution
    training_group.add_argument(
        "--class_dist", 
        help="Comma-separated class distribution (e.g., '0.3,0.2,0.2,0.1,0.1,0.1')"
    )
    
    # Reproducibility options
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument(
        "--seed", "-s", type=int,
        help="Random seed for reproducibility"
    )
    repro_group.add_argument(
        "--deterministic", "-d", action="store_true",
        help="Enable deterministic mode for reproducibility (may reduce performance)"
    )

    args = parser.parse_args()

    # Get configuration singleton
    config = get_config()
    
    # Load configuration if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Override class distribution if provided (and not in binary mode)
    if args.class_dist and not args.binary:
        try:
            dist = [float(x) for x in args.class_dist.split(',')]
            if len(dist) != 5 and not args.binary:
                raise ValueError("Class distribution must have exactly 5 values for multiclass mode")
            config.update_class_distribution(dist)
            print(f"Using custom class distribution: {dist}")
        except Exception as e:
            print(f"Error parsing class distribution: {e}")
            print("Using default class distribution")
    
    # If binary mode is specified, it overrides any class_dist setting
    if args.binary:
        # Binary mode configuration will be handled in train_model
        print("Binary mode specified - will train for 'multiple receipts or not' classification")
    
    # Override hyperparameters if specified
    if args.weight_decay is not None:
        config.update_model_param("weight_decay", args.weight_decay)
        print(f"Using custom weight decay: {args.weight_decay}")
        
    if args.label_smoothing is not None:
        config.update_model_param("label_smoothing", args.label_smoothing)
        print(f"Using custom label smoothing: {args.label_smoothing}")
        
    # Set backbone learning rate multiplier
    config.update_model_param("backbone_lr_multiplier", args.backbone_lr_multiplier)
    print(f"Using backbone learning rate multiplier: {args.backbone_lr_multiplier}")
    
    # Set gradient clipping if specified
    if args.grad_clip is not None:
        config.update_model_param("gradient_clip_value", args.grad_clip)
        print(f"Using gradient clipping max norm: {args.grad_clip}")
        
    # Set reproducibility parameters if provided
    if args.seed is not None:
        config.update_model_param("random_seed", args.seed)
        print(f"Using user-specified random seed: {args.seed}")
        
    if args.deterministic is not None:
        config.update_model_param("deterministic_mode", args.deterministic)
        mode_str = "enabled" if args.deterministic else "disabled"
        print(f"Deterministic mode {mode_str} by user")

    # Validate that files exist
    train_csv_path = Path(args.train_csv)
    train_dir_path = Path(args.train_dir)
    
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV file not found: {train_csv_path}")
    if not train_dir_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir_path}")

    # Optional validation files
    val_csv_path = Path(args.val_csv) if args.val_csv else None
    val_dir_path = Path(args.val_dir) if args.val_dir else None
    
    # Check if validation files exist when provided
    val_csv = args.val_csv if (val_csv_path and val_csv_path.exists()) else None
    val_dir = args.val_dir if (val_dir_path and val_dir_path.exists()) else None
    
    # Check if we're resuming training
    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        resume_checkpoint = args.resume
        print(f"Resuming training from checkpoint: {resume_path}")
    
    # If dry run, just print configuration and exit
    if args.dry_run:
        print("\n=== DRY RUN - CONFIGURATION VALIDATION ===")
        print(f"Model type: Swin-Tiny")
        print(f"Training data: {args.train_csv} ({args.train_dir})")
        print(f"Validation data: {val_csv} ({val_dir})")
        print(f"Binary mode: {args.binary}")
        print(f"Data augmentation: {'disabled' if args.no_augment else 'enabled'}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate - Classifier: {args.lr}, Backbone: {args.lr * args.backbone_lr_multiplier}")
        print(f"Output directory: {args.output_dir}")
        print(f"Reproducibility: seed={config.get_model_param('random_seed')}, deterministic={config.get_model_param('deterministic_mode')}")
        print(f"Class distribution: {config.class_distribution}")
        print(f"Weight decay: {config.get_model_param('weight_decay')}")
        print(f"Label smoothing: {config.get_model_param('label_smoothing')}")
        if resume_checkpoint:
            print(f"Resuming from: {resume_checkpoint}")
        print("=== CONFIGURATION VALID ===\n")
        return

    train_model(
        args.train_csv,
        args.train_dir,
        val_csv,
        val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        binary=args.binary,
        augment=(not args.no_augment),  # Pass augmentation flag to train_model
        resume_checkpoint=resume_checkpoint if args.resume else None,
    )


if __name__ == "__main__":
    main()