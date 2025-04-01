import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import shutil
from model_factory import ModelFactory
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
from config import get_config
from training_utils import validate, plot_confusion_matrix, plot_training_curves, ModelCheckpoint, EarlyStopping
from device_utils import get_device
from reproducibility import set_seed, get_reproducibility_info
from datasets import ReceiptDataset, create_data_loaders, prepare_hierarchical_datasets


def train_hierarchical_model(
    model_type,
    train_csv,
    train_dir,
    val_csv,
    val_dir,
    output_dir="models/hierarchical",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    level2_lr=2e-4,  # Specific LR for Level 2 model
    backbone_lr_multiplier=0.1,
    level2_backbone_lr_multiplier=0.1,  # Specific backbone LR multiplier for Level 2
    augment=True,
    multiclass=False,
):
    """
    Train a hierarchical model for receipt counting.

    Args:
        model_type: Type of model to use ("vit" or "swin")
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file
        val_dir: Directory containing validation images
        output_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        backbone_lr_multiplier: Multiplier for backbone learning rate
        augment: Whether to apply data augmentation
        multiclass: Whether to include the multiclass model for 2+ receipts
    """
    # Create output directory structure
    output_path = Path(output_dir)
    level1_dir = output_path / "level1"
    level2_dir = output_path / "level2"
    multiclass_dir = output_path / "multiclass"
    
    level1_dir.mkdir(parents=True, exist_ok=True)
    level2_dir.mkdir(parents=True, exist_ok=True)
    if multiclass:
        multiclass_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up paths
    train_csv_path = Path(train_csv)
    val_csv_path = Path(val_csv)
    train_dir_path = Path(train_dir)
    val_dir_path = Path(val_dir)
    
    # Ensure base directories exist
    train_dataset_path = train_csv_path.parent
    val_dataset_path = val_csv_path.parent
    
    # Prepare hierarchical datasets
    print("Preparing hierarchical training datasets...")
    train_datasets = prepare_hierarchical_datasets(
        train_dataset_path, 
        train_csv_path.name, 
        output_path=train_dataset_path
    )
    
    print("\nPreparing hierarchical validation datasets...")
    val_datasets = prepare_hierarchical_datasets(
        val_dataset_path, 
        val_csv_path.name, 
        output_path=val_dataset_path
    )
    
    # Train Level 1 model (0 vs 1+ receipts)
    print("\n" + "="*80)
    print("TRAINING LEVEL 1 MODEL (0 receipts vs 1+ receipts)")
    print("="*80)
    
    # Train the Level 1 model
    train_level1_model(
        model_type=model_type,
        train_csv=train_datasets['level1']['path'],
        val_csv=val_datasets['level1']['path'],
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=level1_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone_lr_multiplier=backbone_lr_multiplier,
        augment=augment,
    )
    
    # Train Level 2 model (1 vs 2+ receipts)
    print("\n" + "="*80)
    print("TRAINING LEVEL 2 MODEL (1 receipt vs 2+ receipts)")
    print("="*80)
    
    # Train the Level 2 model with specific parameters
    train_level2_model(
        model_type=model_type,
        train_csv=train_datasets['level2']['path'],
        val_csv=val_datasets['level2']['path'],
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=level2_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=level2_lr,  # Use Level 2 specific learning rate 
        backbone_lr_multiplier=level2_backbone_lr_multiplier,  # Use Level 2 specific backbone LR multiplier
        augment=augment,
    )
    
    # Optionally train multiclass model for 2+ receipts
    if multiclass:
        print("\n" + "="*80)
        print("TRAINING MULTICLASS MODEL (2-5 receipts)")
        print("="*80)
        
        # Train the multiclass model
        train_multiclass_model(
            model_type=model_type,
            train_csv=train_datasets['multiclass']['path'],
            val_csv=val_datasets['multiclass']['path'],
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=multiclass_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            backbone_lr_multiplier=backbone_lr_multiplier,
            augment=augment,
        )
    
    print("\n" + "="*80)
    print("HIERARCHICAL TRAINING COMPLETE")
    print("="*80)
    print(f"Models saved to {output_path}")
    
    # Return the paths to the trained models
    return {
        'level1': level1_dir / f"receipt_counter_{model_type}_best.pth",
        'level2': level2_dir / f"receipt_counter_{model_type}_best.pth",
        'multiclass': multiclass_dir / f"receipt_counter_{model_type}_best.pth" if multiclass else None
    }


def train_level1_model(
    model_type,
    train_csv,
    val_csv,
    train_dir,
    val_dir,
    output_dir,
    epochs=20,
    batch_size=16,
    lr=1e-4,
    backbone_lr_multiplier=0.1,
    augment=True,
):
    """Train the Level 1 model (0 vs 1+ receipts)"""
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    seed_info = set_seed()
    print(f"Using random seed: {seed_info['seed']}, deterministic mode: {seed_info['deterministic']}")
    
    # Create data loaders for Level 1
    train_loader, val_loader, num_train, num_val = create_data_loaders(
        train_csv=train_csv,
        train_dir=train_dir,
        val_csv=val_csv,
        val_dir=val_dir,
        batch_size=batch_size,
        augment_train=augment,
        hierarchical_level="level1"
    )
    
    print(f"Created data loaders for Level 1 - Train: {num_train} samples, Val: {num_val} samples")
    
    # Create model
    model = ModelFactory.create_transformer(
        model_type=model_type,
        pretrained=True,
        num_classes=2  # Binary classification
    ).to(device)
    
    print(f"Created {model_type.upper()} model for binary classification (0 vs 1+ receipts)")
    
    # Loss function and optimizer
    # For Level 1, we may want to adjust class weights if there's imbalance
    config = get_config()
    label_smoothing = config.get_model_param("label_smoothing", 0.1)
    weight_decay = config.get_model_param("weight_decay", 0.01)
    
    # Use weighted loss if classes are imbalanced
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Use different learning rates for backbone and classification head
    backbone_lr = lr * backbone_lr_multiplier
    parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': lr}
    ]
    
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=False,  # Set to False to avoid deprecation warnings
        min_lr=1e-6
    )
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "val_f1_macro": []}
    
    # For early stopping
    patience = config.get_model_param("early_stopping_patience", 5)
    
    # Create checkpoint manager
    checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        metrics=["balanced_accuracy", "f1_macro"],
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
    # Create early stopping manager
    early_stopping = EarlyStopping(
        patience=patience,
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
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
            
            # Apply gradient clipping
            gradient_clip_value = config.get_model_param("gradient_clip_value", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        
        # Validate
        metrics = validate(model, val_loader, criterion, device)
        
        # Extract metrics and update history
        val_loss = metrics['loss']
        val_acc = metrics['accuracy']
        val_balanced_acc = metrics['balanced_accuracy']
        val_f1_macro = metrics['f1_macro']
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_f1_macro"].append(val_f1_macro)
        
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Balanced Accuracy: {val_balanced_acc:.2%}, "
            f"F1 Macro: {val_f1_macro:.2%}"
        )
        
        # Update learning rate scheduler
        scheduler.step(val_f1_macro)
        
        # Check for improvement and save model if needed
        improved = checkpoint.check_improvement(
            metrics_dict=metrics,
            model=model,
            model_type=model_type
        )
        
        # Check for early stopping
        should_stop = early_stopping.check_improvement(val_f1_macro)
        if should_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_model_path = Path(output_dir) / f"receipt_counter_{model_type}_final.pth"
    ModelFactory.save_model(model, final_model_path)
    
    # Save training history
    pd.DataFrame(history).to_csv(
        Path(output_dir) / f"{model_type}_level1_history.csv", index=False
    )
    
    # Plot training curves
    plot_training_curves(
        history,
        output_path=Path(output_dir) / f"{model_type}_level1_curves.png"
    )
    
    # Generate final validation plots
    final_metrics = validate(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    accuracy, balanced_accuracy = plot_confusion_matrix(
        final_metrics['predictions'],
        final_metrics['targets'],
        output_path=Path(output_dir) / f"{model_type}_level1_confusion.png",
    )
    
    print(f"\nLevel 1 training complete! Final model saved to {output_dir}")
    return model


# Define a specialized Level 2 dataset
class Level2Dataset(torch.utils.data.Dataset):
    """Dataset specifically for Level 2 (1 vs 2+ receipts) classification with balanced sampling"""
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        """
        Initialize a Level 2 dataset with proper target mapping.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            augment: Whether to apply data augmentation (used for training)
        """
        from datasets import ReceiptDataset
        
        # Initialize the base dataset with raw receipt counts
        self.base_dataset = ReceiptDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=transform,
            augment=augment,
            binary=False,
            hierarchical_level=None  # Use raw counts, we'll do the mapping ourselves
        )
        
        # Filter to only include samples with at least 1 receipt
        original_count = len(self.base_dataset.data)
        self.base_dataset.data = self.base_dataset.data[self.base_dataset.data['receipt_count'] > 0].reset_index(drop=True)
        filtered_count = len(self.base_dataset.data)
        print(f"Level2Dataset: Filtered {original_count} samples to {filtered_count} samples with 1+ receipts")
        
        # Get class distribution
        class_counts = self.base_dataset.data['receipt_count'].value_counts()
        single_receipts = class_counts.get(1, 0)
        multiple_receipts = class_counts[class_counts.index > 1].sum()
        print(f"Level2Dataset class distribution: 1 receipt: {single_receipts}, 2+ receipts: {multiple_receipts}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        
        # Ensure target is a tensor and convert to int for safety
        if isinstance(target, torch.Tensor):
            target_value = target.item()
        else:
            target_value = int(target)
            
        # Map target: 1 -> 0, 2+ -> 1
        binary_target = 0 if target_value == 1 else 1
        return img, torch.tensor(binary_target, dtype=torch.long)


def train_level2_model(
    model_type,
    train_csv,
    val_csv,
    train_dir,
    val_dir,
    output_dir,
    epochs=20,
    batch_size=16,
    lr=2e-4,  # Increased default learning rate
    backbone_lr_multiplier=0.1,
    augment=True,
):
    """
    Train the Level 2 model (1 vs 2+ receipts) with balanced sampling
    and other improvements to enhance accuracy.
    
    Args:
        model_type: Type of model to use ("vit" or "swin")
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        output_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for the classifier head
        backbone_lr_multiplier: Multiplier for backbone learning rate
        augment: Whether to apply data augmentation
    """
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    seed_info = set_seed()
    print(f"Using random seed: {seed_info['seed']}, deterministic mode: {seed_info['deterministic']}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets using our specialized Level2Dataset
    print("Loading training data...")
    train_dataset = Level2Dataset(
        csv_file=train_csv,
        img_dir=train_dir,
        augment=augment
    )
    
    # Count class distribution by examining actual dataset targets
    targets = []
    for i in range(len(train_dataset)):
        _, target = train_dataset[i]
        targets.append(target.item())
    
    # Create array of targets
    targets = np.array(targets)
    class_counts = {
        0: np.sum(targets == 0),  # Count of class 0 (single receipt)
        1: np.sum(targets == 1)   # Count of class 1 (multiple receipts)
    }
    print(f"Training class distribution after mapping: {class_counts}")
    
    # Calculate class weights for balanced sampling
    class_weights = 1.0 / torch.tensor(
        [class_counts[0], class_counts[1]], dtype=torch.float
    )
    
    # Assign weight to each sample based on its target
    sample_weights = [class_weights[t].item() for t in targets]
    
    # Create sampler for balanced training
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Load validation data
    print("Loading validation data...")
    val_dataset = Level2Dataset(
        csv_file=val_csv,
        img_dir=val_dir,
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler for balanced training
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created data loaders for Level 2 with balanced sampling - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    model = ModelFactory.create_transformer(
        model_type=model_type,
        pretrained=True,
        num_classes=2  # Binary classification
    ).to(device)
    
    print(f"Created {model_type.upper()} model for binary classification (1 vs 2+ receipts)")
    
    # Loss function and optimizer
    config = get_config()
    label_smoothing = config.get_model_param("label_smoothing", 0.1)
    weight_decay = config.get_model_param("weight_decay", 0.01)
    
    # Use class weighting to fix Level 2 model bias toward class 1 (multiple receipts)
    # deliberately favor class 0 (single receipt) to counteract the bias
    class_weights = torch.tensor([1.5, 0.5]).to(device)
    print(f"Using class weights for Level 2 model: {class_weights}")
    
    # Create weighted loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    
    # Set up parameter groups with different learning rates for backbone and classifier
    # Identify backbone and classifier parameters
    backbone_params = []
    classifier_params = []
    
    # Different parameter groups based on model type
    if model_type == "swin":
        # SwinV2 Transformer parameters - check for different attribute names
        if hasattr(model, 'swin'):
            backbone_params.extend(model.swin.parameters())
        elif hasattr(model, 'swinv2'):
            backbone_params.extend(model.swinv2.parameters())
        elif hasattr(model, 'model'):
            backbone_params.extend(model.model.parameters())
        
        # Classifier parameters 
        if hasattr(model, 'classifier'):
            classifier_params.extend(model.classifier.parameters())
    else:
        # ViT parameters
        if hasattr(model, 'vit'):
            backbone_params.extend(model.vit.parameters()) 
        elif hasattr(model, 'model'):
            backbone_params.extend(model.model.parameters())
            
        # Classifier parameters
        if hasattr(model, 'classifier'):
            classifier_params.extend(model.classifier.parameters())
    
    # For debugging
    print(f"Backbone parameters: {sum(p.numel() for p in backbone_params)}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier_params)}")
    
    # Set up optimizer with parameter groups
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * backbone_lr_multiplier},
        {'params': classifier_params, 'lr': lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=False,  # Set to False to avoid deprecation warnings
        min_lr=1e-6
    )
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "val_f1_macro": []}
    
    # For early stopping
    patience = config.get_model_param("early_stopping_patience", 5)
    
    # Create checkpoint manager
    checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        metrics=["balanced_accuracy", "f1_macro"],
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
    # Create early stopping manager
    early_stopping = EarlyStopping(
        patience=patience,
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
    # Get gradient clipping value from config
    gradient_clip_value = config.get_model_param("gradient_clip_value", 1.0)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
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
                _, predicted = torch.max(logits, 1)
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            accuracy = 100.0 * correct / total if total > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'acc': f'{accuracy:.2f}%'
            })
        
        # Calculate train metrics
        train_accuracy = correct / total if total > 0 else 0
        train_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        train_f1_macro = f1_score(all_targets, all_preds, average='macro')
        train_loss = running_loss / len(train_loader)
        
        # Update history
        history["train_loss"].append(train_loss)
        
        # Validate
        metrics = validate(model, val_loader, criterion, device)
        
        # Extract metrics and update history
        val_loss = metrics['loss']
        val_acc = metrics['accuracy']
        val_balanced_acc = metrics['balanced_accuracy']
        val_f1_macro = metrics['f1_macro']
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_f1_macro"].append(val_f1_macro)
        
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1_macro:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, "
            f"Val F1: {val_f1_macro:.4f}"
        )
        
        # Update learning rate scheduler
        scheduler.step(val_f1_macro)
        
        # Check for improvement and save model if needed
        improved = checkpoint.check_improvement(
            metrics_dict=metrics,
            model=model,
            model_type=model_type
        )
        
        # Check for early stopping
        should_stop = early_stopping.check_improvement(val_f1_macro)
        if should_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_model_path = Path(output_dir) / f"receipt_counter_{model_type}_final.pth"
    ModelFactory.save_model(model, final_model_path)
    
    # Save training history
    pd.DataFrame(history).to_csv(
        Path(output_dir) / f"{model_type}_level2_history.csv", index=False
    )
    
    # Plot training curves
    plot_training_curves(
        history,
        output_path=Path(output_dir) / f"{model_type}_level2_curves.png"
    )
    
    # Generate final validation plots
    final_metrics = validate(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    accuracy, balanced_accuracy = plot_confusion_matrix(
        final_metrics['predictions'],
        final_metrics['targets'],
        output_path=Path(output_dir) / f"{model_type}_level2_confusion.png",
    )
    
    # Save a compatibility version of the best model (state dict only)
    try:
        best_model_path = Path(output_dir) / f"receipt_counter_{model_type}_best.pth"
        final_model_path = Path(output_dir) / f"receipt_counter_{model_type}_best_state_dict.pth"
        
        # Check if this is a saved ModelCheckpoint model (dictionary)
        try:
            with open(best_model_path, 'rb') as f:
                checkpoint = torch.load(best_model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    torch.save(checkpoint['model_state_dict'], final_model_path)
                    print(f"Created state dict only version at: {final_model_path}")
        except:
            print("Best model is already a state dict.")
    except Exception as e:
        print(f"Warning: Could not create state dict only version: {e}")
    
    print(f"\nLevel 2 training complete! Final model saved to {output_dir}")
    return model


def train_multiclass_model(
    model_type,
    train_csv,
    val_csv,
    train_dir,
    val_dir,
    output_dir,
    epochs=20,
    batch_size=16,
    lr=1e-4,
    backbone_lr_multiplier=0.1,
    augment=True,
):
    """Train the multiclass model for 2+ receipts (optional)"""
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    seed_info = set_seed()
    print(f"Using random seed: {seed_info['seed']}, deterministic mode: {seed_info['deterministic']}")
    
    # Create data loaders for multiclass
    train_loader, val_loader, num_train, num_val = create_data_loaders(
        train_csv=train_csv,
        train_dir=train_dir,
        val_csv=val_csv,
        val_dir=val_dir,
        batch_size=batch_size,
        augment_train=augment,
        hierarchical_level="multiclass"
    )
    
    print(f"Created data loaders for Multiclass - Train: {num_train} samples, Val: {num_val} samples")
    
    # Create model
    model = ModelFactory.create_transformer(
        model_type=model_type,
        pretrained=True,
        num_classes=4  # 2, 3, 4, 5 receipts
    ).to(device)
    
    print(f"Created {model_type.upper()} model for multiclass classification (2, 3, 4, 5 receipts)")
    
    # Loss function and optimizer
    config = get_config()
    label_smoothing = config.get_model_param("label_smoothing", 0.1)
    weight_decay = config.get_model_param("weight_decay", 0.01)
    
    # Use weighted loss if classes are imbalanced
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Use different learning rates for backbone and classification head
    backbone_lr = lr * backbone_lr_multiplier
    parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': lr}
    ]
    
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=False,  # Set to False to avoid deprecation warnings
        min_lr=1e-6
    )
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "val_f1_macro": []}
    
    # For early stopping
    patience = config.get_model_param("early_stopping_patience", 5)
    
    # Create checkpoint manager
    checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        metrics=["balanced_accuracy", "f1_macro"],
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
    # Create early stopping manager
    early_stopping = EarlyStopping(
        patience=patience,
        mode="max",
        verbose=False  # Set to False to avoid deprecation warnings
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            # For multiclass, we need to offset the targets to be 0-3
            targets = targets - 2
            
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
            
            # Apply gradient clipping
            gradient_clip_value = config.get_model_param("gradient_clip_value", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        
        # Custom validation for multiclass model
        val_metrics = validate_multiclass(model, val_loader, criterion, device)
        
        # Extract metrics and update history
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_balanced_acc = val_metrics['balanced_accuracy']
        val_f1_macro = val_metrics['f1_macro']
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_f1_macro"].append(val_f1_macro)
        
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Balanced Accuracy: {val_balanced_acc:.2%}, "
            f"F1 Macro: {val_f1_macro:.2%}"
        )
        
        # Update learning rate scheduler
        scheduler.step(val_f1_macro)
        
        # Check for improvement and save model if needed
        improved = checkpoint.check_improvement(
            metrics_dict=val_metrics,
            model=model,
            model_type=model_type
        )
        
        # Check for early stopping
        should_stop = early_stopping.check_improvement(val_f1_macro)
        if should_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_model_path = Path(output_dir) / f"receipt_counter_{model_type}_final.pth"
    ModelFactory.save_model(model, final_model_path)
    
    # Save training history
    pd.DataFrame(history).to_csv(
        Path(output_dir) / f"{model_type}_multiclass_history.csv", index=False
    )
    
    # Plot training curves
    plot_training_curves(
        history,
        output_path=Path(output_dir) / f"{model_type}_multiclass_curves.png"
    )
    
    # Generate final validation plots
    final_metrics = validate_multiclass(model, val_loader, criterion, device)
    
    # Plot confusion matrix (adjusted for multiclass)
    accuracy, balanced_accuracy = plot_confusion_matrix(
        final_metrics['predictions'],
        final_metrics['targets'],
        output_path=Path(output_dir) / f"{model_type}_multiclass_confusion.png",
    )
    
    print(f"\nMulticlass training complete! Final model saved to {output_dir}")
    return model


def validate_multiclass(model, dataloader, criterion, device):
    """Custom validation function for multiclass model that handles offset targets"""
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            # For multiclass, we need to offset the targets to be 0-3
            targets_offset = targets - 2
            
            # Forward pass
            outputs = model(images)
            
            # HuggingFace models return an object with logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                loss = criterion(logits, targets_offset)
                preds = torch.argmax(logits, dim=1)
            else:
                loss = criterion(outputs, targets_offset)
                preds = torch.argmax(outputs, dim=1)
            
            # Convert predictions back to original scale (2-5)
            preds = preds + 2
            
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return {
        'loss': val_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a hierarchical model for receipt counting"
    )
    
    # Data input options
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        "--train_csv", "-tc",
        default="receipt_dataset/train.csv",
        help="Path to training CSV file",
    )
    data_group.add_argument(
        "--train_dir", "-td",
        default="receipt_dataset/train",
        help="Directory containing training images",
    )
    data_group.add_argument(
        "--val_csv", "-vc",
        default="receipt_dataset/val.csv",
        help="Path to validation CSV file",
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
    
    # Model options
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        "--model_type", "-m",
        choices=["vit", "swin"],
        default="vit",
        help="Type of transformer model to use (vit or swin)"
    )
    model_group.add_argument(
        "--multiclass", action="store_true",
        help="Include multiclass model for 2+ receipts (optional)"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        "--epochs", "-e", type=int, default=20, 
        help="Number of training epochs (default: 20)"
    )
    training_group.add_argument(
        "--batch_size", "-b", type=int, default=16, 
        help="Batch size for training (default: 16)"
    )
    training_group.add_argument(
        "--lr", "-l", type=float, default=1e-4, 
        help="Learning rate (default: 1e-4)"
    )
    training_group.add_argument(
        "--level2_lr", "-l2lr", type=float, default=2e-4, 
        help="Specific learning rate for Level 2 model (default: 2e-4)"
    )
    training_group.add_argument(
        "--backbone_lr_multiplier", "-blrm", type=float, default=0.1,
        help="Multiplier for backbone learning rate (default: 0.1)"
    )
    training_group.add_argument(
        "--level2_backbone_lr_multiplier", "-l2blrm", type=float, default=0.1,
        help="Specific backbone LR multiplier for Level 2 model (default: 0.1)"
    )
    training_group.add_argument(
        "--output_dir", "-o",
        default="models/hierarchical",
        help="Directory to save trained models",
    )
    training_group.add_argument(
        "--workers", "-w", type=int,
        help="Number of dataloader workers (default: from config, typically 4). Set to 0 or 1 to avoid shared memory issues."
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
    
    # Set reproducibility parameters if provided
    if args.seed is not None:
        config.update_model_param("random_seed", args.seed)
        print(f"Using user-specified random seed: {args.seed}")
        
    if args.deterministic is not None:
        config.update_model_param("deterministic_mode", args.deterministic)
        mode_str = "enabled" if args.deterministic else "disabled"
        print(f"Deterministic mode {mode_str} by user")
        
    # Set number of workers if specified
    if args.workers is not None:
        config.update_model_param("num_workers", args.workers)
        print(f"Using {args.workers} dataloader workers (custom value from command line)")
        if args.workers <= 1:
            print("Reduced worker count should help avoid shared memory (shm) issues")
    
    # Train hierarchical model
    train_hierarchical_model(
        model_type=args.model_type,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        val_csv=args.val_csv,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        level2_lr=args.level2_lr,
        level2_backbone_lr_multiplier=args.level2_backbone_lr_multiplier,
        augment=(not args.no_augment),
        multiclass=args.multiclass,
    )


if __name__ == "__main__":
    main()