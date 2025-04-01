#!/usr/bin/env python3
"""
Bayesian-aware hierarchical model training for receipt counting.

This script trains hierarchical models with explicit class weighting based on data frequencies,
and records the exact weights used during training for Bayesian correction during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse
import os
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from model_factory import ModelFactory
from device_utils import get_device
from datasets import ReceiptDataset
from training_utils import validate
from reproducibility import set_seed
from config import get_config


# Define Level2Dataset as a class at module level for proper pickling
class Level2Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        # Initialize the base dataset for transforms
        from datasets import ReceiptDataset

        self.base_dataset = ReceiptDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=transform,
            augment=augment,
            binary=False,
            hierarchical_level=None,
        )

        # Filter to only include samples with at least 1 receipt
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["receipt_count"] > 0].reset_index(drop=True)
        print(
            f"Level2Dataset contains {len(self.data)} samples after filtering for 1+ receipts"
        )

        # Store paths for loading images
        self.img_dir = img_dir
        self.root_dir = os.path.dirname(img_dir)

        # Get configuration from the base dataset
        self.transform = self.base_dataset.transform
        self.image_size = self.base_dataset.image_size
        self.mean = self.base_dataset.mean
        self.std = self.base_dataset.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the filename and target
        filename = self.data.iloc[idx]["filename"]
        count = self.data.iloc[idx]["receipt_count"]

        # Map target: 1 -> 0, 2+ -> 1
        binary_target = 0 if count == 1 else 1

        # Try to load image from same paths as ReceiptDataset
        potential_paths = [
            os.path.join(self.img_dir, filename),  # Primary location
            os.path.join(self.root_dir, "train", filename),  # Alternative in train dir
            os.path.join(self.root_dir, "val", filename),  # Alternative in val dir
            os.path.join(self.root_dir, filename),  # Alternative in root dir
        ]

        # Try each potential path
        image = None
        for path in potential_paths:
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                break

        if image is None:
            # Fallback to using a blank image rather than crashing
            print(
                f"Warning: Could not find image {filename} in any potential location."
            )
            image = Image.new(
                "RGB", (self.image_size, self.image_size), color=(0, 0, 0)
            )

        # Apply transformations if available - using PIL-based transforms now
        if self.transform:
            # The torchvision transforms take the PIL image directly
            image_tensor = self.transform(image)
        else:
            # Manual resize and normalization if no transform provided
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = (image_np - self.mean) / self.std
            image_tensor = torch.tensor(image_np).permute(2, 0, 1)

        return image_tensor, torch.tensor(binary_target, dtype=torch.long)


class FocalLoss(nn.Module):
    """
    Weighted focal loss implementation.
    
    Focal Loss was proposed in "Focal Loss for Dense Object Detection" 
    (https://arxiv.org/abs/1708.02002) to address class imbalance by
    down-weighting easy examples.
    
    This implementation supports class weighting similar to CrossEntropyLoss.
    """
    
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            weight: Class weights (same as in CrossEntropyLoss)
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        """
        Calculate the focal loss.
        
        Args:
            input: Raw logits from the model (B, C)
            target: Ground truth class indices (B,)
            
        Returns:
            loss: Computed focal loss
        """
        # Get softmax probabilities
        log_softmax = F.log_softmax(input, dim=1)
        logpt = log_softmax.gather(1, target.unsqueeze(1))
        logpt = logpt.squeeze(1)
        
        # Get class probabilities
        pt = logpt.exp()
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.weight is not None:
            # Gather the weights for the target classes
            weight = self.weight.gather(0, target)
            focal_weight = focal_weight * weight
            
        # Compute the focal loss
        loss = -focal_weight * logpt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BayesianHierarchicalTrainer:
    """
    Trainer for hierarchical models with explicit weight tracking for Bayesian inference.
    """

    def __init__(
        self,
        train_csv,
        train_dir,
        val_csv,
        val_dir,
        output_dir,
        model_type="swin",
        epochs=20,
        batch_size=16,
        learning_rate=2e-4,
        backbone_lr_multiplier=0.1,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=42,
        deterministic=True,
        level1_weights=None,  # Custom weights for Level 1 (0 vs 1+)
        level2_weights=None,  # Custom weights for Level 2 (1 vs 2+)
        look_at_misclassifications=False,  # Whether to analyze misclassifications
        use_focal_loss=False,  # Whether to use focal loss instead of cross entropy
        focal_gamma=2.0,  # Gamma parameter for focal loss
        lower_class1_threshold=None,  # Use lower threshold for class 1 in level 1 model
        lower_level2_threshold=None,  # Use lower threshold for multiple receipts in level 2 model
    ):
        """
        Initialize the trainer.

        Args:
            train_csv: Path to training CSV file
            train_dir: Directory containing training images
            val_csv: Path to validation CSV file
            val_dir: Directory containing validation images
            output_dir: Directory to save trained models and metadata
            model_type: Type of model ("vit" or "swin")
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for classifier head
            backbone_lr_multiplier: Multiplier for backbone learning rate
            weight_decay: Weight decay for optimizer
            grad_clip: Gradient clipping max norm
            seed: Random seed for reproducibility
            deterministic: Whether to enable deterministic mode
            level1_weights: Optional custom weights for Level 1 training
            level2_weights: Optional custom weights for Level 2 training
        """
        # Set random seed for reproducibility
        set_seed(seed, deterministic)
        self.seed = seed
        self.deterministic = deterministic

        # Store parameters
        self.train_csv = train_csv
        self.train_dir = train_dir
        self.val_csv = val_csv
        self.val_dir = val_dir
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Custom class weights
        self.level1_weights = level1_weights
        self.level2_weights = level2_weights
        
        # Analysis and loss options
        self.look_at_misclassifications = look_at_misclassifications
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.lower_class1_threshold = lower_class1_threshold
        self.lower_level2_threshold = lower_level2_threshold
        
        # Standard threshold for comparison
        self.standard_threshold = 0.5
        
        # Validation for lowered Level 1 threshold
        if self.lower_class1_threshold is not None:
            if not (0 < self.lower_class1_threshold < 1):
                raise ValueError(f"Lower class 1 threshold must be between 0 and 1, got {self.lower_class1_threshold}")
            
            if self.lower_class1_threshold >= 0.5:
                print(f"Warning: Lower class 1 threshold {self.lower_class1_threshold} is not lower than standard 0.5")
            
            print(f"Using lowered threshold {self.lower_class1_threshold} for class 1 detection in Level 1 during validation")
            print("This reduces false negatives for receipt detection but increases false positives")
            print("Training metadata will store this threshold for Bayesian correction during inference")
            
        # Validation for lowered Level 2 threshold
        if self.lower_level2_threshold is not None:
            if not (0 < self.lower_level2_threshold < 1):
                raise ValueError(f"Lower level 2 threshold must be between 0 and 1, got {self.lower_level2_threshold}")
            
            if self.lower_level2_threshold >= 0.5:
                print(f"Warning: Lower level 2 threshold {self.lower_level2_threshold} is not lower than standard 0.5")
            
            print(f"Using lowered threshold {self.lower_level2_threshold} for multiple receipts detection in Level 2")
            print("This reduces false negatives for multiple receipt detection but increases false positives")
            print("Training metadata will store this threshold for Bayesian correction during inference")

        # Training metadata - will be populated during training
        self.metadata = {
            "training_params": {
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "backbone_lr_multiplier": backbone_lr_multiplier,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "seed": seed,
                "deterministic": deterministic,
                "use_focal_loss": use_focal_loss,
                "focal_gamma": focal_gamma if use_focal_loss else None,
                "lower_class1_threshold": lower_class1_threshold,
                "lower_level2_threshold": lower_level2_threshold,
            },
            "level1": {
                "class_weights": None,  # Will be populated during training
                "class_frequencies": None,  # Will be populated during training
            },
            "level2": {
                "class_weights": None,  # Will be populated during training
                "class_frequencies": None,  # Will be populated during training
            },
        }

        # Get device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create level-specific subdirectories
        self.level1_dir = self.output_dir / "level1"
        self.level2_dir = self.output_dir / "level2"
        self.level1_dir.mkdir(exist_ok=True)
        self.level2_dir.mkdir(exist_ok=True)

    def prepare_level1_data(self):
        """
        Prepare Level 1 datasets (0 vs 1+ receipts).

        Returns:
            train_loader, val_loader, class_weights, class_frequencies
        """
        print("Loading Level 1 training data (0 vs 1+ receipts)...")

        # Load raw training data
        train_df = pd.read_csv(self.train_csv)

        # Calculate original class frequencies
        train_df["level1_target"] = (train_df["receipt_count"] > 0).astype(int)
        class_counts = train_df["level1_target"].value_counts().to_dict()

        # Ensure both classes are represented
        for c in [0, 1]:
            if c not in class_counts:
                class_counts[c] = 0

        total_samples = len(train_df)
        class_frequencies = {
            0: class_counts[0] / total_samples,  # Frequency of class 0 (no receipts)
            1: class_counts[1] / total_samples,  # Frequency of class 1 (has receipts)
        }

        # Calculate inverse frequency class weights
        # Use custom weights if provided, otherwise calculate from data
        if self.level1_weights is not None:
            if len(self.level1_weights) == 2:
                class_weights = np.array(self.level1_weights)
                print(f"Using custom Level 1 weights: {class_weights}")
            else:
                raise ValueError("level1_weights must have exactly 2 values")
        else:
            # Inverse frequency weighting
            epsilon = 1e-8  # To avoid division by zero
            class_weights = np.array(
                [
                    1.0 / (class_frequencies[0] + epsilon),
                    1.0 / (class_frequencies[1] + epsilon),
                ]
            )

            # Normalize weights
            class_weights = class_weights / class_weights.sum() * 2

            print(f"Calculated Level 1 weights from frequencies: {class_weights}")

        # Store the frequencies and weights in metadata
        self.metadata["level1"]["class_frequencies"] = [
            float(class_frequencies[0]),
            float(class_frequencies[1]),
        ]
        self.metadata["level1"]["class_weights"] = [
            float(class_weights[0]),
            float(class_weights[1]),
        ]

        # Also store sampling weights (which are the same as class weights in this implementation)
        self.metadata["level1"]["sampling_weights"] = [
            float(class_weights[0]),
            float(class_weights[1]),
        ]

        # Create Level 1 dataset (0 vs 1+)
        train_dataset = ReceiptDataset(
            csv_file=self.train_csv,
            img_dir=self.train_dir,
            augment=True,
            hierarchical_level="level1",  # This maps to binary 0 vs 1+ classification
        )

        # Create sampler for balanced training
        targets = train_df["level1_target"].values
        sample_weights = [class_weights[t] for t in targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )

        # Create validation dataset
        val_dataset = ReceiptDataset(
            csv_file=self.val_csv,
            img_dir=self.val_dir,
            augment=False,
            hierarchical_level="level1",
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )

        print(f"Level 1 training class distribution: {class_counts}")
        print(f"Level 1 class frequencies: {class_frequencies}")
        print(f"Level 1 class weights: {class_weights}")

        return train_loader, val_loader, class_weights, class_frequencies

    def prepare_level2_data(self):
        """
        Prepare Level 2 datasets (1 vs 2+ receipts).
        Only uses samples with at least 1 receipt.

        Returns:
            train_loader, val_loader, class_weights, class_frequencies
        """
        print("Loading Level 2 training data (1 vs 2+ receipts)...")

        # Load raw training data
        train_df = pd.read_csv(self.train_csv)

        # Filter to only include samples with at least 1 receipt
        train_df = train_df[train_df["receipt_count"] > 0].reset_index(drop=True)

        # Map to binary classification: 1 receipt (0) vs 2+ receipts (1)
        train_df["level2_target"] = (train_df["receipt_count"] > 1).astype(int)

        # Calculate original class frequencies
        class_counts = train_df["level2_target"].value_counts().to_dict()

        # Ensure both classes are represented
        for c in [0, 1]:
            if c not in class_counts:
                class_counts[c] = 0

        total_samples = len(train_df)
        class_frequencies = {
            0: class_counts[0] / total_samples,  # Frequency of class 0 (1 receipt)
            1: class_counts[1] / total_samples,  # Frequency of class 1 (2+ receipts)
        }

        # Calculate inverse frequency class weights
        # Use custom weights if provided, otherwise calculate from data
        if self.level2_weights is not None:
            if len(self.level2_weights) == 2:
                class_weights = np.array(self.level2_weights)
                print(f"Using custom Level 2 weights: {class_weights}")
            else:
                raise ValueError("level2_weights must have exactly 2 values")
        else:
            # Inverse frequency weighting
            epsilon = 1e-8  # To avoid division by zero
            class_weights = np.array(
                [
                    1.0 / (class_frequencies[0] + epsilon),
                    1.0 / (class_frequencies[1] + epsilon),
                ]
            )

            # Normalize weights
            class_weights = class_weights / class_weights.sum() * 2

            print(f"Calculated Level 2 weights from frequencies: {class_weights}")

        # Store the frequencies and weights in metadata
        self.metadata["level2"]["class_frequencies"] = [
            float(class_frequencies[0]),
            float(class_frequencies[1]),
        ]
        self.metadata["level2"]["class_weights"] = [
            float(class_weights[0]),
            float(class_weights[1]),
        ]

        # Also store sampling weights (which are the same as class weights in this implementation)
        self.metadata["level2"]["sampling_weights"] = [
            float(class_weights[0]),
            float(class_weights[1]),
        ]

        # Create Level 2 datasets
        train_dataset = Level2Dataset(
            csv_file=self.train_csv, img_dir=self.train_dir, augment=True
        )

        # Create sampler for balanced training
        targets = train_df["level2_target"].values
        sample_weights = [class_weights[t] for t in targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )

        # Create validation dataset
        val_dataset = Level2Dataset(
            csv_file=self.val_csv, img_dir=self.val_dir, augment=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )

        print(f"Level 2 training class distribution: {class_counts}")
        print(f"Level 2 class frequencies: {class_frequencies}")
        print(f"Level 2 class weights: {class_weights}")

        return train_loader, val_loader, class_weights, class_frequencies

    def analyze_misclassifications(self, level, last_epoch_data, val_df):
        """
        Analyze and save misclassified images from the validation set.
        
        Args:
            level: Level being analyzed ("level1" or "level2")
            last_epoch_data: Dictionary with validation data from the last epoch
            val_df: Validation dataframe with filenames and targets
        """
        if not self.look_at_misclassifications:
            return
            
        print(f"\nAnalyzing {level} misclassifications...")
        
        # Extract data from the last epoch
        filenames = last_epoch_data["filenames"]
        predictions = last_epoch_data["predictions"]
        targets = last_epoch_data["targets"]
        confidences = last_epoch_data["confidences"]
        
        # Create directories for misclassified images
        misclass_dir = self.output_dir / f"{level}_misclassifications"
        misclass_dir.mkdir(exist_ok=True)
        
        false_neg_dir = misclass_dir / "false_negatives"
        false_pos_dir = misclass_dir / "false_positives"
        false_neg_dir.mkdir(exist_ok=True)
        false_pos_dir.mkdir(exist_ok=True)
        
        # Copy misclassified images
        false_negatives = 0
        false_positives = 0
        
        for i, (filename, pred, target, conf) in enumerate(zip(filenames, predictions, targets, confidences)):
            if pred != target:
                # Try to find the image path
                img_path = None
                possible_paths = [
                    os.path.join(self.val_dir, filename),
                    os.path.join(os.path.dirname(self.val_dir), "val", filename),
                    os.path.join(os.path.dirname(self.val_dir), filename),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break
                
                if img_path is None:
                    print(f"Warning: Could not find image {filename} for misclassification analysis")
                    continue
                
                # Determine if false positive or false negative
                if pred == 0 and target == 1:  # False negative
                    dest_dir = false_neg_dir
                    false_negatives += 1
                else:  # False positive
                    dest_dir = false_pos_dir
                    false_positives += 1
                
                # Create informative filename with true label, prediction, and confidence
                ext = os.path.splitext(filename)[1]
                new_filename = f"true_{target}_pred_{pred}_conf_{conf:.4f}_{os.path.basename(filename)}"
                shutil.copy(img_path, dest_dir / new_filename)
        
        print(f"Found {false_negatives} false negatives and {false_positives} false positives in {level}")
        print(f"Misclassified images saved to {misclass_dir}")
        
    def train_level(self, level, train_loader, val_loader, class_weights):
        """
        Train a model for a specific level of the hierarchy.

        Args:
            level: Level to train ("level1" or "level2")
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Class weights for this level

        Returns:
            trained_model_path: Path to the trained model
        """
        print(f"\nTraining {level} model ({self.model_type})...")

        # Create model
        model = ModelFactory.create_transformer(
            model_type=self.model_type,
            num_classes=2,  # Binary classification
            pretrained=True,
        )
        model = model.to(self.device)

        # Define loss with class weights
        weight_tensor = torch.tensor(
            class_weights, device=self.device, dtype=torch.float32
        )
        
        # Use focal loss or cross entropy based on setting
        if self.use_focal_loss:
            criterion = FocalLoss(weight=weight_tensor, gamma=self.focal_gamma)
            print(f"Using Focal Loss with gamma={self.focal_gamma} and class weights={class_weights}")
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"Using Cross Entropy Loss with class weights={class_weights}")

        # Identify backbone and classifier parameters for different learning rates
        backbone_params = []
        classifier_params = []

        # Different parameter groups based on model type
        if hasattr(model, "backbone"):
            backbone_params.extend(model.backbone.parameters())
        elif hasattr(model, "model"):
            backbone_params.extend(model.model.parameters())

        if hasattr(model, "classifier"):
            classifier_params.extend(model.classifier.parameters())

        # Set up optimizer with parameter groups
        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.learning_rate * self.backbone_lr_multiplier,
                },
                {"params": classifier_params, "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        # Set up learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            threshold=0.0001,
            threshold_mode="rel",
        )

        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        best_val_f1 = 0.0
        best_val_balanced_acc = 0.0
        patience_counter = 0
        max_patience = 15

        # Determine output directory for this level
        if level == "level1":
            level_dir = self.level1_dir
        else:
            level_dir = self.level2_dir

        # Create training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_balanced_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_acc": [],
            "val_f1": [],
            "val_filenames": [],  # Store filenames for misclassification analysis
            "val_predictions": [],
            "val_targets": [],
            "val_confidences": [],
        }

        best_model_path = level_dir / f"receipt_counter_{self.model_type}_best.pth"
        best_f1_model_path = (
            level_dir / f"receipt_counter_{self.model_type}_best_f1_macro.pth"
        )
        best_balanced_acc_model_path = (
            level_dir / f"receipt_counter_{self.model_type}_best_balanced_accuracy.pth"
        )

        for epoch in range(self.epochs):
            # Train for one epoch
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            all_train_preds = []
            all_train_targets = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} (Train)")

            for images, targets in progress_bar:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Handle different model output formats
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    loss = criterion(logits, targets)
                    _, predicted = torch.max(logits, 1)
                else:
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)

                # Backward pass and optimize
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()

                train_loss += loss.item()
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

                # Update progress bar
                accuracy = 100.0 * train_correct / train_total
                progress_bar.set_postfix(
                    {
                        "loss": f"{train_loss / (progress_bar.n + 1):.4f}",
                        "acc": f"{accuracy:.2f}%",
                    }
                )

                # Store predictions and targets for metrics
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_targets.extend(targets.cpu().numpy())

            # Calculate train metrics
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_balanced_acc = balanced_accuracy_score(
                all_train_targets, all_train_preds
            )
            train_f1 = f1_score(all_train_targets, all_train_preds, average="macro")

            # Validate
            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_targets = []
            all_val_filenames = []
            all_val_confidences = []
            
            # Collect filenames for validation data
            val_df = None
            if level == "level1":
                val_df = pd.read_csv(self.val_csv)
                val_df["target"] = (val_df["receipt_count"] > 0).astype(int)
            else:  # level2
                val_df = pd.read_csv(self.val_csv)
                val_df = val_df[val_df["receipt_count"] > 0].reset_index(drop=True)
                val_df["target"] = (val_df["receipt_count"] > 1).astype(int)
                
            batch_idx = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Get filenames for this batch
                    batch_size = inputs.size(0)
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + batch_size, len(val_df))
                    batch_filenames = val_df.iloc[start_idx:end_idx]["filename"].tolist()
                    batch_idx += 1
                    
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Forward pass
                    outputs = model(inputs)

                    # Handle different model output formats and extract confidences
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                        loss = criterion(logits, targets)
                        probs = torch.softmax(logits, dim=1)
                        confidences, predicted = torch.max(probs, dim=1)
                    else:
                        loss = criterion(outputs, targets)
                        probs = torch.softmax(outputs, dim=1)
                        confidences, predicted = torch.max(probs, dim=1)

                    val_loss += loss.item()

                    # Store predictions, targets, filenames, and confidences for analysis
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_targets.extend(targets.cpu().numpy())
                    all_val_filenames.extend(batch_filenames)
                    all_val_confidences.extend(confidences.cpu().numpy())

            # If using a lower threshold for class 1 in level 1, apply it during validation
            if level == "level1" and self.lower_class1_threshold is not None:
                # We need the probabilities, not just the predictions
                all_val_probs = []
                with torch.no_grad():
                    for inputs, _ in val_loader:
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        
                        # Handle different model output formats
                        if hasattr(outputs, "logits"):
                            logits = outputs.logits
                        else:
                            logits = outputs
                            
                        probs = torch.softmax(logits, dim=1)
                        all_val_probs.extend(probs.cpu().numpy())
                
                # Apply the lower threshold to class 1
                all_val_preds_custom = []
                for probs in all_val_probs:
                    if probs[1] >= self.lower_class1_threshold:  # If class 1 prob >= threshold
                        all_val_preds_custom.append(1)  # Predict class 1
                    else:
                        all_val_preds_custom.append(0)  # Else predict class 0
                
                # Calculate metrics with both standard and custom thresholds
                val_accuracy_std = accuracy_score(all_val_targets, all_val_preds)
                val_balanced_acc_std = balanced_accuracy_score(all_val_targets, all_val_preds)
                val_f1_std = f1_score(all_val_targets, all_val_preds, average="macro")
                
                val_accuracy = accuracy_score(all_val_targets, all_val_preds_custom)
                val_balanced_acc = balanced_accuracy_score(all_val_targets, all_val_preds_custom)
                val_f1 = f1_score(all_val_targets, all_val_preds_custom, average="macro")
                
                # Store both predictions for reporting
                all_val_preds_original = all_val_preds
                all_val_preds = all_val_preds_custom
                
                # Calculate confusion matrices for both thresholds
                cm_std = confusion_matrix(all_val_targets, all_val_preds_original)
                cm_custom = confusion_matrix(all_val_targets, all_val_preds)
                
                # Print comparison
                print(f"\nStandard threshold (0.5) results:")
                print(f"  Accuracy: {val_accuracy_std:.4f}, Balanced Acc: {val_balanced_acc_std:.4f}, F1: {val_f1_std:.4f}")
                print("Confusion Matrix (standard threshold):")
                print(cm_std)
                
                print(f"\nLowered threshold ({self.lower_class1_threshold}) results:")
                print(f"  Accuracy: {val_accuracy:.4f}, Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}")
                print("Confusion Matrix (lowered threshold):")
                print(cm_custom)
                
                # Calculate change in false negatives and false positives
                fn_std = cm_std[1, 0]  # True class 1, predicted class 0
                fp_std = cm_std[0, 1]  # True class 0, predicted class 1
                
                fn_custom = cm_custom[1, 0]  # True class 1, predicted class 0
                fp_custom = cm_custom[0, 1]  # True class 0, predicted class 1
                
                print(f"False negatives: {fn_std} → {fn_custom} ({fn_std - fn_custom} reduction)")
                print(f"False positives: {fp_std} → {fp_custom} ({fp_custom - fp_std} increase)")
            else:
                # Standard metric calculation for level 2 or when not using lower threshold
                val_accuracy = accuracy_score(all_val_targets, all_val_preds)
                val_balanced_acc = balanced_accuracy_score(all_val_targets, all_val_preds)
                val_f1 = f1_score(all_val_targets, all_val_preds, average="macro")

            val_metrics = {
                "loss": val_loss / len(val_loader),
                "accuracy": val_accuracy,
                "balanced_accuracy": val_balanced_acc,
                "f1_macro": val_f1,
                "predictions": all_val_preds,
                "targets": all_val_targets,
                "filenames": all_val_filenames,
                "confidences": all_val_confidences,
            }

            # Update history
            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(train_accuracy)
            history["train_balanced_acc"].append(train_balanced_acc)
            history["train_f1"].append(train_f1)

            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
            history["val_f1"].append(val_metrics["f1_macro"])
            # Store data for misclassification analysis
            history["val_filenames"].append(val_metrics["filenames"])
            history["val_predictions"].append(val_metrics["predictions"])
            history["val_targets"].append(val_metrics["targets"])
            history["val_confidences"].append(val_metrics["confidences"])

            # Update learning rate based on validation performance
            prev_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            scheduler.step(val_metrics["f1_macro"])
            current_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

            # Print current learning rates for monitoring
            print(f"  Learning rates: {current_lrs}")

            # Explicitly log when learning rates change
            if current_lrs != prev_lrs:
                print(f"  Learning rate changed: {prev_lrs} -> {current_lrs}")

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.epochs}:")
            print(
                f"  Train Loss: {train_loss / len(train_loader):.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Train Balanced Acc: {train_balanced_acc:.4f}, "
                f"Train F1: {train_f1:.4f}"
            )
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            # Print confusion matrices
            train_cm = confusion_matrix(all_train_targets, all_train_preds)
            val_cm = confusion_matrix(all_val_targets, all_val_preds)

            print("\nTraining Confusion Matrix:")
            print(train_cm)
            print("\nValidation Confusion Matrix:")
            print(val_cm)

            # Save checkpoint for best F1 score
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                torch.save(model.state_dict(), best_f1_model_path)
                print(f"  New best F1 model saved! Val F1: {best_val_f1:.4f}")

            # Save checkpoint for best balanced accuracy
            if val_metrics["balanced_accuracy"] > best_val_balanced_acc:
                best_val_balanced_acc = val_metrics["balanced_accuracy"]
                torch.save(model.state_dict(), best_balanced_acc_model_path)
                print(
                    f"  New best balanced accuracy model saved! Val Balanced Acc: {best_val_balanced_acc:.4f}"
                )
                patience_counter = 0

                # Also use this as our "best" model
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Plot and save training curves
            self.plot_training_curves(
                history, level_dir / f"{level}_training_history.png"
            )

        # Save final model
        final_model_path = level_dir / f"receipt_counter_{self.model_type}_final.pth"
        torch.save(model.state_dict(), final_model_path)

        # Convert numpy data types to Python native types for JSON serialization
        json_safe_history = {}
        for key, value in history.items():
            if isinstance(value, list) and value and isinstance(value[0], (list, np.ndarray)):
                # Handle nested lists that might contain numpy values
                json_safe_history[key] = [[float(item) if isinstance(item, np.number) else item 
                                          for item in sublist] for sublist in value]
            elif isinstance(value, list):
                # Handle simple lists that might contain numpy values
                json_safe_history[key] = [float(item) if isinstance(item, np.number) else item 
                                         for item in value]
            else:
                json_safe_history[key] = value
        
        # Save history
        with open(level_dir / f"{level}_training_history.json", "w") as f:
            json.dump(json_safe_history, f, indent=4)

        # Analyze misclassifications after training is complete
        if self.look_at_misclassifications:
            # Get validation data from the last epoch
            last_epoch_data = {
                "filenames": all_val_filenames,
                "predictions": all_val_preds,
                "targets": all_val_targets,
                "confidences": all_val_confidences
            }
            
            # Load validation dataframe for this level
            val_df = None
            if level == "level1":
                val_df = pd.read_csv(self.val_csv)
                val_df["target"] = (val_df["receipt_count"] > 0).astype(int)
            else:  # level2
                val_df = pd.read_csv(self.val_csv)
                val_df = val_df[val_df["receipt_count"] > 0].reset_index(drop=True)
                val_df["target"] = (val_df["receipt_count"] > 1).astype(int)
            
            # Analyze and save misclassifications
            self.analyze_misclassifications(level, last_epoch_data, val_df)

        print(f"\n{level} training complete!")
        print(f"Best validation balanced accuracy: {best_val_balanced_acc:.4f}")
        print(f"Best validation F1 score: {best_val_f1:.4f}")

        return best_model_path

    def plot_training_curves(self, history, output_path):
        """
        Plot and save training curves.

        Args:
            history: Dictionary of training and validation metrics
            output_path: Path to save the plot
        """
        plt.figure(figsize=(15, 10))

        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()

        # Plot balanced accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history["train_balanced_acc"], label="Train Balanced Accuracy")
        plt.plot(history["val_balanced_acc"], label="Validation Balanced Accuracy")
        plt.title("Balanced Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Accuracy")
        plt.legend()
        plt.grid()

        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(history["train_f1"], label="Train F1")
        plt.plot(history["val_f1"], label="Validation F1")
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid()

        # Plot loss
        plt.subplot(2, 2, 4)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, output_path):
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = ["0", "1"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(output_path)
        plt.close()

    def train(self):
        """
        Train all models in the hierarchical structure and save metadata.

        Returns:
            Dictionary with model paths and training metadata
        """
        # Prepare and train Level 1 model (0 vs 1+)
        level1_train_loader, level1_val_loader, level1_weights, level1_freq = (
            self.prepare_level1_data()
        )
        level1_model_path = self.train_level(
            "level1", level1_train_loader, level1_val_loader, level1_weights
        )

        # Prepare and train Level 2 model (1 vs 2+)
        level2_train_loader, level2_val_loader, level2_weights, level2_freq = (
            self.prepare_level2_data()
        )
        level2_model_path = self.train_level(
            "level2", level2_train_loader, level2_val_loader, level2_weights
        )

        # Save metadata with class weights and frequencies for Bayesian inference
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"\nTraining metadata saved to {metadata_path}")

        return {
            "level1_model": str(level1_model_path),
            "level2_model": str(level2_model_path),
            "metadata": str(metadata_path),
            "training_info": self.metadata,
        }


def parse_arguments():
    """
    Parse command line arguments for Bayesian hierarchical model training.

    Returns:
        args: Parsed command line arguments
        level1_weights: Parsed weights for Level 1 (0 vs 1+ receipts)
        level2_weights: Parsed weights for Level 2 (1 vs 2+ receipts)
    """
    parser = argparse.ArgumentParser(
        description="Train Bayesian-aware hierarchical models"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--look_at_misclassifications",
        action="store_true",
        help="Save misclassified validation images for analysis",
    )
    
    # Loss function options
    loss_group = parser.add_argument_group("Loss Function")
    loss_group.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use Focal Loss instead of Cross Entropy Loss",
    )
    loss_group.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for Focal Loss (higher = more focus on hard examples)",
    )
    loss_group.add_argument(
        "--lower_class1_threshold",
        type=float,
        help="Lower threshold for 'has receipts' class during validation to reduce false negatives (e.g., 0.3)",
    )

    # Data options
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train_csv", "-tc", required=True, help="Path to training CSV file"
    )
    data_group.add_argument(
        "--train_dir", "-td", required=True, help="Directory containing training images"
    )
    data_group.add_argument(
        "--val_csv", "-vc", required=True, help="Path to validation CSV file"
    )
    data_group.add_argument(
        "--val_dir", "-vd", required=True, help="Directory containing validation images"
    )

    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_type",
        "-m",
        choices=["vit", "swin"],
        default="swin",
        help="Type of transformer model (vit or swin)",
    )
    model_group.add_argument(
        "--output_dir",
        "-o",
        default="models/bayesian_hierarchical",
        help="Directory to save trained models and metadata",
    )

    # Class weight options
    weights_group = parser.add_argument_group("Class Weights")
    weights_group.add_argument(
        "--level1_weights",
        default=None,
        type=str,
        help="Comma-separated weights for Level 1 training (0 vs 1+)",
    )
    weights_group.add_argument(
        "--level2_weights",
        default=None,
        type=str,
        help="Comma-separated weights for Level 2 training (1 vs 2+)",
    )

    # Training options
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--epochs", "-e", type=int, default=20, help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch_size", "-b", type=int, default=16, help="Batch size for training"
    )
    training_group.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2e-4,
        help="Learning rate for classifier head",
    )
    training_group.add_argument(
        "--backbone_lr_multiplier",
        "-blrm",
        type=float,
        default=0.1,
        help="Multiplier for backbone learning rate",
    )
    training_group.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    training_group.add_argument(
        "--grad_clip", "-gc", type=float, default=1.0, help="Gradient clipping max norm"
    )

    # Reproducibility options
    repro_group = parser.add_argument_group("Reproducibility")
    repro_group.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    repro_group.add_argument(
        "--deterministic", "-d", action="store_true", help="Enable deterministic mode"
    )
    
    # Performance options
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--workers", "-w", type=int, 
        help="Number of dataloader workers (default: from config, typically 4). Set to 0 or 1 to avoid shared memory errors."
    )

    args = parser.parse_args()

    # Parse class weights if provided
    level1_weights = None
    level2_weights = None

    if args.level1_weights:
        level1_weights = [float(w) for w in args.level1_weights.split(",")]
        print(f"Using custom Level 1 weights: {level1_weights}")

    if args.level2_weights:
        level2_weights = [float(w) for w in args.level2_weights.split(",")]
        print(f"Using custom Level 2 weights: {level2_weights}")

    return args, level1_weights, level2_weights


def main():
    # Parse arguments
    args, level1_weights, level2_weights = parse_arguments()
    
    # Get configuration singleton
    config = get_config()
    
    # Set number of workers if specified
    if args.workers is not None:
        config.update_model_param("num_workers", args.workers)
        print(f"Using {args.workers} dataloader workers (custom value from command line)")
        if args.workers <= 1:
            print("Reduced worker count should help avoid shared memory (shm) issues")

    # Create trainer
    trainer = BayesianHierarchicalTrainer(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        val_csv=args.val_csv,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        deterministic=args.deterministic,
        level1_weights=level1_weights,
        level2_weights=level2_weights,
        look_at_misclassifications=args.look_at_misclassifications,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        lower_class1_threshold=args.lower_class1_threshold,
    )

    # Train the models
    results = trainer.train()

    # Print final summary
    print("\nTraining complete!")
    print(f"Level 1 model: {results['level1_model']}")
    print(f"Level 2 model: {results['level2_model']}")
    print(f"Training metadata: {results['metadata']}")
    print("\nClass weights and frequencies:")
    print(f"Level 1 weights: {results['training_info']['level1']['class_weights']}")
    print(
        f"Level 1 frequencies: {results['training_info']['level1']['class_frequencies']}"
    )
    print(f"Level 2 weights: {results['training_info']['level2']['class_weights']}")
    print(
        f"Level 2 frequencies: {results['training_info']['level2']['class_frequencies']}"
    )


if __name__ == "__main__":
    main()
