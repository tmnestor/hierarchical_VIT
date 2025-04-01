#!/usr/bin/env python
"""
Evaluate a pruned model on a test dataset.

This script loads a pruned model created using the Lottery Ticket Hypothesis
approach and evaluates its performance on a test dataset.
"""

import argparse
from pathlib import Path
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

from prune_utils import load_pruned_model
from device_utils import get_device
from datasets import ReceiptDataset, create_data_loaders
from hierarchical_predictor import HierarchicalPredictor


def evaluate_pruned_model(
    model_path,
    test_csv,
    test_dir,
    output_dir=None,
    model_type="swin",
    level=None,
    hierarchical=False,
    level1_path=None,
    level2_path=None,
    multiclass_path=None
):
    """
    Evaluate a pruned model on a test dataset.
    
    Args:
        model_path: Path to the pruned model
        test_csv: Path to test CSV file
        test_dir: Directory containing test images
        output_dir: Directory to save evaluation results
        model_type: Type of model ("vit" or "swin")
        level: Level of the model ("level1", "level2", "multiclass")
        hierarchical: Whether to evaluate using hierarchical approach
        level1_path: Path to Level 1 model (for hierarchical evaluation)
        level2_path: Path to Level 2 model (for hierarchical evaluation)
        multiclass_path: Path to multiclass model (for hierarchical evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load pruned model
    try:
        model = load_pruned_model(model_path, device=device)
        print(f"Loaded pruned model from {model_path}")
        print(f"Model type: {model_type}")
    except Exception as e:
        print(f"Error loading pruned model: {e}")
        return None
    
    # Set model to evaluation mode
    model.eval()
    
    # Determine appropriate dataset based on level
    if level == "level1":
        binary = True
        hierarchical_level = "level1"
    elif level == "level2":
        binary = True
        hierarchical_level = "level2"
    elif level == "multiclass":
        binary = False
        hierarchical_level = "multiclass"
    else:
        binary = False
        hierarchical_level = None
    
    # Create test dataset
    test_dataset = ReceiptDataset(
        csv_file=test_csv,
        img_dir=test_dir,
        binary=binary,
        hierarchical_level=hierarchical_level
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=1  # Use 1 worker to avoid shared memory issues
    )
    
    print(f"Evaluating on {len(test_dataset)} test samples")
    
    # Hierarchical evaluation
    if hierarchical:
        return evaluate_hierarchical(
            model_path=model_path,
            test_csv=test_csv,
            test_dir=test_dir,
            output_dir=output_dir,
            model_type=model_type,
            level=level,
            level1_path=level1_path,
            level2_path=level2_path,
            multiclass_path=multiclass_path
        )
    
    # Standard evaluation
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle different model output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
            else:
                predictions = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='macro')
    report = classification_report(all_targets, all_predictions, output_dict=True)
    cm = confusion_matrix(all_targets, all_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Macro: {f1:.4f}")
    
    # Save results if output directory provided
    if output_dir:
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'file': test_dataset.data['filename'],
            'actual': all_targets,
            'predicted': all_predictions
        })
        
        # Save to CSV
        results_df.to_csv(output_path / "evaluation_results.csv", index=False)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=range(max(all_targets) + 1), 
                  yticklabels=range(max(all_targets) + 1))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(output_path / "confusion_matrix.png")
        plt.close()
        
        # Save metrics to JSON
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1,
            'classification_report': report
        }
        
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1,
        'classification_report': report
    }


def evaluate_hierarchical(
    model_path,
    test_csv,
    test_dir,
    output_dir=None,
    model_type="swin",
    level=None,
    level1_path=None,
    level2_path=None,
    multiclass_path=None
):
    """
    Evaluate a pruned model using the hierarchical approach.
    
    Args:
        model_path: Path to the pruned model
        test_csv: Path to test CSV file
        test_dir: Directory containing test images
        output_dir: Directory to save evaluation results
        model_type: Type of model ("vit" or "swin")
        level: Level of the model ("level1", "level2", "multiclass")
        level1_path: Path to Level 1 model (for hierarchical evaluation)
        level2_path: Path to Level 2 model (for hierarchical evaluation)
        multiclass_path: Path to multiclass model (for hierarchical evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Determine which paths to use
    if level == "level1":
        level1_path = model_path
    elif level == "level2":
        level2_path = model_path
    elif level == "multiclass":
        multiclass_path = model_path
    
    # Check if we have the necessary models
    if not level1_path:
        print("Level 1 model path required for hierarchical evaluation")
        return None
    
    if not level2_path:
        print("Level 2 model path required for hierarchical evaluation")
        return None
    
    print(f"Creating hierarchical predictor with pruned {level} model")
    print(f"Level 1 model: {level1_path}")
    print(f"Level 2 model: {level2_path}")
    if multiclass_path:
        print(f"Multiclass model: {multiclass_path}")
    
    # Create hierarchical predictor
    predictor = HierarchicalPredictor(
        level1_model_path=level1_path,
        level2_model_path=level2_path,
        multiclass_model_path=multiclass_path,
        model_type=model_type
    )
    
    # Evaluate
    metrics = predictor.evaluate_on_dataset(
        csv_file=test_csv,
        image_dir=test_dir,
        output_dir=output_dir
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pruned model on a test dataset"
    )
    
    # Model and data options
    parser.add_argument(
        "--model_path", "-m", required=True,
        help="Path to the pruned model"
    )
    parser.add_argument(
        "--test_csv", "-tc", required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--test_dir", "-td", required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--model_type",
        choices=["vit", "swin"],
        default="swin",
        help="Type of model (vit or swin)"
    )
    parser.add_argument(
        "--level",
        choices=["level1", "level2", "multiclass"],
        help="Level of the model"
    )
    
    # Hierarchical evaluation options
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Evaluate using hierarchical approach"
    )
    parser.add_argument(
        "--level1_path",
        help="Path to Level 1 model (for hierarchical evaluation)"
    )
    parser.add_argument(
        "--level2_path",
        help="Path to Level 2 model (for hierarchical evaluation)"
    )
    parser.add_argument(
        "--multiclass_path",
        help="Path to multiclass model (for hierarchical evaluation)"
    )
    
    args = parser.parse_args()
    
    # Evaluate pruned model
    metrics = evaluate_pruned_model(
        model_path=args.model_path,
        test_csv=args.test_csv,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        level=args.level,
        hierarchical=args.hierarchical,
        level1_path=args.level1_path,
        level2_path=args.level2_path,
        multiclass_path=args.multiclass_path
    )
    
    if not metrics:
        return
    
    # Print overall metrics
    print("\nEvaluation Results:")
    if 'overall' in metrics:
        # Hierarchical metrics
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Overall Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.4f}")
        print(f"Overall F1 Macro: {metrics['overall']['f1_macro']:.4f}")
        
        # Level-specific metrics
        if 'level1' in metrics:
            print(f"\nLevel 1 (0 vs 1+) Accuracy: {metrics['level1']['accuracy']:.4f}")
            print(f"Level 1 Balanced Accuracy: {metrics['level1']['balanced_accuracy']:.4f}")
        
        if 'level2' in metrics:
            print(f"\nLevel 2 (1 vs 2+) Accuracy: {metrics['level2']['accuracy']:.4f}")
            print(f"Level 2 Balanced Accuracy: {metrics['level2']['balanced_accuracy']:.4f}")
    else:
        # Standard metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()