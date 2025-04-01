#!/usr/bin/env python
"""
Lottery Ticket Hypothesis pruning for hierarchical transformer models.

This script implements the Lottery Ticket Hypothesis (LTH) approach to prune
hierarchical Vision Transformer models for receipt counting. The LTH approach
finds sparse subnetworks ("winning tickets") that can be trained to similar or
better performance than the original dense network.

The script can prune individual levels of the hierarchical model or the entire model.
"""

import os
import torch
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model_factory import ModelFactory
from prune_utils import (
    LotteryTicketPruner, 
    iterative_lottery_ticket_pruning,
    create_lottery_ticket_train_func,
    create_lottery_ticket_eval_func
)
from train_swin_classification import train_model
from train_hierarchical_model import (
    train_level1_model, 
    train_level2_model,
    train_multiclass_model
)
from hierarchical_predictor import HierarchicalPredictor
from evaluation import evaluate_model
from config import get_config
from device_utils import get_device


def prune_hierarchical_model(
    model_base_path,
    model_type,
    train_csv,
    train_dir,
    val_csv,
    val_dir,
    output_dir,
    pruning_percentages=None,
    epochs=10,
    batch_size=16,
    level="all",
    train_after_pruning=True,
    evaluate_models=True,
    hierarchical_eval=True,
    skip_existing=False
):
    """
    Prune a hierarchical model using the Lottery Ticket Hypothesis approach.
    
    Args:
        model_base_path: Base path to the hierarchical model
        model_type: Type of model ("vit" or "swin")
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file
        val_dir: Directory containing validation images
        output_dir: Directory to save pruned models
        pruning_percentages: List of pruning percentages to try
        epochs: Number of training epochs after pruning
        batch_size: Batch size for training
        level: Which level to prune ("level1", "level2", "multiclass", or "all")
        train_after_pruning: Whether to train the pruned models
        evaluate_models: Whether to evaluate the pruned models
        hierarchical_eval: Whether to evaluate using the hierarchical approach
        skip_existing: Skip pruning if a pruned model already exists
    """
    if pruning_percentages is None:
        pruning_percentages = [20, 40, 60, 80, 90]
    
    device = get_device()
    config = get_config()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which levels to prune
    levels_to_prune = []
    if level.lower() == "all":
        levels_to_prune = ["level1", "level2"]
        # Check if multiclass exists
        if (Path(model_base_path) / "multiclass").exists():
            levels_to_prune.append("multiclass")
    else:
        levels_to_prune = [level.lower()]
    
    pruning_results = {}
    
    for level_name in levels_to_prune:
        print(f"\n{'='*80}")
        print(f"PRUNING {level_name.upper()} MODEL")
        print(f"{'='*80}")
        
        level_output_dir = output_path / level_name
        level_output_dir.mkdir(exist_ok=True)
        
        # Model paths
        pretrained_model_path = Path(model_base_path) / level_name / f"receipt_counter_{model_type}_best.pth"
        
        if not pretrained_model_path.exists():
            print(f"Warning: Pretrained model not found at {pretrained_model_path}")
            print(f"Skipping pruning for {level_name}")
            continue
        
        # Determine number of classes based on level
        if level_name == "level1" or level_name == "level2":
            num_classes = 2  # Binary classification
        else:  # multiclass
            num_classes = 4  # 2, 3, 4, 5 receipts
        
        # Set binary mode for training/evaluation based on level
        binary = (level_name == "level1")
        hierarchical_level = level_name
        
        # Training function for pruned models
        def train_func(pruned_model_path, level=level_name):
            print(f"Training pruned {level} model...")
            
            # Prepare hierarchical datasets
            from datasets import prepare_hierarchical_datasets
            train_datasets = prepare_hierarchical_datasets(
                Path(train_csv).parent, 
                Path(train_csv).name, 
                output_path=Path(train_csv).parent
            )
            
            val_datasets = prepare_hierarchical_datasets(
                Path(val_csv).parent, 
                Path(val_csv).name, 
                output_path=Path(val_csv).parent
            )
            
            # Train based on level
            output_dir = level_output_dir / f"pruned_{pruning_pct:.1f}pct" / "trained"
            os.makedirs(output_dir, exist_ok=True)
            
            if level == "level1":
                trained_model = train_level1_model(
                    model_type=model_type,
                    train_csv=train_datasets['level1']['path'],
                    val_csv=val_datasets['level1']['path'],
                    train_dir=train_dir,
                    val_dir=val_dir,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    resume_checkpoint=pruned_model_path
                )
                return str(Path(output_dir) / f"receipt_counter_{model_type}_best.pth")
            
            elif level == "level2":
                trained_model = train_level2_model(
                    model_type=model_type,
                    train_csv=train_datasets['level2']['path'],
                    val_csv=val_datasets['level2']['path'],
                    train_dir=train_dir,
                    val_dir=val_dir,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    resume_checkpoint=pruned_model_path
                )
                return str(Path(output_dir) / f"receipt_counter_{model_type}_best.pth")
            
            elif level == "multiclass":
                trained_model = train_multiclass_model(
                    model_type=model_type,
                    train_csv=train_datasets['multiclass']['path'],
                    val_csv=val_datasets['multiclass']['path'],
                    train_dir=train_dir,
                    val_dir=val_dir,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    resume_checkpoint=pruned_model_path
                )
                return str(Path(output_dir) / f"receipt_counter_{model_type}_best.pth")
        
        # Evaluation function for pruned models
        def eval_func(model, level=level_name):
            print(f"Evaluating pruned {level} model...")
            
            # Prepare hierarchical datasets for evaluation
            from datasets import prepare_hierarchical_datasets
            val_datasets = prepare_hierarchical_datasets(
                Path(val_csv).parent, 
                Path(val_csv).name, 
                output_path=Path(val_csv).parent
            )
            
            # Adapt evaluation based on level
            if level == "level1":
                # Evaluate with level1 dataset (binary: 0 vs 1+)
                eval_csv = val_datasets['level1']['path']
                binary = True
                metrics = evaluate_model(
                    model=model,
                    test_csv=eval_csv,
                    test_dir=val_dir,
                    binary=binary
                )
            elif level == "level2":
                # Evaluate with level2 dataset (binary: 1 vs 2+)
                eval_csv = val_datasets['level2']['path']
                binary = True
                metrics = evaluate_model(
                    model=model,
                    test_csv=eval_csv,
                    test_dir=val_dir,
                    binary=binary
                )
            elif level == "multiclass":
                # Evaluate with multiclass dataset (2, 3, 4, 5 receipts)
                eval_csv = val_datasets['multiclass']['path']
                binary = False
                metrics = evaluate_model(
                    model=model,
                    test_csv=eval_csv,
                    test_dir=val_dir,
                    binary=binary
                )
            
            return metrics
        
        # Hierarchical evaluation - puts all the pieces together
        def hierarchical_eval_func(level=level_name, pruning_pct=None):
            print(f"Running hierarchical evaluation for pruned {level} model ({pruning_pct:.1f}%)...")
            
            # Create a HierarchicalPredictor with pruned models
            pruned_dir = level_output_dir / f"pruned_{pruning_pct:.1f}pct" / "trained"
            
            if not pruned_dir.exists():
                print(f"Trained pruned model directory not found: {pruned_dir}")
                return None
            
            # Need to determine which models to use - unpruned or pruned
            # We assume we're evaluating a pruned model for just one level at a time
            level1_path = pretrained_model_path if level != "level1" else pruned_dir / f"receipt_counter_{model_type}_best.pth"
            level2_path = pretrained_model_path if level != "level2" else pruned_dir / f"receipt_counter_{model_type}_best.pth"
            multiclass_path = pretrained_model_path if level != "multiclass" else pruned_dir / f"receipt_counter_{model_type}_best.pth"
            
            if not level1_path.exists() or not level2_path.exists():
                print("Missing required models for hierarchical evaluation")
                return None
            
            # Create hierarchical predictor
            predictor = HierarchicalPredictor(
                level1_model_path=level1_path,
                level2_model_path=level2_path,
                multiclass_model_path=multiclass_path if Path(multiclass_path).exists() else None,
                model_type=model_type
            )
            
            # Evaluate
            metrics = predictor.evaluate_on_dataset(
                csv_file=val_csv,
                image_dir=val_dir,
                output_dir=pruned_dir / "evaluation"
            )
            
            return metrics
        
        # Run pruning for each percentage
        level_results = []
        for pruning_pct in pruning_percentages:
            print(f"\n--- Pruning {level_name} model at {pruning_pct:.1f}% ---")
            
            # Check if this pruned model already exists
            pruned_dir = level_output_dir / f"pruned_{pruning_pct:.1f}pct"
            pruned_model_path = pruned_dir / f"pruned_{model_type}_model.pth"
            
            if pruned_model_path.exists() and skip_existing:
                print(f"Pruned model already exists at {pruned_model_path}. Skipping.")
                
                # Load existing results if available
                results_path = pruned_dir / "results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        result = json.load(f)
                        level_results.append(result)
                continue
            
            # Create pruner
            pruner = LotteryTicketPruner(
                model_type=model_type,
                num_classes=num_classes,
                save_dir=level_output_dir
            )
            
            # Create pruned model
            pruned_model = pruner.create_pruned_model(pruning_pct, pretrained_model_path)
            pruned_model_path = pruner.save_pruned_model(pruning_pct)
            
            # Train pruned model if requested
            trained_model_path = None
            if train_after_pruning:
                try:
                    trained_model_path = train_func(pruned_model_path)
                    print(f"Trained pruned model saved to {trained_model_path}")
                except Exception as e:
                    print(f"Error training pruned model: {e}")
            
            # Evaluate pruned model if requested
            eval_metrics = None
            if evaluate_models:
                if trained_model_path and Path(trained_model_path).exists():
                    # Load the trained pruned model
                    pruned_model = ModelFactory.load_model(
                        trained_model_path, 
                        model_type=model_type, 
                        num_classes=num_classes
                    )
                
                try:
                    # Basic evaluation
                    eval_metrics = eval_func(pruned_model)
                    print(f"Evaluation metrics: {eval_metrics}")
                    
                    # Hierarchical evaluation
                    if hierarchical_eval and train_after_pruning:
                        hierarchical_metrics = hierarchical_eval_func(level_name, pruning_pct)
                        if hierarchical_metrics:
                            eval_metrics['hierarchical'] = hierarchical_metrics
                except Exception as e:
                    print(f"Error evaluating pruned model: {e}")
            
            # Save results
            result = {
                'level': level_name,
                'pruning_percentage': pruning_pct,
                'model_path': str(pruned_model_path),
                'trained_model_path': trained_model_path,
                'metrics': eval_metrics
            }
            
            # Save result to file
            results_path = pruned_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            level_results.append(result)
        
        # Save all results for this level
        pruning_results[level_name] = level_results
        
        # Plot metrics vs. pruning percentage
        if level_results and evaluate_models:
            plot_metrics_vs_pruning(level_results, level_output_dir, level_name)
    
    # Save overall results
    with open(output_path / "pruning_results.json", 'w') as f:
        json.dump(pruning_results, f, indent=4)
    
    return pruning_results


def plot_metrics_vs_pruning(results, output_dir, level_name):
    """
    Plot metrics vs. pruning percentage.
    
    Args:
        results: List of pruning results
        output_dir: Directory to save plots
        level_name: Name of the level (for title)
    """
    # Extract pruning percentages and metrics
    pruning_pcts = []
    accuracy = []
    balanced_accuracy = []
    f1_macro = []
    
    for result in results:
        pct = result['pruning_percentage']
        metrics = result.get('metrics', {})
        
        if not metrics:
            continue
        
        pruning_pcts.append(pct)
        
        # Handle hierarchical metrics differently
        if 'hierarchical' in metrics:
            hier_metrics = metrics['hierarchical']['overall']
            accuracy.append(hier_metrics.get('accuracy', 0))
            balanced_accuracy.append(hier_metrics.get('balanced_accuracy', 0))
            f1_macro.append(hier_metrics.get('f1_macro', 0))
        else:
            # Regular metrics
            accuracy.append(metrics.get('accuracy', 0))
            balanced_accuracy.append(metrics.get('balanced_accuracy', 0))
            f1_macro.append(metrics.get('f1_macro', 0))
    
    if not pruning_pcts:
        print("No metrics to plot")
        return
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(pruning_pcts, accuracy, marker='o')
    plt.title(f'Accuracy vs. Pruning Percentage - {level_name.upper()}')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(pruning_pcts, balanced_accuracy, marker='o')
    plt.title(f'Balanced Accuracy vs. Pruning Percentage - {level_name.upper()}')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Balanced Accuracy')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(pruning_pcts, f1_macro, marker='o')
    plt.title(f'F1 Macro vs. Pruning Percentage - {level_name.upper()}')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('F1 Macro')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{level_name}_metrics_vs_pruning.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Lottery Ticket Hypothesis pruning for hierarchical Vision Transformer models"
    )
    
    # Model and data options
    parser.add_argument(
        "--model_base_path", "-m", required=True,
        help="Base path to the hierarchical model directory"
    )
    parser.add_argument(
        "--model_type", choices=["vit", "swin"], default="swin",
        help="Type of model (vit or swin)"
    )
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
    
    # Pruning options
    parser.add_argument(
        "--output_dir", "-o", default="models/pruned",
        help="Directory to save pruned models"
    )
    parser.add_argument(
        "--pruning_percentages", "-p", type=float, nargs="+", 
        default=[20, 40, 60, 80, 90],
        help="Pruning percentages to try (default: 20 40 60 80 90)"
    )
    parser.add_argument(
        "--level", "-l", choices=["level1", "level2", "multiclass", "all"], default="all",
        help="Which level to prune (default: all)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip pruning if pruned model already exists"
    )
    
    # Training options
    parser.add_argument(
        "--no_train", action="store_true",
        help="Skip training after pruning"
    )
    parser.add_argument(
        "--no_evaluate", action="store_true",
        help="Skip evaluation after pruning/training"
    )
    parser.add_argument(
        "--no_hierarchical_eval", action="store_true",
        help="Skip hierarchical evaluation (only evaluate individual levels)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=10,
        help="Number of training epochs after pruning (default: 10)"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size for training (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Run pruning
    results = prune_hierarchical_model(
        model_base_path=args.model_base_path,
        model_type=args.model_type,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        val_csv=args.val_csv,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        pruning_percentages=args.pruning_percentages,
        epochs=args.epochs,
        batch_size=args.batch_size,
        level=args.level,
        train_after_pruning=not args.no_train,
        evaluate_models=not args.no_evaluate,
        hierarchical_eval=not args.no_hierarchical_eval,
        skip_existing=args.skip_existing
    )
    
    # Print summary of results
    print("\nPruning Summary:")
    for level, level_results in results.items():
        print(f"\n{level.upper()} Model:")
        for result in level_results:
            pct = result['pruning_percentage']
            metrics = result.get('metrics', {})
            
            if not metrics:
                print(f"  {pct:.1f}% pruned: No metrics available")
                continue
            
            if 'hierarchical' in metrics:
                hier_metrics = metrics['hierarchical']['overall']
                print(f"  {pct:.1f}% pruned: Accuracy={hier_metrics.get('accuracy', 0):.4f}, "
                      f"F1={hier_metrics.get('f1_macro', 0):.4f}")
            else:
                print(f"  {pct:.1f}% pruned: Accuracy={metrics.get('accuracy', 0):.4f}, "
                      f"F1={metrics.get('f1_macro', 0):.4f}")


if __name__ == "__main__":
    main()