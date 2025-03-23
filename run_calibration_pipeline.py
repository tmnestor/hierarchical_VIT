#!/usr/bin/env python3
"""
Complete training-evaluation pipeline with temperature calibration and Bayesian correction.
This script coordinates the entire process from training to evaluation with optimal parameters.
"""

import argparse
import subprocess
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

def run_command(command, description=None):
    """Run a shell command and print its output"""
    if description:
        print(f"\n{description}")
        print(f"Running: {command}")
    
    start_time = time.time()
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    # Stream output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
        sys.stdout.flush()
    
    # Wait for process to complete
    process.wait()
    
    # Check if command was successful
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    else:
        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f} seconds")
    
    return process.returncode, ''.join(output_lines)

def run_pipeline(args):
    """Run the complete pipeline"""
    # Create output directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Train hierarchical models (if needed)
    if args.train:
        print("\n=== TRAINING HIERARCHICAL MODELS ===")
        
        # Train command
        train_cmd = (
            f"python train_bayesian_hierarchical.py "
            f"--train_csv {args.train_csv} "
            f"--train_dir {args.train_dir} "
            f"--val_csv {args.val_csv} "
            f"--val_dir {args.val_dir} "
            f"--model_type {args.model_type} "
            f"--output_dir {args.model_dir} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--seed {args.seed} "
            f"{'-d' if args.deterministic else ''}"
        )
        
        return_code, _ = run_command(train_cmd, "Training hierarchical models")
        
        if return_code != 0:
            print("Training failed. Exiting.")
            return
    
    # 2. First run evaluation without temperature calibration as baseline
    print("\n=== RUNNING BASELINE EVALUATION ===")
    baseline_cmd = (
        f"python bayesian_hierarchical_predictor.py "
        f"--mode evaluate "
        f"--model_base_path {args.model_dir} "
        f"--test_csv {args.test_csv} "
        f"--test_dir {args.test_dir} "
        f"--model_type {args.model_type} "
        f"--output_dir {args.output_dir}/baseline "
        f"--temperature 0.0 "  # No Bayesian correction
        f"{'-debug' if args.debug else ''}"
    )
    
    run_command(baseline_cmd, "Running baseline evaluation (no Bayesian correction)")
    
    # 3. Run temperature calibration
    if not args.skip_calibration:
        print("\n=== RUNNING TEMPERATURE CALIBRATION ===")
        calibrate_cmd = (
            f"python bayesian_temperature_calibration.py "
            f"--calibrate "
            f"--model_base_path {args.model_dir} "
            f"--model_type {args.model_type} "
            f"--test_csv {args.val_csv} "
            f"--test_dir {args.val_dir} "
            f"{'-debug' if args.debug else ''}"
        )
        
        return_code, _ = run_command(calibrate_cmd, "Calibrating model temperatures")
        
        if return_code != 0:
            print("Calibration failed. Using default temperatures.")
    
    # 4. Run temperature sweep to find optimal Bayesian correction temperature
    if args.sweep:
        print("\n=== RUNNING TEMPERATURE SWEEP ===")
        sweep_cmd = (
            f"python bayesian_temperature_calibration.py "
            f"--sweep "
            f"--model_base_path {args.model_dir} "
            f"--model_type {args.model_type} "
            f"--test_csv {args.test_csv} "
            f"--test_dir {args.test_dir} "
            f"--output_dir {args.output_dir}/sweep "
            f"--min_temp {args.min_temp} "
            f"--max_temp {args.max_temp} "
            f"--num_temps {args.num_temps} "
            f"{'--log_scale' if args.log_scale else ''} "
            f"{'-debug' if args.debug else ''}"
        )
        
        return_code, output = run_command(sweep_cmd, "Running Bayesian temperature sweep")
        
        if return_code != 0:
            print("Temperature sweep failed. Using default temperature.")
            best_temp = args.bayesian_temperature
        else:
            # Try to extract best temperature from output
            for line in output.split('\n'):
                if "Best temperature for balanced accuracy" in line:
                    try:
                        best_temp = float(line.split(":")[1].strip())
                        break
                    except:
                        best_temp = args.bayesian_temperature
            else:
                best_temp = args.bayesian_temperature
            
            print(f"Using best temperature from sweep: {best_temp}")
    else:
        best_temp = args.bayesian_temperature
    
    # 5. Run final evaluation with optimal parameters
    print("\n=== RUNNING FINAL EVALUATION ===")
    
    # Try to load calibration parameters if available
    cal_params = {}
    cal_path = Path(args.model_dir) / "calibration_params.json"
    if cal_path.exists():
        try:
            with open(cal_path, 'r') as f:
                cal_params = json.load(f)
            level1_temp = cal_params.get('level1_temperature', 1.0)
            level2_temp = cal_params.get('level2_temperature', 1.0)
        except:
            level1_temp = 1.0
            level2_temp = 1.0
    else:
        level1_temp = 1.0
        level2_temp = 1.0
    
    final_cmd = (
        f"python bayesian_temperature_calibration.py "
        f"--mode evaluate "
        f"--model_base_path {args.model_dir} "
        f"--model_type {args.model_type} "
        f"--test_csv {args.test_csv} "
        f"--test_dir {args.test_dir} "
        f"--output_dir {args.output_dir}/calibrated "
        f"--level1_temperature {level1_temp} "
        f"--level2_temperature {level2_temp} "
        f"--bayesian_temperature {best_temp} "
        f"{'-debug' if args.debug else ''}"
    )
    
    run_command(final_cmd, "Running final evaluation with optimized parameters")
    
    # 6. Also run confident calibrated predictor for comparison
    print("\n=== RUNNING CONFIDENT CALIBRATED PREDICTOR FOR COMPARISON ===")
    
    # First get paths to calibrated model if they exist
    level1_model = Path(args.model_dir) / "calibrated" / f"{args.model_type}_level1_calibrated.pth"
    level2_model = Path(args.model_dir) / "calibrated" / f"{args.model_type}_level2_calibrated.pth"
    
    # Use standard models if calibrated ones don't exist
    if not level1_model.exists():
        level1_model = Path(args.model_dir) / "level1" / f"receipt_counter_{args.model_type}_best.pth"
    
    if not level2_model.exists():
        level2_model = Path(args.model_dir) / "level2" / f"receipt_counter_{args.model_type}_best.pth"
    
    confident_cmd = (
        f"python confident_calibrated_predictor.py "
        f"--level1_model {level1_model} "
        f"--level2_model {level2_model} "
        f"--test_csv {args.test_csv} "
        f"--test_dir {args.test_dir} "
        f"--model_type {args.model_type} "
        f"--output_dir {args.output_dir}/confident "
        f"--confidence_threshold {args.confidence_threshold} "
        f"--fallback_threshold {args.fallback_threshold}"
    )
    
    run_command(confident_cmd, "Evaluating with confident calibrated predictor")
    
    # 7. Compare results from different approaches
    print("\n=== SUMMARY OF RESULTS ===")
    try:
        results = {}
        
        # Load baseline results
        try:
            baseline_metrics = f"{args.output_dir}/baseline/metrics_0.000000.json"
            if os.path.exists(baseline_metrics):
                with open(baseline_metrics, 'r') as f:
                    data = json.load(f)
                    results['baseline'] = {
                        'accuracy': data['overall']['accuracy'],
                        'balanced_accuracy': data['overall']['balanced_accuracy'],
                        'f1_macro': data['overall']['f1_macro']
                    }
        except Exception as e:
            print(f"Could not load baseline metrics: {e}")
        
        # Load Bayesian calibrated results
        try:
            calibrated_metrics = f"{args.output_dir}/calibrated/metrics_{best_temp:.6f}.json"
            if os.path.exists(calibrated_metrics):
                with open(calibrated_metrics, 'r') as f:
                    data = json.load(f)
                    results['calibrated'] = {
                        'accuracy': data['overall']['accuracy'],
                        'balanced_accuracy': data['overall']['balanced_accuracy'],
                        'f1_macro': data['overall']['f1_macro']
                    }
        except Exception as e:
            print(f"Could not load calibrated metrics: {e}")
        
        # Load confident predictor results
        try:
            confident_metrics = f"{args.output_dir}/confident/{args.model_type}_metrics.csv"
            if os.path.exists(confident_metrics):
                df = pd.read_csv(confident_metrics)
                results['confident'] = {
                    'accuracy': df['accuracy'].iloc[0],
                    'balanced_accuracy': df['balanced_accuracy'].iloc[0] if 'balanced_accuracy' in df.columns else None,
                    'f1_score': df['f1_score'].iloc[0] if 'f1_score' in df.columns else None,
                    'abstention_rate': df['abstention_rate'].iloc[0] if 'abstention_rate' in df.columns else None
                }
        except Exception as e:
            print(f"Could not load confident metrics: {e}")
        
        # Print comparison
        if results:
            print("\nComparison of different approaches:")
            for approach, metrics in results.items():
                print(f"\n{approach.capitalize()} approach:")
                for metric, value in metrics.items():
                    if value is not None:
                        print(f"  {metric}: {value:.4f}")
            
            # Create comparison plot
            methods = list(results.keys())
            metrics = ['accuracy', 'balanced_accuracy']
            values = {metric: [results[method].get(metric, 0) for method in methods] for metric in metrics}
            
            plt.figure(figsize=(10, 6))
            width = 0.35
            x = np.arange(len(methods))
            
            plt.bar(x - width/2, values['accuracy'], width, label='Accuracy')
            plt.bar(x + width/2, values['balanced_accuracy'], width, label='Balanced Accuracy')
            
            plt.xlabel('Method')
            plt.ylabel('Score')
            plt.title('Comparison of Different Approaches')
            plt.xticks(x, methods)
            plt.legend()
            plt.grid(alpha=0.3)
            
            os.makedirs(f"{args.output_dir}/comparison", exist_ok=True)
            plt.savefig(f"{args.output_dir}/comparison/method_comparison.png")
            plt.close()
            
            # Save comparison data
            pd.DataFrame(results).T.to_csv(f"{args.output_dir}/comparison/method_comparison.csv")
            
    except Exception as e:
        print(f"Error generating comparison: {e}")
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to {args.output_dir}")

def parse_arguments():
    """
    Parse command line arguments for the complete training-evaluation pipeline.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run complete training-evaluation pipeline")
    
    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train", action="store_true", help="Train models")
    train_group.add_argument("--train_csv", default="receipts/train.csv", help="Training CSV")
    train_group.add_argument("--train_dir", default="receipts/train", help="Training directory")
    train_group.add_argument("--val_csv", default="receipts/val.csv", help="Validation CSV")
    train_group.add_argument("--val_dir", default="receipts/val", help="Validation directory")
    train_group.add_argument("--epochs", type=int, default=15, help="Training epochs")
    train_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")
    train_group.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    
    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_dir", default="models/calibrated_hierarchical", 
                          help="Directory for model storage")
    model_group.add_argument("--model_type", choices=["vit", "swin"], default="swin",
                          help="Model type")
    
    # Calibration options
    cal_group = parser.add_argument_group("Calibration")
    cal_group.add_argument("--skip_calibration", action="store_true", 
                         help="Skip model confidence calibration temperature calibration")
    cal_group.add_argument("--bayesian_temperature", type=float, default=0.01,
                         help="Default Bayesian correction temperature (0.0-1.0 range)")
    
    # Bayesian Temperature sweep options
    sweep_group = parser.add_argument_group("Bayesian Temperature Sweep")
    sweep_group.add_argument("--sweep", action="store_true", 
                           help="Run a Bayesian correction temperature sweep")
    sweep_group.add_argument("--min_temp", type=float, default=0.001,
                           help="Minimum Bayesian temperature for sweep (0.0-1.0 range)")
    sweep_group.add_argument("--max_temp", type=float, default=0.1,
                           help="Maximum Bayesian temperature for sweep (0.0-1.0 range)")
    sweep_group.add_argument("--num_temps", type=int, default=10,
                           help="Number of Bayesian temperature values to evaluate")
    sweep_group.add_argument("--log_scale", action="store_true",
                           help="Use logarithmic spacing for Bayesian temperature values")
    
    # Confident predictor options
    conf_group = parser.add_argument_group("Confident Predictor")
    conf_group.add_argument("--confidence_threshold", type=float, default=0.8,
                          help="Confidence threshold for high confidence predictions")
    conf_group.add_argument("--fallback_threshold", type=float, default=0.6,
                          help="Fallback threshold for low confidence predictions")
    
    # Test options
    test_group = parser.add_argument_group("Test")
    test_group.add_argument("--test_csv", default="receipts/test.csv", help="Test CSV")
    test_group.add_argument("--test_dir", default="receipts/test", help="Test directory")
    test_group.add_argument("--output_dir", default="evaluation/pipeline",
                          help="Output directory for evaluation")
    
    # Other options
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return parser.parse_args()

def main():
    """
    Main entry point for the complete training-evaluation pipeline.
    """
    args = parse_arguments()
    run_pipeline(args)

if __name__ == "__main__":
    main()