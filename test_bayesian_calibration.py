#!/usr/bin/env python3
"""
Comprehensive testing framework for evaluating hierarchical models with different 
threshold and focal loss configurations.

This script performs parameter sweeps to find optimal settings for:
1. Level 1 classification threshold adjustments (0 vs 1+ receipts)
2. Level 2 classification threshold adjustments (1 vs 2+ receipts)
3. Focal loss gamma parameter effects
4. Bayesian correction effectiveness

The results are analyzed to find the optimal settings that minimize false negatives
while maintaining overall performance metrics.
"""

import torch
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve, 
    auc
)
from itertools import product

# Local imports
from device_utils import get_device
from hierarchical_predictor import HierarchicalPredictor
from bayesian_temperature_calibration import BayesianCalibratedPredictor
from model_factory import ModelFactory
from datasets import ReceiptDataset
from torch.utils.data import DataLoader


class ThresholdSweeper:
    """
    Performs parameter sweeps across different classification thresholds
    and focal loss settings, evaluating the impact on model performance.
    """
    
    def __init__(
        self,
        model_base_path,
        test_csv,
        test_dir,
        output_dir,
        model_type="swin",
        level1_threshold_values=None,
        level2_threshold_values=None,
        focal_gamma_values=None,
        bayesian_temp_values=None,
        device=None,
        enhance_images=True,
    ):
        """
        Initialize the parameter sweeper.
        
        Args:
            model_base_path: Base path to the hierarchical models
            test_csv: Path to test CSV file
            test_dir: Directory containing test images
            model_type: Type of model ("vit" or "swin")
            level1_threshold_values: List of level 1 threshold values to sweep
            level2_threshold_values: List of level 2 threshold values to sweep
            focal_gamma_values: List of focal gamma values to evaluate
            bayesian_temp_values: List of Bayesian temperature values to evaluate
            device: Pytorch device
            enhance_images: Whether to enhance images before processing
        """
        self.model_base_path = Path(model_base_path)
        self.test_csv = test_csv
        self.test_dir = test_dir
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.device = device or get_device()
        self.enhance_images = enhance_images
        
        # Set default values if not provided
        self.level1_threshold_values = level1_threshold_values or [0.5, 0.4, 0.3, 0.2]
        self.level2_threshold_values = level2_threshold_values or [0.5, 0.4, 0.3, 0.2]
        self.focal_gamma_values = focal_gamma_values or [0.0, 1.0, 2.0, 3.0]
        self.bayesian_temp_values = bayesian_temp_values or [0.0, 0.01, 0.05, 0.1]
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.test_df = pd.read_csv(test_csv)
        print(f"Loaded {len(self.test_df)} samples from {test_csv}")
        
        # Map receipt counts for hierarchical evaluation
        if 'receipt_count' in self.test_df.columns:
            self.test_df['original_count'] = self.test_df['receipt_count']
            # For hierarchical evaluation, map all counts > 2 to 2
            self.test_df['receipt_count'] = self.test_df['receipt_count'].apply(lambda x: min(x, 2))
        
        # Identify model paths
        self.level1_model_path = self.model_base_path / "level1" / f"receipt_counter_{model_type}_best.pth"
        self.level2_model_path = self.model_base_path / "level2" / f"receipt_counter_{model_type}_best.pth"
        self.metadata_path = self.model_base_path / "training_metadata.json"
        
        # Validate paths
        if not self.level1_model_path.exists():
            raise FileNotFoundError(f"Level 1 model not found at {self.level1_model_path}")
        
        if not self.level2_model_path.exists():
            print(f"Warning: Level 2 model not found at {self.level2_model_path}")
            self.level2_model_path = None
            
        # Try to load metadata
        self.metadata = None
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Successfully loaded metadata from {self.metadata_path}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
    
    def run_single_evaluation(
        self, 
        level1_threshold=None, 
        level2_threshold=None, 
        use_bayesian=False,
        bayesian_temp=0.01
    ):
        """
        Run a single evaluation with the specified parameters.
        
        Args:
            level1_threshold: Threshold for level 1 model (0 vs 1+)
            level2_threshold: Threshold for level 2 model (1 vs 2+)
            use_bayesian: Whether to use Bayesian calibration
            bayesian_temp: Bayesian calibration temperature
            
        Returns:
            Dictionary with evaluation metrics
        """
        if use_bayesian:
            # Use Bayesian calibrated predictor
            predictor = BayesianCalibratedPredictor(
                level1_model_path=self.level1_model_path,
                level2_model_path=self.level2_model_path,
                model_type=self.model_type,
                metadata_path=self.metadata_path if self.metadata_path.exists() else None,
                bayesian_temperature=bayesian_temp,
            )
            
            # Run evaluation
            metrics = predictor.evaluate_on_dataset(
                csv_file=self.test_csv,
                image_dir=self.test_dir,
                output_dir=None,  # Don't save individual run results
                enhance=self.enhance_images,
                debug=False,
                bayesian_temperature=bayesian_temp
            )
            
            # Add parameter info to metrics
            metrics['parameters'] = {
                'level1_threshold': None,
                'level2_threshold': None,
                'use_bayesian': True,
                'bayesian_temp': bayesian_temp
            }
            
        else:
            # Use regular hierarchical predictor with threshold adjustments
            predictor = HierarchicalPredictor(
                level1_model_path=self.level1_model_path,
                level2_model_path=self.level2_model_path,
                model_type=self.model_type,
                lower_class1_threshold=level1_threshold
            )
            
            # Run evaluation
            metrics = predictor.evaluate_on_dataset(
                csv_file=self.test_csv,
                image_dir=self.test_dir,
                output_dir=None,  # Don't save individual run results
                enhance=self.enhance_images
            )
            
            # Add parameter info to metrics
            metrics['parameters'] = {
                'level1_threshold': level1_threshold,
                'level2_threshold': level2_threshold,
                'use_bayesian': False,
                'bayesian_temp': None
            }
        
        return metrics
    
    def calculate_fn_rate(self, confusion_mat):
        """
        Calculate false negative rate from a confusion matrix.
        
        Args:
            confusion_mat: Confusion matrix (2x2 for binary classification)
            
        Returns:
            False negative rate (float)
        """
        if confusion_mat.shape[0] < 2 or confusion_mat.shape[1] < 2:
            return 0.0
        
        # Ensure it's a numpy array
        cm = np.array(confusion_mat)
        
        # For binary classification
        fn = cm[1, 0]  # True class 1, predicted class 0
        tp = cm[1, 1]  # True class 1, predicted class 1
        
        # Avoid division by zero
        if fn + tp == 0:
            return 0.0
            
        fn_rate = fn / (fn + tp)
        return fn_rate
    
    def calculate_fp_rate(self, confusion_mat):
        """
        Calculate false positive rate from a confusion matrix.
        
        Args:
            confusion_mat: Confusion matrix (2x2 for binary classification)
            
        Returns:
            False positive rate (float)
        """
        if confusion_mat.shape[0] < 2 or confusion_mat.shape[1] < 2:
            return 0.0
        
        # Ensure it's a numpy array
        cm = np.array(confusion_mat)
        
        # For binary classification
        fp = cm[0, 1]  # True class 0, predicted class 1
        tn = cm[0, 0]  # True class 0, predicted class 0
        
        # Avoid division by zero
        if fp + tn == 0:
            return 0.0
            
        fp_rate = fp / (fp + tn)
        return fp_rate
    
    def run_threshold_sweep(self):
        """
        Run a sweep of different classification thresholds for both levels.
        
        This evaluates combinations of level 1 and level 2 thresholds to find
        the best trade-off points for minimizing false negatives.
        
        Returns:
            DataFrame with results
        """
        print(f"Running threshold sweep with {len(self.level1_threshold_values)} level 1 values and {len(self.level2_threshold_values)} level 2 values")
        
        # Store results
        results = []
        
        # Run evaluation for each threshold combination
        for level1_thresh in self.level1_threshold_values:
            for level2_thresh in self.level2_threshold_values:
                print(f"\nEvaluating with Level 1 threshold = {level1_thresh}, Level 2 threshold = {level2_thresh}")
                
                # Run evaluation
                metrics = self.run_single_evaluation(
                    level1_threshold=level1_thresh,
                    level2_threshold=level2_thresh,
                    use_bayesian=False
                )
                
                # Extract key metrics
                result = {
                    'level1_threshold': level1_thresh,
                    'level2_threshold': level2_thresh,
                    'accuracy': metrics['overall']['accuracy'],
                    'balanced_accuracy': metrics['overall']['balanced_accuracy'],
                    'f1_macro': metrics['overall']['f1_macro'],
                    'level1_accuracy': metrics['level1']['accuracy'],
                    'level1_balanced_accuracy': metrics['level1']['balanced_accuracy'],
                    'level2_accuracy': metrics['level2']['accuracy'],
                    'level2_balanced_accuracy': metrics['level2']['balanced_accuracy'],
                }
                
                # Calculate false negative rates
                if 'confusion_matrix' in metrics:
                    cm_overall = metrics['confusion_matrix']
                    
                    # Calculate level 1 false negative rate (0 vs 1+)
                    # Create a 2x2 binary confusion matrix for level 1
                    level1_cm = np.zeros((2, 2))
                    # True 0, Predicted 0
                    level1_cm[0, 0] = cm_overall[0, 0]
                    # True 0, Predicted 1+
                    level1_cm[0, 1] = cm_overall[0, 1] + cm_overall[0, 2]
                    # True 1+, Predicted 0
                    level1_cm[1, 0] = cm_overall[1, 0] + cm_overall[2, 0]
                    # True 1+, Predicted 1+
                    level1_cm[1, 1] = cm_overall[1, 1] + cm_overall[1, 2] + cm_overall[2, 1] + cm_overall[2, 2]
                    
                    # Calculate level 2 false negative rate (1 vs 2+)
                    # Create a 2x2 binary confusion matrix for level 2
                    level2_cm = np.zeros((2, 2))
                    # True 1, Predicted 1
                    level2_cm[0, 0] = cm_overall[1, 1]
                    # True 1, Predicted 2+
                    level2_cm[0, 1] = cm_overall[1, 2]
                    # True 2+, Predicted 1
                    level2_cm[1, 0] = cm_overall[2, 1]
                    # True 2+, Predicted 2+
                    level2_cm[1, 1] = cm_overall[2, 2]
                    
                    result['level1_fn_rate'] = self.calculate_fn_rate(level1_cm)
                    result['level1_fp_rate'] = self.calculate_fp_rate(level1_cm)
                    result['level2_fn_rate'] = self.calculate_fn_rate(level2_cm)
                    result['level2_fp_rate'] = self.calculate_fp_rate(level2_cm)
                    
                    # Store confusion matrices for reference
                    result['level1_confusion_matrix'] = level1_cm.tolist()
                    result['level2_confusion_matrix'] = level2_cm.tolist()
                    result['overall_confusion_matrix'] = cm_overall.tolist()
                
                # Add result
                results.append(result)
                
                # Print key metrics
                print(f"Accuracy: {result['accuracy']:.4f}, F1: {result['f1_macro']:.4f}")
                print(f"Level 1 (0 vs 1+): FN Rate: {result.get('level1_fn_rate', 'N/A'):.4f}, FP Rate: {result.get('level1_fp_rate', 'N/A'):.4f}")
                print(f"Level 2 (1 vs 2+): FN Rate: {result.get('level2_fn_rate', 'N/A'):.4f}, FP Rate: {result.get('level2_fp_rate', 'N/A'):.4f}")
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'threshold_sweep_results.csv', index=False)
        
        # Plot results - Overall metrics
        self._plot_threshold_heatmap(results_df, 'accuracy', 'Accuracy')
        self._plot_threshold_heatmap(results_df, 'balanced_accuracy', 'Balanced Accuracy')
        self._plot_threshold_heatmap(results_df, 'f1_macro', 'F1 Score')
        
        # Plot FN/FP rates
        if 'level1_fn_rate' in results_df.columns:
            self._plot_threshold_heatmap(results_df, 'level1_fn_rate', 'Level 1 False Negative Rate', cmap='Reds_r')
            self._plot_threshold_heatmap(results_df, 'level1_fp_rate', 'Level 1 False Positive Rate')
            self._plot_threshold_heatmap(results_df, 'level2_fn_rate', 'Level 2 False Negative Rate', cmap='Reds_r')
            self._plot_threshold_heatmap(results_df, 'level2_fp_rate', 'Level 2 False Positive Rate')
        
        # Plot FN/FP trade-off for level 1
        if 'level1_fn_rate' in results_df.columns and 'level1_fp_rate' in results_df.columns:
            self._plot_fn_fp_tradeoff(results_df, level=1)
            self._plot_fn_fp_tradeoff(results_df, level=2)
            
        # Find best settings
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_f1_idx = results_df['f1_macro'].idxmax()
        
        # For balanced FN/FP tradeoff, find setting with lowest combined rate
        if 'level1_fn_rate' in results_df.columns and 'level1_fp_rate' in results_df.columns:
            results_df['level1_combined_rate'] = results_df['level1_fn_rate'] + results_df['level1_fp_rate']
            best_level1_tradeoff_idx = results_df['level1_combined_rate'].idxmin()
            
            results_df['level2_combined_rate'] = results_df['level2_fn_rate'] + results_df['level2_fp_rate']
            best_level2_tradeoff_idx = results_df['level2_combined_rate'].idxmin()
            
            # Find settings with lowest FN rate but still with high F1 (above 90% of max)
            min_acceptable_f1 = 0.9 * results_df['f1_macro'].max()
            good_f1_mask = results_df['f1_macro'] >= min_acceptable_f1
            
            if good_f1_mask.any():
                good_f1_df = results_df[good_f1_mask]
                best_fn_idx = good_f1_df['level1_fn_rate'].idxmin()
                
                # Save recommended settings to a file
                with open(self.output_dir / 'recommended_threshold_settings.txt', 'w') as f:
                    f.write("RECOMMENDED THRESHOLD SETTINGS\n")
                    f.write("============================\n\n")
                    
                    f.write("Best for overall accuracy:\n")
                    f.write(f"  Level 1 threshold: {results_df.loc[best_accuracy_idx, 'level1_threshold']}\n")
                    f.write(f"  Level 2 threshold: {results_df.loc[best_accuracy_idx, 'level2_threshold']}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_accuracy_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_accuracy_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_accuracy_idx, 'level1_fn_rate']:.4f}\n\n")
                    
                    f.write("Best for F1 score:\n")
                    f.write(f"  Level 1 threshold: {results_df.loc[best_f1_idx, 'level1_threshold']}\n")
                    f.write(f"  Level 2 threshold: {results_df.loc[best_f1_idx, 'level2_threshold']}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_f1_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_f1_idx, 'level1_fn_rate']:.4f}\n\n")
                    
                    f.write("Best for Level 1 FN/FP tradeoff:\n")
                    f.write(f"  Level 1 threshold: {results_df.loc[best_level1_tradeoff_idx, 'level1_threshold']}\n")
                    f.write(f"  Level 2 threshold: {results_df.loc[best_level1_tradeoff_idx, 'level2_threshold']}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_level1_tradeoff_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_level1_tradeoff_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_level1_tradeoff_idx, 'level1_fn_rate']:.4f}\n")
                    f.write(f"  Level 1 FP Rate: {results_df.loc[best_level1_tradeoff_idx, 'level1_fp_rate']:.4f}\n\n")
                    
                    f.write("Best for minimizing FN rate with good F1 (>= 90% of max):\n")
                    f.write(f"  Level 1 threshold: {results_df.loc[best_fn_idx, 'level1_threshold']}\n")
                    f.write(f"  Level 2 threshold: {results_df.loc[best_fn_idx, 'level2_threshold']}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_fn_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_fn_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_fn_idx, 'level1_fn_rate']:.4f}\n")
                    f.write(f"  Level 1 FP Rate: {results_df.loc[best_fn_idx, 'level1_fp_rate']:.4f}\n")
        
        print(f"\nThreshold sweep complete! Results saved to {self.output_dir}")
        return results_df
    
    def run_bayesian_sweep(self):
        """
        Run a sweep of different Bayesian correction temperatures.
        
        This evaluates the effectiveness of Bayesian correction at different
        temperature values.
        
        Returns:
            DataFrame with results
        """
        print(f"Running Bayesian temperature sweep with {len(self.bayesian_temp_values)} values")
        
        # Store results
        results = []
        
        # Run evaluation for each Bayesian temperature
        for temp in self.bayesian_temp_values:
            print(f"\nEvaluating with Bayesian temperature = {temp:.4f}")
            
            # Run evaluation
            metrics = self.run_single_evaluation(
                use_bayesian=True,
                bayesian_temp=temp
            )
            
            # Extract key metrics
            result = {
                'bayesian_temp': temp,
                'accuracy': metrics['overall']['accuracy'],
                'balanced_accuracy': metrics['overall']['balanced_accuracy'],
                'f1_macro': metrics['overall']['f1_macro'],
                'level1_accuracy': metrics['level1']['accuracy'],
                'level1_balanced_accuracy': metrics['level1']['balanced_accuracy'],
                'level2_accuracy': metrics['level2']['accuracy'],
                'level2_balanced_accuracy': metrics['level2']['balanced_accuracy'],
            }
            
            # Calculate false negative rates
            if 'confusion_matrix' in metrics:
                cm_overall = metrics['confusion_matrix']
                
                # Calculate level 1 false negative rate (0 vs 1+)
                # Create a 2x2 binary confusion matrix for level 1
                level1_cm = np.zeros((2, 2))
                # True 0, Predicted 0
                level1_cm[0, 0] = cm_overall[0, 0]
                # True 0, Predicted 1+
                level1_cm[0, 1] = cm_overall[0, 1] + cm_overall[0, 2]
                # True 1+, Predicted 0
                level1_cm[1, 0] = cm_overall[1, 0] + cm_overall[2, 0]
                # True 1+, Predicted 1+
                level1_cm[1, 1] = cm_overall[1, 1] + cm_overall[1, 2] + cm_overall[2, 1] + cm_overall[2, 2]
                
                # Calculate level 2 false negative rate (1 vs 2+)
                # Create a 2x2 binary confusion matrix for level 2
                level2_cm = np.zeros((2, 2))
                # True 1, Predicted 1
                level2_cm[0, 0] = cm_overall[1, 1]
                # True 1, Predicted 2+
                level2_cm[0, 1] = cm_overall[1, 2]
                # True 2+, Predicted 1
                level2_cm[1, 0] = cm_overall[2, 1]
                # True 2+, Predicted 2+
                level2_cm[1, 1] = cm_overall[2, 2]
                
                result['level1_fn_rate'] = self.calculate_fn_rate(level1_cm)
                result['level1_fp_rate'] = self.calculate_fp_rate(level1_cm)
                result['level2_fn_rate'] = self.calculate_fn_rate(level2_cm)
                result['level2_fp_rate'] = self.calculate_fp_rate(level2_cm)
                
                # Store confusion matrices for reference
                result['level1_confusion_matrix'] = level1_cm.tolist()
                result['level2_confusion_matrix'] = level2_cm.tolist()
                result['overall_confusion_matrix'] = cm_overall.tolist()
            
            # Add result
            results.append(result)
            
            # Print key metrics
            print(f"Accuracy: {result['accuracy']:.4f}, F1: {result['f1_macro']:.4f}")
            
            # Handle level1 rates, checking if they're strings or numbers
            l1_fn = result.get('level1_fn_rate', 'N/A')
            l1_fp = result.get('level1_fp_rate', 'N/A')
            if isinstance(l1_fn, (int, float)) and isinstance(l1_fp, (int, float)):
                print(f"Level 1 (0 vs 1+): FN Rate: {l1_fn:.4f}, FP Rate: {l1_fp:.4f}")
            else:
                print(f"Level 1 (0 vs 1+): FN Rate: {l1_fn}, FP Rate: {l1_fp}")
            
            # Handle level2 rates, checking if they're strings or numbers
            l2_fn = result.get('level2_fn_rate', 'N/A') 
            l2_fp = result.get('level2_fp_rate', 'N/A')
            if isinstance(l2_fn, (int, float)) and isinstance(l2_fp, (int, float)):
                print(f"Level 2 (1 vs 2+): FN Rate: {l2_fn:.4f}, FP Rate: {l2_fp:.4f}")
            else:
                print(f"Level 2 (1 vs 2+): FN Rate: {l2_fn}, FP Rate: {l2_fp}")
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'bayesian_sweep_results.csv', index=False)
        
        # Plot results
        self._plot_bayesian_metrics(results_df)
        
        # Find best settings
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_f1_idx = results_df['f1_macro'].idxmax()
        
        # For balanced FN/FP tradeoff, find setting with lowest combined rate
        if 'level1_fn_rate' in results_df.columns and 'level1_fp_rate' in results_df.columns:
            results_df['level1_combined_rate'] = results_df['level1_fn_rate'] + results_df['level1_fp_rate']
            best_level1_tradeoff_idx = results_df['level1_combined_rate'].idxmin()
            
            # Find settings with lowest FN rate but still with high F1 (above 90% of max)
            min_acceptable_f1 = 0.9 * results_df['f1_macro'].max()
            good_f1_mask = results_df['f1_macro'] >= min_acceptable_f1
            
            if good_f1_mask.any():
                good_f1_df = results_df[good_f1_mask]
                best_fn_idx = good_f1_df['level1_fn_rate'].idxmin()
                
                # Save recommended settings to a file
                with open(self.output_dir / 'recommended_bayesian_settings.txt', 'w') as f:
                    f.write("RECOMMENDED BAYESIAN CORRECTION SETTINGS\n")
                    f.write("=====================================\n\n")
                    
                    f.write("Best for overall accuracy:\n")
                    f.write(f"  Bayesian Temperature: {results_df.loc[best_accuracy_idx, 'bayesian_temp']:.4f}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_accuracy_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_accuracy_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_accuracy_idx, 'level1_fn_rate']:.4f}\n\n")
                    
                    f.write("Best for F1 score:\n")
                    f.write(f"  Bayesian Temperature: {results_df.loc[best_f1_idx, 'bayesian_temp']:.4f}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_f1_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_f1_idx, 'level1_fn_rate']:.4f}\n\n")
                    
                    f.write("Best for Level 1 FN/FP tradeoff:\n")
                    f.write(f"  Bayesian Temperature: {results_df.loc[best_level1_tradeoff_idx, 'bayesian_temp']:.4f}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_level1_tradeoff_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_level1_tradeoff_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_level1_tradeoff_idx, 'level1_fn_rate']:.4f}\n")
                    f.write(f"  Level 1 FP Rate: {results_df.loc[best_level1_tradeoff_idx, 'level1_fp_rate']:.4f}\n\n")
                    
                    f.write("Best for minimizing FN rate with good F1 (>= 90% of max):\n")
                    f.write(f"  Bayesian Temperature: {results_df.loc[best_fn_idx, 'bayesian_temp']:.4f}\n")
                    f.write(f"  Accuracy: {results_df.loc[best_fn_idx, 'accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {results_df.loc[best_fn_idx, 'f1_macro']:.4f}\n")
                    f.write(f"  Level 1 FN Rate: {results_df.loc[best_fn_idx, 'level1_fn_rate']:.4f}\n")
                    f.write(f"  Level 1 FP Rate: {results_df.loc[best_fn_idx, 'level1_fp_rate']:.4f}\n")
        
        print(f"\nBayesian sweep complete! Results saved to {self.output_dir}")
        return results_df
    
    def compare_threshold_vs_bayesian(self):
        """
        Compare the performance of threshold adjustment vs Bayesian correction.
        
        This runs a limited sweep to compare the two approaches head-to-head
        and visualize their differences.
        
        Returns:
            DataFrame with comparison results
        """
        print("Running comparison: Threshold Adjustment vs Bayesian Correction")
        
        # Store results
        results = []
        
        # Threshold values to test
        threshold_values = [0.5, 0.4, 0.3, 0.2]
        
        # Bayesian temp values to test
        bayesian_values = [0.0, 0.01, 0.05, 0.1]
        
        # First, run threshold adjustments
        for threshold in threshold_values:
            print(f"\nEvaluating with Threshold = {threshold}")
            
            # Run evaluation
            metrics = self.run_single_evaluation(
                level1_threshold=threshold,
                level2_threshold=None,
                use_bayesian=False
            )
            
            # Extract key metrics
            result = {
                'method': f'Threshold ({threshold})',
                'threshold': threshold,
                'bayesian_temp': None,
                'accuracy': metrics['overall']['accuracy'],
                'balanced_accuracy': metrics['overall']['balanced_accuracy'],
                'f1_macro': metrics['overall']['f1_macro'],
                'level1_accuracy': metrics['level1']['accuracy'],
                'level1_balanced_accuracy': metrics['level1']['balanced_accuracy'],
            }
            
            # Calculate false negative rates
            if 'confusion_matrix' in metrics:
                cm_overall = metrics['confusion_matrix']
                
                # Calculate level 1 false negative rate (0 vs 1+)
                # Create a 2x2 binary confusion matrix for level 1
                level1_cm = np.zeros((2, 2))
                # True 0, Predicted 0
                level1_cm[0, 0] = cm_overall[0, 0]
                # True 0, Predicted 1+
                level1_cm[0, 1] = cm_overall[0, 1] + cm_overall[0, 2]
                # True 1+, Predicted 0
                level1_cm[1, 0] = cm_overall[1, 0] + cm_overall[2, 0]
                # True 1+, Predicted 1+
                level1_cm[1, 1] = cm_overall[1, 1] + cm_overall[1, 2] + cm_overall[2, 1] + cm_overall[2, 2]
                
                result['level1_fn_rate'] = self.calculate_fn_rate(level1_cm)
                result['level1_fp_rate'] = self.calculate_fp_rate(level1_cm)
            
            # Add result
            results.append(result)
        
        # Then, run Bayesian correction
        for temp in bayesian_values:
            print(f"\nEvaluating with Bayesian temperature = {temp:.4f}")
            
            # Run evaluation
            metrics = self.run_single_evaluation(
                use_bayesian=True,
                bayesian_temp=temp
            )
            
            # Extract key metrics
            result = {
                'method': f'Bayesian ({temp:.3f})',
                'threshold': None,
                'bayesian_temp': temp,
                'accuracy': metrics['overall']['accuracy'],
                'balanced_accuracy': metrics['overall']['balanced_accuracy'],
                'f1_macro': metrics['overall']['f1_macro'],
                'level1_accuracy': metrics['level1']['accuracy'],
                'level1_balanced_accuracy': metrics['level1']['balanced_accuracy'],
            }
            
            # Calculate false negative rates
            if 'confusion_matrix' in metrics:
                cm_overall = metrics['confusion_matrix']
                
                # Calculate level 1 false negative rate (0 vs 1+)
                # Create a 2x2 binary confusion matrix for level 1
                level1_cm = np.zeros((2, 2))
                # True 0, Predicted 0
                level1_cm[0, 0] = cm_overall[0, 0]
                # True 0, Predicted 1+
                level1_cm[0, 1] = cm_overall[0, 1] + cm_overall[0, 2]
                # True 1+, Predicted 0
                level1_cm[1, 0] = cm_overall[1, 0] + cm_overall[2, 0]
                # True 1+, Predicted 1+
                level1_cm[1, 1] = cm_overall[1, 1] + cm_overall[1, 2] + cm_overall[2, 1] + cm_overall[2, 2]
                
                result['level1_fn_rate'] = self.calculate_fn_rate(level1_cm)
                result['level1_fp_rate'] = self.calculate_fp_rate(level1_cm)
            
            # Add result
            results.append(result)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'threshold_vs_bayesian_comparison.csv', index=False)
        
        # Plot comparison - Accuracy and F1
        plt.figure(figsize=(12, 6))
        
        # Filter DataFrames for each method
        threshold_df = results_df[results_df['threshold'].notna()]
        bayesian_df = results_df[results_df['bayesian_temp'].notna()]
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        sns.barplot(x='method', y='accuracy', data=results_df)
        plt.title('Accuracy Comparison')
        plt.xlabel('')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0.7, 1.0)  # Adjust as needed
        
        # Plot F1
        plt.subplot(1, 2, 2)
        sns.barplot(x='method', y='f1_macro', data=results_df)
        plt.title('F1 Score Comparison')
        plt.xlabel('')
        plt.ylabel('F1 Score (Macro)')
        plt.xticks(rotation=45)
        plt.ylim(0.7, 1.0)  # Adjust as needed
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_vs_bayesian_metrics.png')
        plt.close()
        
        # Plot FN/FP rates
        if 'level1_fn_rate' in results_df.columns and 'level1_fp_rate' in results_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Plot FN rate
            plt.subplot(1, 2, 1)
            sns.barplot(x='method', y='level1_fn_rate', data=results_df)
            plt.title('False Negative Rate Comparison (Level 1)')
            plt.xlabel('')
            plt.ylabel('False Negative Rate')
            plt.xticks(rotation=45)
            plt.ylim(0, 0.3)  # Adjust as needed
            
            # Plot FP rate
            plt.subplot(1, 2, 2)
            sns.barplot(x='method', y='level1_fp_rate', data=results_df)
            plt.title('False Positive Rate Comparison (Level 1)')
            plt.xlabel('')
            plt.ylabel('False Positive Rate')
            plt.xticks(rotation=45)
            plt.ylim(0, 0.3)  # Adjust as needed
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'threshold_vs_bayesian_error_rates.png')
            plt.close()
            
            # Plot FN vs FP scatter
            plt.figure(figsize=(10, 8))
            plt.scatter(
                threshold_df['level1_fp_rate'], 
                threshold_df['level1_fn_rate'], 
                s=100, 
                marker='o', 
                label='Threshold Adjustment'
            )
            for i, row in threshold_df.iterrows():
                plt.annotate(
                    f"{row['threshold']}", 
                    (row['level1_fp_rate'], row['level1_fn_rate']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
                
            plt.scatter(
                bayesian_df['level1_fp_rate'], 
                bayesian_df['level1_fn_rate'], 
                s=100, 
                marker='^', 
                label='Bayesian Correction'
            )
            for i, row in bayesian_df.iterrows():
                plt.annotate(
                    f"{row['bayesian_temp']:.3f}", 
                    (row['level1_fp_rate'], row['level1_fn_rate']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
            
            plt.title('False Positive Rate vs False Negative Rate (Level 1)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('False Negative Rate')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Draw optimal point (0,0)
            plt.plot(0, 0, 'rx', markersize=10)
            plt.annotate('Optimal', (0, 0), textcoords="offset points", xytext=(10, 10), color='red')
            
            plt.savefig(self.output_dir / 'threshold_vs_bayesian_fn_fp_scatter.png')
            plt.close()
        
        print(f"\nComparison complete! Results saved to {self.output_dir}")
        return results_df
    
    def _plot_threshold_heatmap(self, df, metric, title, cmap='viridis'):
        """
        Plot a heatmap of a metric across different threshold values.
        
        Args:
            df: DataFrame with results
            metric: Column name of the metric to plot
            title: Title for the plot
            cmap: Colormap to use
        """
        # Pivot the data for heatmap
        pivot_df = df.pivot(
            index='level1_threshold', 
            columns='level2_threshold', 
            values=metric
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.3f', 
            cmap=cmap,
            vmin=pivot_df.min().min() if 'rate' in metric else None,
            vmax=pivot_df.max().max() if 'rate' in metric else None
        )
        plt.title(f'{title} vs Classification Thresholds')
        plt.xlabel('Level 2 Threshold')
        plt.ylabel('Level 1 Threshold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'threshold_heatmap_{metric}.png')
        plt.close()
    
    def _plot_bayesian_metrics(self, df):
        """
        Plot metrics across different Bayesian temperature values.
        
        Args:
            df: DataFrame with results
        """
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy metrics
        plt.subplot(2, 1, 1)
        plt.plot(df['bayesian_temp'], df['accuracy'], 'o-', label='Accuracy')
        plt.plot(df['bayesian_temp'], df['balanced_accuracy'], 's-', label='Balanced Accuracy')
        plt.plot(df['bayesian_temp'], df['f1_macro'], '^-', label='F1 Score')
        
        plt.title('Metrics vs Bayesian Temperature')
        plt.xlabel('Bayesian Temperature')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot FN/FP rates if available
        if 'level1_fn_rate' in df.columns and 'level1_fp_rate' in df.columns:
            plt.subplot(2, 1, 2)
            plt.plot(df['bayesian_temp'], df['level1_fn_rate'], 'o-', label='Level 1 FN Rate')
            plt.plot(df['bayesian_temp'], df['level1_fp_rate'], 's-', label='Level 1 FP Rate')
            
            if 'level2_fn_rate' in df.columns and 'level2_fp_rate' in df.columns:
                plt.plot(df['bayesian_temp'], df['level2_fn_rate'], '^--', label='Level 2 FN Rate')
                plt.plot(df['bayesian_temp'], df['level2_fp_rate'], 'v--', label='Level 2 FP Rate')
            
            plt.xlabel('Bayesian Temperature')
            plt.ylabel('Error Rate')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bayesian_metrics.png')
        plt.close()
        
        # If enough points, also plot log scale
        if len(df) >= 3 and df['bayesian_temp'].max() / df['bayesian_temp'].min() > 5:
            plt.figure(figsize=(12, 8))
            
            # Plot accuracy metrics
            plt.subplot(2, 1, 1)
            plt.semilogx(df['bayesian_temp'], df['accuracy'], 'o-', label='Accuracy')
            plt.semilogx(df['bayesian_temp'], df['balanced_accuracy'], 's-', label='Balanced Accuracy')
            plt.semilogx(df['bayesian_temp'], df['f1_macro'], '^-', label='F1 Score')
            
            plt.title('Metrics vs Bayesian Temperature (Log Scale)')
            plt.xlabel('Bayesian Temperature (log scale)')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Plot FN/FP rates if available
            if 'level1_fn_rate' in df.columns and 'level1_fp_rate' in df.columns:
                plt.subplot(2, 1, 2)
                plt.semilogx(df['bayesian_temp'], df['level1_fn_rate'], 'o-', label='Level 1 FN Rate')
                plt.semilogx(df['bayesian_temp'], df['level1_fp_rate'], 's-', label='Level 1 FP Rate')
                
                if 'level2_fn_rate' in df.columns and 'level2_fp_rate' in df.columns:
                    plt.semilogx(df['bayesian_temp'], df['level2_fn_rate'], '^--', label='Level 2 FN Rate')
                    plt.semilogx(df['bayesian_temp'], df['level2_fp_rate'], 'v--', label='Level 2 FP Rate')
                
                plt.xlabel('Bayesian Temperature (log scale)')
                plt.ylabel('Error Rate')
                plt.legend()
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'bayesian_metrics_log_scale.png')
            plt.close()
    
    def _plot_fn_fp_tradeoff(self, df, level=1):
        """
        Plot false negative vs false positive rate to visualize tradeoffs.
        
        Args:
            df: DataFrame with results
            level: Level to plot (1 or 2)
        """
        fn_col = f'level{level}_fn_rate'
        fp_col = f'level{level}_fp_rate'
        
        if fn_col not in df.columns or fp_col not in df.columns:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with color representing threshold
        scatter = plt.scatter(
            df[fp_col],
            df[fn_col],
            c=df[f'level{level}_threshold'],
            cmap='viridis',
            s=100,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'Level {level} Threshold')
        
        # Annotate points with threshold values
        for i, row in df.iterrows():
            plt.annotate(
                f"{row[f'level{level}_threshold']}", 
                (row[fp_col], row[fn_col]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        # Add labels and title
        plt.title(f'FN vs FP Rate Tradeoff (Level {level})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.grid(alpha=0.3)
        
        # Draw optimal point (0,0)
        plt.plot(0, 0, 'rx', markersize=10)
        plt.annotate('Optimal', (0, 0), textcoords="offset points", xytext=(10, 10), color='red')
        
        # Save the plot
        plt.savefig(self.output_dir / f'fn_fp_tradeoff_level{level}.png')
        plt.close()


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test Bayesian calibration and threshold adjustments")
    
    # Model and data options
    parser.add_argument(
        "--model_base_path", "-m",
        default="models/bayesian_hierarchical",
        help="Base path to hierarchical models"
    )
    parser.add_argument(
        "--model_type",
        choices=["vit", "swin"],
        default="swin",
        help="Type of transformer model (vit or swin)"
    )
    parser.add_argument(
        "--test_csv",
        default="rectangle_dataset/test.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--test_dir",
        default="rectangle_dataset/test",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="evaluation/sweep",
        help="Directory to save results"
    )
    
    # Testing mode options
    parser.add_argument(
        "--mode",
        choices=["threshold_sweep", "bayesian_sweep", "compare", "all"],
        default="all",
        help="Testing mode"
    )
    
    # Threshold sweep options
    parser.add_argument(
        "--level1_thresholds",
        default="0.5,0.4,0.3,0.2",
        help="Comma-separated Level 1 threshold values"
    )
    parser.add_argument(
        "--level2_thresholds",
        default="0.5,0.4,0.3,0.2",
        help="Comma-separated Level 2 threshold values"
    )
    
    # Bayesian sweep options
    parser.add_argument(
        "--bayesian_temps",
        default="0.0,0.01,0.05,0.1",
        help="Comma-separated Bayesian temperature values"
    )
    
    # Processing options
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Disable image enhancement before processing"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the tests.
    """
    args = parse_arguments()
    
    # Parse threshold and temperature values
    level1_thresholds = [float(t) for t in args.level1_thresholds.split(',')]
    level2_thresholds = [float(t) for t in args.level2_thresholds.split(',')]
    bayesian_temps = [float(t) for t in args.bayesian_temps.split(',')]
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the sweeper
    sweeper = ThresholdSweeper(
        model_base_path=args.model_base_path,
        test_csv=args.test_csv,
        test_dir=args.test_dir,
        output_dir=output_path,
        model_type=args.model_type,
        level1_threshold_values=level1_thresholds,
        level2_threshold_values=level2_thresholds,
        bayesian_temp_values=bayesian_temps,
        enhance_images=not args.no_enhance
    )
    
    # Run the requested tests
    if args.mode == "threshold_sweep" or args.mode == "all":
        sweeper.run_threshold_sweep()
    
    if args.mode == "bayesian_sweep" or args.mode == "all":
        sweeper.run_bayesian_sweep()
    
    if args.mode == "compare" or args.mode == "all":
        sweeper.compare_threshold_vs_bayesian()
    
    print("\nAll tests completed!")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()