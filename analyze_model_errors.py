#!/usr/bin/env python3
"""
Script to analyze false positives and false negatives from a trained hierarchical model.

This script evaluates a trained model on a test set and saves misclassified samples
to separate directories for analysis.
"""

import argparse
import os
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

try:
    import cv2
except ImportError:
    print("Warning: OpenCV (cv2) not found. Image annotation will be disabled.")
    cv2 = None

from model_factory import ModelFactory
from device_utils import get_device
from datasets import ReceiptDataset
from hierarchical_predictor import HierarchicalPredictor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze model errors (false positives and false negatives)"
    )
    
    # Data arguments
    parser.add_argument(
        "--test_csv", 
        required=True, 
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--test_dir", 
        required=True, 
        help="Directory containing test images"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_base_path", 
        required=True,
        help="Base path to the hierarchical model directory"
    )
    parser.add_argument(
        "--model_type", 
        choices=["vit", "swin"],
        default="vit",
        help="Type of model (vit or swin)"
    )
    
    # Analysis arguments
    parser.add_argument(
        "--output_dir", 
        default="error_analysis",
        help="Directory to save error analysis results"
    )
    # Removed batch_size parameter as HierarchicalPredictor doesn't accept it
    parser.add_argument(
        "--annotate", 
        action="store_true",
        help="Annotate images with correct label and prediction"
    )
    parser.add_argument(
        "--use_calibrated", 
        action="store_true",
        help="Use temperature-calibrated models if available"
    )
    
    return parser.parse_args()


def analyze_errors(predictor, test_df, test_dir, output_dir, annotate=False):
    """
    Analyze errors (false positives and false negatives) in the test set.
    
    Args:
        predictor: HierarchicalPredictor instance
        test_df: Test dataframe with filenames and receipt counts
        test_dir: Directory containing test images
        output_dir: Directory to save error analysis results
        annotate: Whether to annotate images with correct label and prediction
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories for error types
    error_dirs = {
        "level1_false_negatives": output_dir / "level1_false_negatives",
        "level1_false_positives": output_dir / "level1_false_positives",
        "level2_false_negatives": output_dir / "level2_false_negatives",
        "level2_false_positives": output_dir / "level2_false_positives",
        "overall_false_negatives": output_dir / "overall_false_negatives",
        "overall_false_positives": output_dir / "overall_false_positives",
    }
    
    for dir_path in error_dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Initialize counters and data storage
    error_counts = {key: 0 for key in error_dirs.keys()}
    all_results = []
    
    # Process each image in the test set
    print(f"Processing {len(test_df)} test images...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row["filename"]
        true_count = row["receipt_count"]
        
        # Try to find the image path
        img_path = None
        possible_paths = [
            os.path.join(test_dir, filename),
            os.path.join(os.path.dirname(test_dir), "test", filename),
            os.path.join(os.path.dirname(test_dir), filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            print(f"Warning: Could not find image {filename} for analysis")
            continue
        
        # Run prediction
        prediction_result = {}
        try:
            # The method is actually called 'predict', not 'predict_single_image'
            prediction, confidence, confidences = predictor.predict(img_path, enhance=False, return_confidences=True)
            
            # Reconstruct the expected prediction_result format
            prediction_result = {
                "predicted_count": prediction,
                "confidence": confidence,
                "level1_prediction": 1 if prediction > 0 else 0,
                "level1_confidence": confidences['level1']['has_receipts'] if prediction > 0 else confidences['level1']['no_receipts']
            }
            
            # Add level2 information if available
            if prediction > 0 and 'level2' in confidences:
                prediction_result["level2_prediction"] = 0 if prediction == 1 else 1
                prediction_result["level2_confidence"] = confidences['level2']['one_receipt'] if prediction == 1 else confidences['level2']['multiple_receipts']
        except Exception as e:
            print(f"Error predicting image {img_path}: {e}")
            # Default values if prediction fails
            prediction_result = {
                "predicted_count": 0,
                "confidence": 0.0,
                "level1_prediction": 0,
                "level1_confidence": 0.0
            }
        
        # Extract prediction and confidence
        predicted_count = prediction_result["predicted_count"]
        
        # Get level-specific predictions
        level1_pred = prediction_result["level1_prediction"]
        level1_true = 1 if true_count > 0 else 0
        level1_conf = prediction_result["level1_confidence"]
        
        # Check if this prediction involves level 2
        level2_pred = None
        level2_true = None
        level2_conf = None
        if level1_pred == 1:  # If level 1 predicted receipts exist
            level2_pred = prediction_result["level2_prediction"]
            level2_true = 1 if true_count > 1 else 0
            level2_conf = prediction_result["level2_confidence"]
        
        # Create result entry
        result = {
            "filename": filename,
            "image_path": img_path,
            "true_count": true_count,
            "predicted_count": predicted_count,
            "level1_true": level1_true,
            "level1_pred": level1_pred,
            "level1_conf": level1_conf,
        }
        
        if level2_pred is not None:
            result.update({
                "level2_true": level2_true,
                "level2_pred": level2_pred,
                "level2_conf": level2_conf,
            })
        
        all_results.append(result)
        
        # Analyze errors
        
        # Level 1 errors (receipt detection)
        if level1_pred == 0 and level1_true == 1:
            # False negative at level 1 (missed receipts)
            error_counts["level1_false_negatives"] += 1
            copy_and_annotate_image(
                img_path, 
                error_dirs["level1_false_negatives"],
                f"true_{true_count}_pred_0_conf_{level1_conf:.4f}_{filename}",
                annotation_text=f"True: {true_count}, Pred: 0, Conf: {level1_conf:.4f}",
                annotate=annotate
            )
        elif level1_pred == 1 and level1_true == 0:
            # False positive at level 1 (false alarm)
            error_counts["level1_false_positives"] += 1
            copy_and_annotate_image(
                img_path, 
                error_dirs["level1_false_positives"],
                f"true_0_pred_1+_conf_{level1_conf:.4f}_{filename}",
                annotation_text=f"True: 0, Pred: 1+, Conf: {level1_conf:.4f}",
                annotate=annotate
            )
        
        # Level 2 errors (only relevant if level 1 is correct and true label > 0)
        if level1_pred == 1 and level1_true == 1 and level2_pred is not None:
            if level2_pred == 0 and level2_true == 1:
                # False negative at level 2 (missed multiple receipts)
                error_counts["level2_false_negatives"] += 1
                copy_and_annotate_image(
                    img_path, 
                    error_dirs["level2_false_negatives"],
                    f"true_{true_count}_pred_1_conf_{level2_conf:.4f}_{filename}",
                    annotation_text=f"True: {true_count}, Pred: 1, Conf: {level2_conf:.4f}",
                    annotate=annotate
                )
            elif level2_pred == 1 and level2_true == 0:
                # False positive at level 2 (falsely detected multiple)
                error_counts["level2_false_positives"] += 1
                copy_and_annotate_image(
                    img_path, 
                    error_dirs["level2_false_positives"],
                    f"true_1_pred_2+_conf_{level2_conf:.4f}_{filename}",
                    annotation_text=f"True: 1, Pred: 2+, Conf: {level2_conf:.4f}",
                    annotate=annotate
                )
        
        # Overall errors (final prediction vs true count)
        if predicted_count == 0 and true_count > 0:
            # Overall false negative (missed receipts)
            error_counts["overall_false_negatives"] += 1
            copy_and_annotate_image(
                img_path, 
                error_dirs["overall_false_negatives"],
                f"true_{true_count}_pred_0_{filename}",
                annotation_text=f"True: {true_count}, Pred: 0",
                annotate=annotate
            )
        elif predicted_count > 0 and true_count == 0:
            # Overall false positive (false alarm)
            error_counts["overall_false_positives"] += 1
            copy_and_annotate_image(
                img_path, 
                error_dirs["overall_false_positives"],
                f"true_0_pred_{predicted_count}_{filename}",
                annotation_text=f"True: 0, Pred: {predicted_count}",
                annotate=annotate
            )
    
    # Calculate metrics
    level1_true = [r["level1_true"] for r in all_results]
    level1_pred = [r["level1_pred"] for r in all_results]
    
    # Filter for level 2 evaluation (only where level1_true == 1)
    level2_results = [r for r in all_results if "level2_true" in r and r["level1_true"] == 1]
    if level2_results:
        level2_true = [r["level2_true"] for r in level2_results]
        level2_pred = [r["level2_pred"] for r in level2_results]
    else:
        level2_true = []
        level2_pred = []
    
    # Calculate true/predicted counts for overall evaluation
    true_counts = [r["true_count"] for r in all_results]
    pred_counts = [r["predicted_count"] for r in all_results]
    
    # Convert to binary for overall error rate calculations
    binary_true = [1 if count > 0 else 0 for count in true_counts]
    binary_pred = [1 if count > 0 else 0 for count in pred_counts]
    
    # Compute confusion matrices
    level1_cm = confusion_matrix(level1_true, level1_pred, labels=[0, 1])
    overall_cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    
    level2_cm = None
    if level2_true and level2_pred:
        level2_cm = confusion_matrix(level2_true, level2_pred, labels=[0, 1])
    
    # Extract error rates
    level1_fn_rate = level1_cm[1, 0] / max(1, level1_cm[1, 0] + level1_cm[1, 1])
    level1_fp_rate = level1_cm[0, 1] / max(1, level1_cm[0, 0] + level1_cm[0, 1])
    
    overall_fn_rate = overall_cm[1, 0] / max(1, overall_cm[1, 0] + overall_cm[1, 1])
    overall_fp_rate = overall_cm[0, 1] / max(1, overall_cm[0, 0] + overall_cm[0, 1])
    
    level2_fn_rate = None
    level2_fp_rate = None
    if level2_cm is not None:
        level2_fn_rate = level2_cm[1, 0] / max(1, level2_cm[1, 0] + level2_cm[1, 1])
        level2_fp_rate = level2_cm[0, 1] / max(1, level2_cm[0, 0] + level2_cm[0, 1])
    
    # Compute accuracy metrics
    level1_accuracy = accuracy_score(level1_true, level1_pred)
    level1_balanced_acc = balanced_accuracy_score(level1_true, level1_pred)
    level1_f1 = f1_score(level1_true, level1_pred, average='macro')
    
    overall_accuracy = accuracy_score(binary_true, binary_pred)
    overall_balanced_acc = balanced_accuracy_score(binary_true, binary_pred)
    overall_f1 = f1_score(binary_true, binary_pred, average='macro')
    
    level2_metrics = {}
    if level2_true and level2_pred:
        level2_accuracy = accuracy_score(level2_true, level2_pred)
        level2_balanced_acc = balanced_accuracy_score(level2_true, level2_pred)
        level2_f1 = f1_score(level2_true, level2_pred, average='macro')
        level2_metrics = {
            "level2_accuracy": level2_accuracy,
            "level2_balanced_accuracy": level2_balanced_acc,
            "level2_f1": level2_f1,
        }
    
    # Save results
    results = {
        "error_counts": error_counts,
        "level1_confusion_matrix": level1_cm.tolist(),
        "level1_false_negative_rate": level1_fn_rate,
        "level1_false_positive_rate": level1_fp_rate,
        "level1_accuracy": level1_accuracy,
        "level1_balanced_accuracy": level1_balanced_acc,
        "level1_f1": level1_f1,
        "overall_confusion_matrix": overall_cm.tolist(),
        "overall_false_negative_rate": overall_fn_rate,
        "overall_false_positive_rate": overall_fp_rate,
        "overall_accuracy": overall_accuracy,
        "overall_balanced_accuracy": overall_balanced_acc,
        "overall_f1": overall_f1,
    }
    
    if level2_cm is not None:
        results["level2_confusion_matrix"] = level2_cm.tolist()
        results["level2_false_negative_rate"] = level2_fn_rate
        results["level2_false_positive_rate"] = level2_fp_rate
        results.update(level2_metrics)
    
    # Save summary to JSON file
    with open(output_dir / "error_analysis_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "detailed_predictions.csv", index=False)
    
    # Try to load metadata for detailed report
    try:
        metadata_path = Path(args.model_base_path) / "training_metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    except Exception:
        metadata = None
    
    # Generate and save summary report with metadata if available
    generate_summary_report(results, output_dir, metadata)
    
    # Print summary with explanations
    print("\nError Analysis Summary:")
    print(f"Total test images: {len(test_df)}")
    
    # Level 1 explanations
    print(f"Level 1 false negatives: {error_counts['level1_false_negatives']} (rate: {level1_fn_rate:.4f})")
    if error_counts['level1_false_negatives'] == 0:
        print("  - The model correctly identifies all images that have receipts")
    else:
        print(f"  - The model fails to detect receipts in {error_counts['level1_false_negatives']} images")
    
    print(f"Level 1 false positives: {error_counts['level1_false_positives']} (rate: {level1_fp_rate:.4f})")
    if error_counts['level1_false_positives'] == 0:
        print("  - The model doesn't falsely detect receipts in images without them")
    else:
        print(f"  - The model falsely detects receipts in {error_counts['level1_false_positives']} empty images")
    
    # Level 2 explanations if available
    if level2_fn_rate is not None:
        print(f"Level 2 false negatives: {error_counts['level2_false_negatives']} (rate: {level2_fn_rate:.4f})")
        if error_counts['level2_false_negatives'] == 0:
            print("  - The model doesn't miss multiple receipts")
        else:
            print(f"  - The model misses multiple receipts in {error_counts['level2_false_negatives']} images, classifying them as single receipts")
        
        print(f"Level 2 false positives: {error_counts['level2_false_positives']} (rate: {level2_fp_rate:.4f})")
        if error_counts['level2_false_positives'] == 0:
            print("  - The model correctly distinguishes between single and multiple receipts")
        else:
            print(f"  - The model sometimes predicts multiple receipts when there's only one ({error_counts['level2_false_positives']} instances)")
    
    # Overall explanations
    print(f"Overall false negatives: {error_counts['overall_false_negatives']} (rate: {overall_fn_rate:.4f})")
    if error_counts['overall_false_negatives'] == 0:
        print("  - The hierarchical system never completely misses receipts")
    else:
        print(f"  - The hierarchical system completely fails to detect receipts in {error_counts['overall_false_negatives']} images")
    
    print(f"Overall false positives: {error_counts['overall_false_positives']} (rate: {overall_fp_rate:.4f})")
    if error_counts['overall_false_positives'] == 0:
        print("  - The hierarchical system doesn't falsely detect receipts in empty images")
    else:
        print(f"  - The hierarchical system falsely detects receipts in {error_counts['overall_false_positives']} empty images")
    
    print(f"\nDetailed report saved to: {output_dir / 'error_analysis_report.txt'}")
    
    return results


def copy_and_annotate_image(src_path, dest_dir, dest_filename, annotation_text=None, annotate=False):
    """
    Copy image to destination directory, optionally with annotation.
    
    Args:
        src_path: Source image path
        dest_dir: Destination directory
        dest_filename: Destination filename
        annotation_text: Text to annotate on the image
        annotate: Whether to annotate the image
    """
    if not annotate or cv2 is None:
        # Simple copy without annotation
        shutil.copy2(src_path, os.path.join(dest_dir, dest_filename))
        return
    
    try:
        # Open image for annotation
        img = Image.open(src_path)
        
        # Need to convert to RGB if it's not already
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy array for OpenCV
        img_np = np.array(img)
        
        # Add annotation text
        cv2.putText(
            img_np,
            annotation_text,
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 0, 0),  # Color (BGR)
            2,  # Thickness
            cv2.LINE_AA  # Line type
        )
        
        # Convert back to PIL and save
        img_annotated = Image.fromarray(img_np)
        img_annotated.save(os.path.join(dest_dir, dest_filename))
        
    except Exception as e:
        print(f"Error annotating image {src_path}: {e}")
        # Fall back to simple copy if annotation fails
        shutil.copy2(src_path, os.path.join(dest_dir, dest_filename))


def generate_summary_report(results, output_dir, metadata=None):
    """
    Generate a text report summarizing the error analysis.
    
    Args:
        results: Dictionary containing error analysis results
        output_dir: Directory to save the report
        metadata: Optional model metadata for more detailed report
    """
    output_path = output_dir / "error_analysis_report.txt"
    error_counts = results["error_counts"]
    
    with open(output_path, "w") as f:
        f.write("=============================================\n")
        f.write("       ERROR ANALYSIS SUMMARY REPORT        \n")
        f.write("=============================================\n\n")
        
        # Include model metadata if available
        if metadata:
            f.write("MODEL INFORMATION:\n")
            f.write("-----------------\n")
            
            # Model architecture
            model_type = metadata.get('training_params', {}).get('model_type', 'unknown')
            f.write(f"Architecture: Hierarchical {model_type.upper()} transformer\n\n")
            
            # Training parameters
            if 'training_params' in metadata:
                f.write("Training Hyperparameters:\n")
                for param, value in metadata['training_params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
            
            # Class weights
            if 'level1' in metadata and 'class_weights' in metadata['level1']:
                f.write("Level 1 Class Weights:\n")
                f.write(f"  Class 0 (no receipts): {metadata['level1']['class_weights'][0]}\n")
                f.write(f"  Class 1 (has receipts): {metadata['level1']['class_weights'][1]}\n\n")
            
            if 'level2' in metadata and 'class_weights' in metadata['level2']:
                f.write("Level 2 Class Weights:\n")
                f.write(f"  Class 0 (one receipt): {metadata['level2']['class_weights'][0]}\n")
                f.write(f"  Class 1 (multiple receipts): {metadata['level2']['class_weights'][1]}\n\n")
            
            # Class frequencies
            if 'level1' in metadata and 'class_frequencies' in metadata['level1']:
                f.write("Level 1 Class Frequencies (training set):\n")
                f.write(f"  Class 0 (no receipts): {metadata['level1']['class_frequencies'][0]:.4f}\n")
                f.write(f"  Class 1 (has receipts): {metadata['level1']['class_frequencies'][1]:.4f}\n\n")
            
            if 'level2' in metadata and 'class_frequencies' in metadata['level2']:
                f.write("Level 2 Class Frequencies (training set):\n")
                f.write(f"  Class 0 (one receipt): {metadata['level2']['class_frequencies'][0]:.4f}\n")
                f.write(f"  Class 1 (multiple receipts): {metadata['level2']['class_frequencies'][1]:.4f}\n\n")
            
            f.write("---------------------------------------------\n\n")
        
        # Error counts with explanations
        f.write("ERROR COUNTS AND EXPLANATIONS:\n")
        f.write("----------------------------\n")
        
        # Level 1 counts and explanations
        f.write(f"Level 1 false negatives: {error_counts['level1_false_negatives']} (rate: {results['level1_false_negative_rate']:.4f})\n")
        if error_counts['level1_false_negatives'] == 0:
            f.write("  - The model correctly identifies all images that have receipts\n")
        else:
            f.write(f"  - The model fails to detect receipts in {error_counts['level1_false_negatives']} images\n")
        
        f.write(f"Level 1 false positives: {error_counts['level1_false_positives']} (rate: {results['level1_false_positive_rate']:.4f})\n")
        if error_counts['level1_false_positives'] == 0:
            f.write("  - The model doesn't falsely detect receipts in images without them\n")
        else:
            f.write(f"  - The model falsely detects receipts in {error_counts['level1_false_positives']} empty images\n")
        
        # Level 2 counts and explanations if available
        if "level2_false_negative_rate" in results:
            f.write(f"\nLevel 2 false negatives: {error_counts['level2_false_negatives']} (rate: {results['level2_false_negative_rate']:.4f})\n")
            if error_counts['level2_false_negatives'] == 0:
                f.write("  - The model doesn't miss multiple receipts\n")
            else:
                f.write(f"  - The model misses multiple receipts in {error_counts['level2_false_negatives']} images, classifying them as single receipts\n")
            
            f.write(f"Level 2 false positives: {error_counts['level2_false_positives']} (rate: {results['level2_false_positive_rate']:.4f})\n")
            if error_counts['level2_false_positives'] == 0:
                f.write("  - The model correctly distinguishes between single and multiple receipts\n")
            else:
                f.write(f"  - The model sometimes predicts multiple receipts when there's only one ({error_counts['level2_false_positives']} instances)\n")
        
        # Overall counts and explanations
        f.write(f"\nOverall false negatives: {error_counts['overall_false_negatives']} (rate: {results['overall_false_negative_rate']:.4f})\n")
        if error_counts['overall_false_negatives'] == 0:
            f.write("  - The hierarchical system never completely misses receipts\n")
        else:
            f.write(f"  - The hierarchical system completely fails to detect receipts in {error_counts['overall_false_negatives']} images\n")
        
        f.write(f"Overall false positives: {error_counts['overall_false_positives']} (rate: {results['overall_false_positive_rate']:.4f})\n")
        if error_counts['overall_false_positives'] == 0:
            f.write("  - The hierarchical system doesn't falsely detect receipts in empty images\n")
        else:
            f.write(f"  - The hierarchical system falsely detects receipts in {error_counts['overall_false_positives']} empty images\n")
        
        f.write("\n")
        
        # Level 1 metrics
        f.write("LEVEL 1 METRICS (0 vs 1+ receipts):\n")
        f.write("----------------------------------\n")
        f.write(f"Accuracy: {results['level1_accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['level1_balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score (macro): {results['level1_f1']:.4f}\n")
        f.write(f"False Negative Rate: {results['level1_false_negative_rate']:.4f}\n")
        f.write(f"False Positive Rate: {results['level1_false_positive_rate']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        cm = np.array(results["level1_confusion_matrix"])
        f.write(f"    Pred 0  Pred 1+\n")
        f.write(f"True 0  {cm[0, 0]:6d}  {cm[0, 1]:6d}\n")
        f.write(f"True 1+ {cm[1, 0]:6d}  {cm[1, 1]:6d}\n\n")
        
        # Level 2 metrics if available
        if "level2_confusion_matrix" in results:
            f.write("LEVEL 2 METRICS (1 vs 2+ receipts):\n")
            f.write("----------------------------------\n")
            f.write(f"Accuracy: {results['level2_accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {results['level2_balanced_accuracy']:.4f}\n")
            f.write(f"F1 Score (macro): {results['level2_f1']:.4f}\n")
            f.write(f"False Negative Rate: {results['level2_false_negative_rate']:.4f}\n")
            f.write(f"False Positive Rate: {results['level2_false_positive_rate']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            cm = np.array(results["level2_confusion_matrix"])
            f.write(f"    Pred 1  Pred 2+\n")
            f.write(f"True 1  {cm[0, 0]:6d}  {cm[0, 1]:6d}\n")
            f.write(f"True 2+ {cm[1, 0]:6d}  {cm[1, 1]:6d}\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS (binary receipt detection):\n")
        f.write("----------------------------------------\n")
        f.write(f"Accuracy: {results['overall_accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['overall_balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score (macro): {results['overall_f1']:.4f}\n")
        f.write(f"False Negative Rate: {results['overall_false_negative_rate']:.4f}\n")
        f.write(f"False Positive Rate: {results['overall_false_positive_rate']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        cm = np.array(results["overall_confusion_matrix"])
        f.write(f"    Pred 0  Pred 1+\n")
        f.write(f"True 0  {cm[0, 0]:6d}  {cm[0, 1]:6d}\n")
        f.write(f"True 1+ {cm[1, 0]:6d}  {cm[1, 1]:6d}\n\n")
        
        # Final summary with recommendations
        f.write("======== SUMMARY AND RECOMMENDATIONS ========\n")
        
        # Identify the biggest issue and provide recommendations
        level1_fn = results["level1_false_negative_rate"]
        level1_fp = results["level1_false_positive_rate"]
        
        if "level2_false_negative_rate" in results:
            level2_fn = results["level2_false_negative_rate"]
            level2_fp = results["level2_false_positive_rate"]
            max_rate = max(level1_fn, level1_fp, level2_fn, level2_fp)
            
            if max_rate == level1_fn:
                f.write("Primary issue: Level 1 false negatives (failing to detect receipts)\n")
                f.write("Recommendation: Adjust decision threshold to reduce false negatives or retrain level 1 model with additional data\n")
            elif max_rate == level1_fp:
                f.write("Primary issue: Level 1 false positives (falsely detecting receipts)\n")
                f.write("Recommendation: Adjust decision threshold or add more hard negative examples to the training set\n")
            elif max_rate == level2_fn:
                f.write("Primary issue: Level 2 false negatives (detecting single receipt when multiple exist)\n")
                f.write("Recommendation: Adjust level 2 threshold or add more examples with multiple receipts to the training set\n")
            else:
                f.write("Primary issue: Level 2 false positives (detecting multiple receipts when only one exists)\n")
                f.write("Recommendation: Improve level 2 model discrimination by adding more diverse single receipt examples\n")
        else:
            max_rate = max(level1_fn, level1_fp)
            if max_rate == level1_fn:
                f.write("Primary issue: Level 1 false negatives (failing to detect receipts)\n")
                f.write("Recommendation: Lower the decision threshold to capture more positive cases\n")
            else:
                f.write("Primary issue: Level 1 false positives (falsely detecting receipts)\n")
                f.write("Recommendation: Raise the decision threshold or add more hard negative examples to the training\n")


def main():
    """Main function to run error analysis."""
    args = parse_arguments()
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    
    # Check if hierarchical model directory exists
    model_base_path = Path(args.model_base_path)
    if not model_base_path.exists():
        print(f"Error: Model base path {model_base_path} does not exist.")
        return
    
    model_type = args.model_type
    
    # Let's look at the actual directory structure to find model files
    print(f"Looking for models in {model_base_path}...")
    potential_level1_paths = list(model_base_path.glob(f"**/level1/**/*{model_type}*best*.pth"))
    potential_level2_paths = list(model_base_path.glob(f"**/level2/**/*{model_type}*best*.pth"))
    
    # Also check without level subdirectories
    if not potential_level1_paths:
        potential_level1_paths = list(model_base_path.glob(f"**/*level1*{model_type}*best*.pth"))
    if not potential_level2_paths:
        potential_level2_paths = list(model_base_path.glob(f"**/*level2*{model_type}*best*.pth"))
    
    # Also check for calibrated versions if requested
    if args.use_calibrated:
        calibrated_level1_paths = list(model_base_path.glob(f"**/calibrated/**/level1/**/*{model_type}*calibrated*.pth"))
        calibrated_level2_paths = list(model_base_path.glob(f"**/calibrated/**/level2/**/*{model_type}*calibrated*.pth"))
        
        if calibrated_level1_paths:
            potential_level1_paths = calibrated_level1_paths + potential_level1_paths
        if calibrated_level2_paths:
            potential_level2_paths = calibrated_level2_paths + potential_level2_paths
    
    if not potential_level1_paths:
        print(f"Error: Could not find any Level 1 {model_type} model files.")
        return
    
    level1_model_path = potential_level1_paths[0]
    level2_model_path = potential_level2_paths[0] if potential_level2_paths else None
    
    print(f"Found Level 1 model: {level1_model_path}")
    if level2_model_path:
        print(f"Found Level 2 model: {level2_model_path}")
    else:
        print(f"Warning: No Level 2 model found. Will use Level 1 model only.")
    
    # Create predictor
    predictor = HierarchicalPredictor(
        level1_model_path=str(level1_model_path),
        level2_model_path=str(level2_model_path) if level2_model_path else None,
        model_type=args.model_type,
    )
    
    # Extract and report model parameters
    print("\nModel Details:")
    print(f"Model architecture: Hierarchical {args.model_type.upper()}")
    
    # Try to load training metadata if it exists
    try:
        metadata_path = model_base_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                training_metadata = json.load(f)
                
            # Display training hyperparameters
            if 'training_params' in training_metadata:
                params = training_metadata['training_params']
                print("\nTraining Hyperparameters:")
                for param, value in params.items():
                    print(f"  {param}: {value}")
                    
            # Display class weights information
            if 'level1' in training_metadata and 'class_weights' in training_metadata['level1']:
                print("\nLevel 1 Class Weights:")
                print(f"  Class 0 (no receipts): {training_metadata['level1']['class_weights'][0]}")
                print(f"  Class 1 (has receipts): {training_metadata['level1']['class_weights'][1]}")
                
            if 'level2' in training_metadata and 'class_weights' in training_metadata['level2']:
                print("\nLevel 2 Class Weights:")
                print(f"  Class 0 (one receipt): {training_metadata['level2']['class_weights'][0]}")
                print(f"  Class 1 (multiple receipts): {training_metadata['level2']['class_weights'][1]}")
                
            # Display class frequencies information
            if 'level1' in training_metadata and 'class_frequencies' in training_metadata['level1']:
                print("\nLevel 1 Class Frequencies (training set):")
                print(f"  Class 0 (no receipts): {training_metadata['level1']['class_frequencies'][0]:.4f}")
                print(f"  Class 1 (has receipts): {training_metadata['level1']['class_frequencies'][1]:.4f}")
                
            if 'level2' in training_metadata and 'class_frequencies' in training_metadata['level2']:
                print("\nLevel 2 Class Frequencies (training set):")
                print(f"  Class 0 (one receipt): {training_metadata['level2']['class_frequencies'][0]:.4f}")
                print(f"  Class 1 (multiple receipts): {training_metadata['level2']['class_frequencies'][1]:.4f}")
                
        else:
            # If no metadata file, try to extract basic model information
            print("\nModel parameters:")
            level1_params = sum(p.numel() for p in predictor.level1_model.parameters())
            print(f"  Level 1 parameters: {level1_params:,}")
            
            if predictor.level2_model is not None:
                level2_params = sum(p.numel() for p in predictor.level2_model.parameters())
                print(f"  Level 2 parameters: {level2_params:,}")
                print(f"  Total parameters: {level1_params + level2_params:,}")
            
    except Exception as e:
        print(f"Note: Could not extract detailed model information: {e}")
    
    print("\nEvaluation Info:")
    print(f"  Test set: {args.test_csv}")
    print(f"  Number of test samples: {len(test_df)}")
    print(f"  Output directory: {args.output_dir}")
    if args.use_calibrated:
        print(f"  Using temperature calibrated models: {'Yes' if args.use_calibrated else 'No'}")
    print()
    
    # Run error analysis
    analyze_errors(
        predictor,
        test_df,
        args.test_dir,
        args.output_dir,
        annotate=args.annotate,
    )
    
    print(f"Error analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()