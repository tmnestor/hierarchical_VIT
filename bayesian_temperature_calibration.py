#!/usr/bin/env python3
"""
Integrate temperature calibration with Bayesian correction for hierarchical model.

This script calibrates model temperature for proper confidence estimation,
then applies controlled Bayesian correction for class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

from device_utils import get_device
from datasets import ReceiptDataset
from model_factory import ModelFactory
from receipt_processor import ReceiptProcessor


class ModelWithTemperature(nn.Module):
    """
    A wrapper for a model that applies temperature scaling to the logits.
    
    This uses a calibration_temperature parameter to scale model logits,
    which is different from the bayesian_temperature used for Bayesian correction.
    """
    def __init__(self, model, calibration_temperature=1.0):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.calibration_temperature = nn.Parameter(torch.ones(1) * calibration_temperature)
        
    def forward(self, input_tensor):
        """Forward pass with temperature scaling"""
        output = self.model(input_tensor)
        
        # Extract logits
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        # Scale logits by calibration temperature
        return logits / self.calibration_temperature


class BayesianCalibratedPredictor:
    """
    Hierarchical model for receipt counting with temperature calibration and Bayesian correction.
    
    This class implements a 2-level hierarchical approach:
    - Level 1: Binary classifier (0 vs 1+ receipts)
    - Level 2: Binary classifier (1 vs 2+ receipts)
    
    It uses temperature scaling for proper confidence estimation,
    and Bayesian correction to handle class imbalance.
    """
    def __init__(
        self, 
        level1_model_path, 
        level2_model_path=None, 
        model_type="vit",
        device=None,
        # Parameters for Bayesian correction
        level1_class_weights=None,  # for 0 vs 1+ (loss weights)
        level2_class_weights=None,  # for 1 vs 2+ (loss weights)
        level1_class_freq=None,     # 0 vs 1+
        level2_class_freq=None,     # 1 vs 2+
        level1_sampling_weights=None,  # for 0 vs 1+ (sampling weights)
        level2_sampling_weights=None,  # for 1 vs 2+ (sampling weights)
        metadata_path=None,         # Path to training metadata JSON
        # Temperature parameters
        level1_temperature=None,    # Temperature for level 1 model
        level2_temperature=None,    # Temperature for level 2 model
        bayesian_temperature=0.01,  # Temperature for Bayesian correction
        use_cpu=False               # Force using CPU for all operations
    ):
        """
        Initialize the Bayesian calibrated predictor.
        
        Args:
            level1_model_path: Path to Level 1 model (0 vs 1+)
            level2_model_path: Path to Level 2 model (1 vs 2+)
            model_type: Type of model ("vit" or "swin")
            device: Device to use for inference (if None, uses device_utils)
            level1_class_weights: Weights used when training level 1 model
            level2_class_weights: Weights used when training level 2 model 
            level1_class_freq: Original class frequencies for level 1
            level2_class_freq: Original class frequencies for level 2
            metadata_path: Path to training metadata JSON file
            level1_temperature: Temperature for level 1 model calibration
            level2_temperature: Temperature for level 2 model calibration
            bayesian_temperature: Temperature factor for Bayesian correction
            use_cpu: Force CPU usage for all operations (useful to avoid device mismatch errors)
        """
        # Force CPU for all operations on Apple Silicon to prevent device mismatch errors
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        self.model_type = model_type
        
        # Always use CPU on macOS to avoid MPS/CPU device mismatch
        if use_cpu or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Apple Silicon detected or CPU forced - using CPU to avoid device mismatch errors")
            self.device = torch.device('cpu')
        else:
            self.device = device or get_device()
        self.processor = ReceiptProcessor()
        self.bayesian_temperature = bayesian_temperature
        
        # Initialize temperature attributes to prevent AttributeError
        self.level1_temperature = level1_temperature or 1.0
        self.level2_temperature = level2_temperature or 1.0
        
        # Try to load metadata from file if provided
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract weights and frequencies from metadata
                if 'level1' in metadata and 'class_weights' in metadata['level1']:
                    level1_class_weights = metadata['level1']['class_weights']
                if 'level1' in metadata and 'class_frequencies' in metadata['level1']:
                    level1_class_freq = metadata['level1']['class_frequencies']
                if 'level1' in metadata and 'sampling_weights' in metadata['level1']:
                    level1_sampling_weights = metadata['level1']['sampling_weights']
                if 'level2' in metadata and 'class_weights' in metadata['level2']:
                    level2_class_weights = metadata['level2']['class_weights']
                if 'level2' in metadata and 'class_frequencies' in metadata['level2']:
                    level2_class_freq = metadata['level2']['class_frequencies']
                if 'level2' in metadata and 'sampling_weights' in metadata['level2']:
                    level2_sampling_weights = metadata['level2']['sampling_weights']
                
                print(f"Successfully loaded training metadata from {metadata_path}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                print("Falling back to default or provided weights")
        
        # Use defaults if weights/frequencies not provided or loaded from metadata
        if level1_class_weights is None:
            level1_class_weights = [0.2, 1.8]  # Default values if not provided
            print("Using default Level 1 weights")
        if level2_class_weights is None:
            level2_class_weights = [0.4, 1.6]  # Default values if not provided
            print("Using default Level 2 weights")
        if level1_class_freq is None:
            level1_class_freq = [0.8, 0.2]     # Default values if not provided
            print("Using default Level 1 frequencies")
        if level2_class_freq is None:
            level2_class_freq = [0.7, 0.3]     # Default values if not provided
            print("Using default Level 2 frequencies")
        
        # Store class weights, sampling weights and frequencies for Bayesian correction
        self.level1_class_weights = np.array(level1_class_weights)
        self.level2_class_weights = np.array(level2_class_weights)
        self.level1_class_freq = np.array(level1_class_freq)
        self.level2_class_freq = np.array(level2_class_freq)
        
        # Store sampling weights if provided, or use None to signal using class weights
        self.level1_sampling_weights = np.array(level1_sampling_weights) if level1_sampling_weights is not None else None
        self.level2_sampling_weights = np.array(level2_sampling_weights) if level2_sampling_weights is not None else None
        
        # Set calibration temperature scaling parameters (default to 1.0 if not provided)
        # These are different from bayesian_temperature and control confidence calibration
        self.level1_calibration_temperature = level1_temperature or 1.0
        self.level2_calibration_temperature = level2_temperature or 1.0
        
        print(f"Initializing Bayesian calibrated predictor with {model_type.upper()} models")
        print(f"Using device: {self.device}")
        print(f"Level 1 class weights: {self.level1_class_weights}, frequencies: {self.level1_class_freq}")
        if self.level1_sampling_weights is not None:
            print(f"Level 1 sampling weights: {self.level1_sampling_weights}")
        print(f"Level 1 calibration temperature: {self.level1_calibration_temperature}")
        print(f"Bayesian correction temperature: {self.bayesian_temperature}")
        
        # Load Level 1 model (0 vs 1+)
        print(f"Loading Level 1 model from {level1_model_path}")
        base_level1_model = ModelFactory.load_model(
            level1_model_path, 
            model_type=model_type,
            num_classes=2,
            mode="eval"
        ).to(self.device)
        base_level1_model.eval()
        
        # Wrap Level 1 model with temperature scaling for confidence calibration
        self.level1_model = ModelWithTemperature(base_level1_model, self.level1_calibration_temperature)
        self.level1_model.eval()
        
        # Load Level 2 model (1 vs 2+) if provided
        self.level2_model = None
        if level2_model_path is not None and os.path.exists(level2_model_path):
            print(f"Loading Level 2 model from {level2_model_path}")
            print(f"Level 2 class weights: {self.level2_class_weights}, frequencies: {self.level2_class_freq}")
            if self.level2_sampling_weights is not None:
                print(f"Level 2 sampling weights: {self.level2_sampling_weights}")
            print(f"Level 2 calibration temperature: {self.level2_calibration_temperature}")
            
            try:
                base_level2_model = ModelFactory.load_model(
                    level2_model_path, 
                    model_type=model_type,
                    num_classes=2,
                    mode="eval"
                ).to(self.device)
                base_level2_model.eval()
                
                # Wrap Level 2 model with temperature scaling for confidence calibration
                self.level2_model = ModelWithTemperature(base_level2_model, self.level2_calibration_temperature)
                self.level2_model.eval()
            except Exception as e:
                print(f"Error loading Level 2 model: {e}")
                self.level2_model = None
                print("Falling back to Level 1 model only due to loading error.")
        else:
            self.level2_model = None
            print("Level 2 model not available. Using level 1 model only.")
    
    def apply_bayesian_correction(self, probs, class_weights, class_freq=None, sampling_weights=None, verbose=False, temperature=None):
        """
        Apply enhanced Bayesian correction to undo both biases introduced by weighted training and weighted sampling.
        
        Args:
            probs: Model output probabilities (numpy array)
            class_weights: Weights used during training in the loss function
            class_freq: True class frequencies (if None, assumes uniform)
            sampling_weights: Weights used for weighted sampling (if None, assumes same as class_weights)
            verbose: Whether to print debug information
            temperature: Temperature factor to soften the correction (defaults to self.bayesian_temperature)
            
        Returns:
            Corrected probabilities
        """
        # Use instance temperature if not provided
        if temperature is None:
            temperature = self.bayesian_temperature
            
        # Convert to numpy for easier manipulation
        if isinstance(probs, torch.Tensor):
            # Ensure probs is on CPU before converting to numpy
            probs_np = probs.detach().cpu().numpy()
        else:
            probs_np = np.array(probs)
        weights_np = np.array(class_weights)
        
        # Debug original probabilities
        if verbose:
            print(f"Original probabilities: {probs_np[0]}")
            print(f"Using loss weights: {weights_np}")
            
        # If sampling weights not provided, assume they are the same as loss weights
        # In practice, they're often the same or proportional
        if sampling_weights is None:
            sampling_weights = weights_np
        else:
            sampling_weights = np.array(sampling_weights)
            if verbose:
                print(f"Using sampling weights: {sampling_weights}")
        
        # Apply temperature to weights to soften the correction
        # temperature=1.0 means full correction, temperature=0.0 means no correction
        # Values > 1.0 are treated the same as 1.0 (full correction)
        if temperature < 1.0:
            # Interpolate between original weights and uniform weights [1, 1, ...]
            # As temperature approaches 0, softened_weights approaches uniform weights
            # As temperature approaches 1, softened_weights approaches the original weights
            softened_loss_weights = (1.0 - temperature) * np.ones_like(weights_np) + temperature * weights_np
            softened_sampling_weights = (1.0 - temperature) * np.ones_like(sampling_weights) + temperature * sampling_weights
            if verbose:
                print(f"Softened loss weights (temp={temperature}): {softened_loss_weights}")
                print(f"Softened sampling weights (temp={temperature}): {softened_sampling_weights}")
        else:
            # For temperature >= 1.0, use the full original weights (full correction)
            softened_loss_weights = weights_np
            softened_sampling_weights = sampling_weights
        
        # STEP 1: Correct for loss weights bias
        # P(data|class) = P_model(data|class) / weight_class
        loss_corrected = probs_np / softened_loss_weights
        if verbose:
            print(f"After loss weights correction: {loss_corrected[0]}")
        
        # STEP 2: Correct for sampling weights bias
        # The sampling weights affect how frequently each class is seen during training
        # We need to further correct for this sampling bias
        # If a class was over-sampled, its probability is inflated and needs reduction
        sampling_corrected = loss_corrected / softened_sampling_weights
        if verbose:
            print(f"After sampling weights correction: {sampling_corrected[0]}")
        
        # Re-normalize 
        corrected = sampling_corrected / sampling_corrected.sum(axis=1, keepdims=True)
        if verbose:
            print(f"After normalization: {corrected[0]}")
        
        # STEP 3: Apply true class frequencies as priors if provided
        if class_freq is not None:
            freq_np = np.array(class_freq)
            
            # Also soften the frequencies if using temperature
            if temperature < 1.0:
                # Interpolate between uniform frequencies and original frequencies
                # As temperature approaches 0, softened_freq approaches uniform frequencies
                # As temperature approaches 1, softened_freq approaches the original frequencies
                uniform_freq = np.ones_like(freq_np) / len(freq_np)
                softened_freq = (1.0 - temperature) * uniform_freq + temperature * freq_np
                if verbose:
                    print(f"Softened frequencies (temp={temperature}): {softened_freq}")
            else:
                # For temperature >= 1.0, use the full original frequencies (full correction)
                softened_freq = freq_np
                
            if verbose:
                print(f"Applying true class frequencies: {softened_freq}")
                
            # P(class|data) = P(data|class) * P(class) / P(data)
            # Where P(data) is just a normalization constant
            corrected = corrected * softened_freq
            if verbose:
                print(f"After applying frequencies: {corrected[0]}")
            corrected = corrected / corrected.sum(axis=1, keepdims=True)
            if verbose:
                print(f"Final corrected: {corrected[0]}")
        
        if verbose:
            print(f"Final prediction would be class {np.argmax(corrected[0])}")
        
        return corrected
    
    def preprocess_image(self, image_path, enhance=True):
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to the image
            enhance: Whether to enhance the image
            
        Returns:
            Tensor ready for model input
        """
        # Process image with the same logic as the original predictor
        if enhance:
            try:
                enhanced_path = "enhanced_temp.jpg"
                self.processor.enhance_scan_quality(image_path, enhanced_path)
                image_path = enhanced_path
            except Exception as e:
                print(f"Warning: Image enhancement failed: {e}")
        
        try:
            # Use the processor to preprocess the image - it now uses PIL internally
            img_tensor = self.processor.preprocess_image(image_path)
            
            # Move to device
            img_tensor = img_tensor.to(self.device)
            return img_tensor
        finally:
            # Clean up enhanced image if it was created
            if enhance and os.path.exists("enhanced_temp.jpg"):
                os.remove("enhanced_temp.jpg")
    
    def predict(self, image_path, enhance=True, return_confidences=False, debug=False, bayesian_temperature=None):
        """
        Run hierarchical prediction on an image with temperature calibration and Bayesian correction.
        
        Args:
            image_path: Path to the image
            enhance: Whether to enhance the image before processing
            return_confidences: Whether to return confidence scores
            debug: Whether to print debug information
            bayesian_temperature: Override the instance bayesian_temperature
            
        Returns:
            predicted_count: Final receipt count prediction
            If return_confidences=True, also returns a tuple with confidence and detailed confidences
        """
        # Use instance temperature if not provided
        if bayesian_temperature is None:
            bayesian_temperature = self.bayesian_temperature
            
        # Preprocess image
        img_tensor = self.preprocess_image(image_path, enhance)
        
        # Level 1: 0 vs 1+ receipts
        with torch.no_grad():
            # Get temperature-scaled logits
            level1_scaled_logits = self.level1_model(img_tensor)
                
            # Apply softmax to get calibrated probabilities
            level1_raw_probs = F.softmax(level1_scaled_logits, dim=1)
            
            if debug:
                print("=== LEVEL 1 ===")
                print(f"Raw Level 1 probabilities (temperature-scaled): [No receipts: {level1_raw_probs[0, 0].item():.4f}, Has receipts: {level1_raw_probs[0, 1].item():.4f}]")
            
            # Make sure to detach the tensor before applying Bayesian correction
            # This ensures we don't have device conflicts
            level1_raw_probs_detached = level1_raw_probs.detach().cpu()
            
            # Apply Bayesian correction to handle class imbalance
            level1_corrected = self.apply_bayesian_correction(
                level1_raw_probs_detached, 
                self.level1_class_weights,
                self.level1_class_freq,
                sampling_weights=self.level1_sampling_weights,
                verbose=debug,
                temperature=bayesian_temperature
            )
            
            # Make prediction using corrected probabilities
            level1_prediction = np.argmax(level1_corrected[0])
            level1_confidence = level1_corrected[0, level1_prediction]
            
            if debug:
                print(f"Level 1 prediction: {'No receipts' if level1_prediction == 0 else 'Has receipts'}")
                print(f"Level 1 confidence: {level1_confidence:.4f}")
        
        # Store confidences at each level - use both raw and corrected
        confidences = {
            'level1': {
                'raw': {
                    'no_receipts': level1_raw_probs[0, 0].item(),
                    'has_receipts': level1_raw_probs[0, 1].item()
                },
                'corrected': {
                    'no_receipts': float(level1_corrected[0, 0]),  # Ensure numpy values are converted to Python float
                    'has_receipts': float(level1_corrected[0, 1])
                }
            }
        }
        
        # If prediction is 0 receipts, we're done
        if level1_prediction == 0:
            if debug:
                print("Final prediction: 0 receipts (from Level 1)")
                
            if return_confidences:
                return 0, level1_confidence, confidences
            else:
                return 0
        
        # Check if we have a level 2 model
        if self.level2_model is None:
            # If no level 2 model, return 1 (we already know it's not 0 from level 1)
            final_prediction = 1
            final_confidence = level1_confidence
            
            if debug:
                print("Final prediction: 1 receipt (from Level 1, no Level 2 model)")
            
            # Store placeholder level 2 confidences
            confidences['level2'] = {
                'raw': {
                    'one_receipt': 1.0,
                    'multiple_receipts': 0.0
                },
                'corrected': {
                    'one_receipt': 1.0,
                    'multiple_receipts': 0.0
                }
            }
            
            if return_confidences:
                return final_prediction, final_confidence, confidences
            else:
                return final_prediction
        
        # If we have a level 2 model, use it
        # Level 2: 1 vs 2+ receipts
        with torch.no_grad():
            # Get temperature-scaled logits
            level2_scaled_logits = self.level2_model(img_tensor)
                
            # Apply softmax to get calibrated probabilities
            level2_raw_probs = F.softmax(level2_scaled_logits, dim=1)
            
            if debug:
                print("=== LEVEL 2 ===")
                print(f"Raw Level 2 probabilities (temperature-scaled): [One receipt: {level2_raw_probs[0, 0].item():.4f}, Multiple receipts: {level2_raw_probs[0, 1].item():.4f}]")
            
            # Make sure to detach the tensor before applying Bayesian correction
            # This ensures we don't have device conflicts
            level2_raw_probs_detached = level2_raw_probs.detach().cpu()
            
            # Apply Bayesian correction to handle class imbalance
            level2_corrected = self.apply_bayesian_correction(
                level2_raw_probs_detached, 
                self.level2_class_weights,
                self.level2_class_freq,
                sampling_weights=self.level2_sampling_weights,
                verbose=debug,
                temperature=bayesian_temperature
            )
            
            # Make prediction using corrected probabilities
            level2_prediction = np.argmax(level2_corrected[0])
            level2_confidence = level2_corrected[0, level2_prediction]
            
            # Here's the issue - we need to map level2_prediction correctly
            # In level 2 classification, class 0 means "1 receipt" and class 1 means "2+ receipts"
            level2_label = "One receipt" if level2_prediction == 0 else "Multiple receipts"
            
            if debug:
                print(f"Level 2 prediction: {level2_label} (class {level2_prediction})")
                print(f"Level 2 confidence: {level2_confidence:.4f}")
        
        # Store level 2 confidences - both raw and corrected
        confidences['level2'] = {
            'raw': {
                'one_receipt': level2_raw_probs[0, 0].item(),
                'multiple_receipts': level2_raw_probs[0, 1].item()
            },
            'corrected': {
                'one_receipt': float(level2_corrected[0, 0]),  # Ensure numpy values are converted to Python float
                'multiple_receipts': float(level2_corrected[0, 1])
            }
        }
        
        # If prediction is 1 receipt, we're done
        if level2_prediction == 0:  # 0 means class "1 receipt" in level 2
            if debug:
                print("Final prediction: 1 receipt (from Level 2)")
                
            if return_confidences:
                return 1, level2_confidence, confidences
            else:
                return 1
        
        # If 2+ receipts, predict 2
        final_prediction = 2
        final_confidence = level2_confidence
        
        if debug:
            print("Final prediction: 2+ receipts (from Level 2)")
            
        if return_confidences:
            return final_prediction, final_confidence, confidences
        else:
            return final_prediction
    
    def evaluate_on_dataset(self, csv_file, image_dir, output_dir=None, enhance=True, debug=False, bayesian_temperature=None):
        """
        Evaluate the hierarchical model on a dataset.
        
        Args:
            csv_file: Path to CSV file with filenames and ground truth labels
            image_dir: Directory containing the images
            output_dir: Directory to save evaluation results
            enhance: Whether to enhance images
            debug: Whether to print debug information
            bayesian_temperature: Override the instance bayesian_temperature
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Use instance temperature if not provided
        if bayesian_temperature is None:
            bayesian_temperature = self.bayesian_temperature
            
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Map all receipt counts > 1 to class 2 (multiple receipts)
        if 'receipt_count' in df.columns:
            df['original_count'] = df['receipt_count']  # Save original count
            df['receipt_count'] = df['receipt_count'].apply(lambda x: min(x, 2))
            
        print(f"Loaded {len(df)} samples from {csv_file} (mapped to 0/1/2+ classification)")
        
        # Initialize results lists
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_detailed_confidences = []
        
        # Process each image
        failed_images = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                # Construct image path
                if os.path.isabs(row['filename']):
                    image_path = row['filename']
                else:
                    image_path = os.path.join(image_dir, row['filename'])
                
                # Get ground truth - already mapped to hierarchical structure (0, 1, 2+)
                true_count = row['receipt_count']
                
                # Get prediction using hierarchical model - wrap in its own try block
                try:
                    result = self.predict(
                        image_path, 
                        enhance=enhance,
                        return_confidences=True,
                        debug=debug,
                        bayesian_temperature=bayesian_temperature
                    )
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        predicted_count, confidence, detailed_confidences = result
                    else:
                        # Handle case where predict returns just the count
                        predicted_count = result
                        confidence = 1.0  # Default confidence
                        detailed_confidences = {}
                    
                    all_predictions.append(predicted_count)
                    all_targets.append(true_count)
                    all_confidences.append(confidence)
                    all_detailed_confidences.append(detailed_confidences)
                except Exception as e:
                    print(f"Prediction error for {row['filename']}: {e}")
                    # Use a default value for failed predictions
                    all_predictions.append(-1)
                    all_targets.append(true_count)
                    all_confidences.append(0.0)
                    all_detailed_confidences.append({})
                    failed_images.append((row['filename'], f"Prediction error: {str(e)}"))
                
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
                failed_images.append((row['filename'], str(e)))
                # Use a default value for failed predictions
                all_predictions.append(-1)
                all_targets.append(true_count)
                all_confidences.append(0.0)
                all_detailed_confidences.append({})
        
        # Filter out failed predictions
        valid_indices = [i for i, p in enumerate(all_predictions) if p != -1]
        if len(valid_indices) < len(all_predictions):
            print(f"Warning: {len(all_predictions) - len(valid_indices)} images failed processing")
            
            # Filter predictions and targets
            filtered_predictions = [all_predictions[i] for i in valid_indices]
            filtered_targets = [all_targets[i] for i in valid_indices]
            filtered_confidences = [all_confidences[i] for i in valid_indices]
        else:
            filtered_predictions = all_predictions
            filtered_targets = all_targets
            filtered_confidences = all_confidences
        
        # Only compute metrics if we have valid predictions
        if len(filtered_predictions) == 0:
            print("No valid predictions!")
            return {
                "error": "No valid predictions",
                "failed_images": failed_images,
                "overall": {
                    "accuracy": 0,
                    "balanced_accuracy": 0,
                    "f1_macro": 0
                },
                "level1": {
                    "accuracy": 0,
                    "balanced_accuracy": 0
                },
                "level2": {
                    "accuracy": 0,
                    "balanced_accuracy": 0
                },
                "parameters": {
                    "level1_temperature": self.level1_temperature,
                    "level2_temperature": self.level2_temperature,
                    "bayesian_temperature": bayesian_temperature
                }
            }
        
        # We don't need to map here anymore since we mapped at loading time
        # Calculate metrics using the already mapped targets
        accuracy = accuracy_score(filtered_targets, filtered_predictions)
        balanced_accuracy = balanced_accuracy_score(filtered_targets, filtered_predictions)
        f1 = f1_score(filtered_targets, filtered_predictions, average='macro')
        
        # Per-class metrics
        class_report = classification_report(filtered_targets, filtered_predictions, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(filtered_targets, filtered_predictions)
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        
        # Save results if output directory provided
        if output_dir:
            # Save results to CSV, including both original and mapped targets for clarity
            results_df = pd.DataFrame({
                'filename': df['filename'],
                'actual': all_targets,  # Already mapped at loading time
                'original_count': df['original_count'] if 'original_count' in df.columns else all_targets,
                'predicted': all_predictions,
                'confidence': all_confidences
            })
            results_df.to_csv(output_path / f'calibrated_hierarchical_results_{bayesian_temperature:.6f}.csv', index=False)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=range(3), yticklabels=range(3))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix (Bayesian Temp = {bayesian_temperature:.6f})')
            plt.savefig(output_path / f'confusion_matrix_{bayesian_temperature:.6f}.png')
            plt.close()
            
            # Save failed images list
            if failed_images:
                with open(output_path / 'failed_images.txt', 'w') as f:
                    for img, error in failed_images:
                        f.write(f"{img}: {error}\n")
        
        # Calculate hierarchical performance
        # Level 1 performance (0 vs 1+)
        level1_targets = [0 if t == 0 else 1 for t in filtered_targets]
        level1_preds = [0 if p == 0 else 1 for p in filtered_predictions]
        level1_accuracy = accuracy_score(level1_targets, level1_preds)
        level1_balanced_accuracy = balanced_accuracy_score(level1_targets, level1_preds)
        
        # Print detailed Level 1 confusion
        if debug:
            level1_cm = confusion_matrix(level1_targets, level1_preds)
            print("\nLevel 1 Confusion Matrix (0 vs 1+):")
            print(level1_cm)
        
        # Level 2 performance (1 vs 2+)
        # Filter to only samples with 1+ receipts
        level2_indices = [i for i, t in enumerate(filtered_targets) if t > 0]
        if level2_indices:
            level2_targets = [0 if filtered_targets[i] == 1 else 1 for i in level2_indices]
            level2_preds = [0 if filtered_predictions[i] == 1 else 1 for i in level2_indices]
            level2_accuracy = accuracy_score(level2_targets, level2_preds)
            level2_balanced_accuracy = balanced_accuracy_score(level2_targets, level2_preds)
            
            # Print detailed Level 2 confusion
            if debug:
                level2_cm = confusion_matrix(level2_targets, level2_preds)
                print("\nLevel 2 Confusion Matrix (1 vs 2+):")
                print(level2_cm)
        else:
            level2_accuracy = 0
            level2_balanced_accuracy = 0
        
        # Print hierarchical performance
        print("\nHierarchical Performance:")
        print(f"Level 1 (0 vs 1+): Accuracy={level1_accuracy:.4f}, Balanced Accuracy={level1_balanced_accuracy:.4f}")
        print(f"Level 2 (1 vs 2+): Accuracy={level2_accuracy:.4f}, Balanced Accuracy={level2_balanced_accuracy:.4f}")
        
        # Package all metrics
        hierarchical_metrics = {
            'overall': {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'f1_macro': f1,
            },
            'level1': {
                'accuracy': level1_accuracy,
                'balanced_accuracy': level1_balanced_accuracy
            },
            'level2': {
                'accuracy': level2_accuracy,
                'balanced_accuracy': level2_balanced_accuracy
            },
            'class_report': class_report,
            'confusion_matrix': cm,
            'parameters': {
                'level1_calibration_temperature': self.level1_calibration_temperature,
                'level2_calibration_temperature': self.level2_calibration_temperature,
                'bayesian_temperature': bayesian_temperature
            }
        }
        
        if output_dir:
            # Save detailed metrics as JSON
            with open(output_path / f'metrics_{bayesian_temperature:.6f}.json', 'w') as f:
                # Convert NumPy values to Python native types for JSON serialization
                metrics_json = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                       {k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2) 
                       for k2, v2 in v.items()} if isinstance(v, dict) else v)
                    for k, v in hierarchical_metrics.items()
                }
                json.dump(metrics_json, f, indent=4)
                
            hierarchical_metrics['results_csv'] = str(output_path / f'calibrated_hierarchical_results_{bayesian_temperature:.6f}.csv')
        
        return hierarchical_metrics


def calibrate_model(model_path, val_loader, device, init_calibration_temp=1.0, max_iter=50, lr=0.01):
    """
    Calibrate a model's confidence calibration temperature parameter.
    
    This calibration_temperature is different from bayesian_temperature:
    - calibration_temperature: Scales model logits to calibrate confidence (typically >1.0)
    - bayesian_temperature: Controls strength of Bayesian correction (0.0-1.0)
    
    Args:
        model_path: Path to the model
        val_loader: DataLoader with validation data
        device: Device to run on
        init_calibration_temp: Initial calibration temperature value
        max_iter: Maximum iterations for optimization
        lr: Learning rate
        
    Returns:
        float: Calibrated temperature
    """
    # For OpenCV issues, we'll use a simplified approach
    try:
        # Load the model with our patched ModelFactory
        model_path_str = str(model_path)  # Convert PosixPath to string
        model = ModelFactory.load_model(
            model_path,
            model_type="swin" if "swin" in model_path_str else "vit",
            num_classes=2,
            mode="eval"
        ).to(device)
        model.eval()
        
        # Create a temperature wrapper
        temp_model = ModelWithTemperature(model, init_calibration_temp)
        temp_model.to(device)
        
        # Define NLL loss
        nll_criterion = nn.CrossEntropyLoss()
        
        # Define optimizer (LBFGS works well for this task)
        optimizer = optim.LBFGS([temp_model.calibration_temperature], lr=lr, max_iter=max_iter)
        
        # Function to evaluate and update the calibration temperature parameter
        def eval_for_calibration_temperature():
            optimizer.zero_grad()
            
            # Collect all logits and targets
            all_logits = []
            all_targets = []
            
            # First get all logits with no gradients
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Get logits from base model
                    if hasattr(model(inputs), 'logits'):
                        logits = model(inputs).logits
                    else:
                        logits = model(inputs)
                    
                    all_logits.append(logits)
                    all_targets.append(targets)
            
            # Concatenate all logits and targets
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            
            # Apply temperature scaling
            scaled_logits = all_logits / temp_model.calibration_temperature
            
            # Calculate loss
            loss = nll_criterion(scaled_logits, all_targets)
            loss.backward()
            
            # Print current state
            current_temp = temp_model.calibration_temperature.item()
            print(f"Calibration Temperature: {current_temp:.4f}, Loss: {loss.item():.6f}")
            
            return loss
        
        # Optimize the calibration temperature parameter
        print("\nOptimizing calibration temperature parameter...")
        optimizer.step(eval_for_calibration_temperature)
        
        # Get optimized temperature
        final_temp = temp_model.calibration_temperature.item()
        print(f"\nCalibration complete! Optimal calibration temperature: {final_temp:.4f}")
        
        return final_temp
    
    except Exception as e:
        # If there's an error, use a fallback value and print the warning
        print(f"Error during calibration: {e}")
        print("Using fallback temperature of 1.0")
        return 1.0


def run_temperature_sweep(predictor, csv_file, image_dir, output_dir, temp_values, enhance=True, debug=False, optimize_metric="f1"):
    """
    Evaluate the predictor across a range of Bayesian temperature values.
    
    Note: This sweeps the Bayesian correction temperature (0.0-1.0), not the 
    model confidence calibration temperature (which is typically >1.0).
    
    Args:
        predictor: BayesianCalibratedPredictor instance
        csv_file: Path to CSV file with ground truth labels
        image_dir: Directory containing the images
        output_dir: Directory to save results
        temp_values: List of Bayesian temperature values to evaluate (0.0-1.0 range)
        enhance: Whether to enhance images
        debug: Whether to print debug information
        optimize_metric: Which metric to optimize ("accuracy", "balanced_accuracy", or "f1")
        
    Returns:
        tuple: (DataFrame with results, best Bayesian temperature for specified metric)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    
    # Evaluate each temperature value
    for temp in temp_values:
        print(f"\nEvaluating with bayesian_temperature = {temp:.6f}")
        
        # Run evaluation
        metrics = predictor.evaluate_on_dataset(
            csv_file=csv_file,
            image_dir=image_dir,
            output_dir=output_dir,
            enhance=enhance,
            debug=debug,
            bayesian_temperature=temp
        )
        
        # Store result
        result = {
            'temperature': temp,
            'accuracy': metrics['overall']['accuracy'],
            'balanced_accuracy': metrics['overall']['balanced_accuracy'],
            'f1_macro': metrics['overall']['f1_macro'],
        }
        
        # Add hierarchical metrics if available
        if 'level1' in metrics:
            result['level1_accuracy'] = metrics['level1']['accuracy']
            result['level1_balanced_accuracy'] = metrics['level1']['balanced_accuracy']
        else:
            result['level1_accuracy'] = 0.0
            result['level1_balanced_accuracy'] = 0.0
            
        if 'level2' in metrics:
            result['level2_accuracy'] = metrics['level2']['accuracy']
            result['level2_balanced_accuracy'] = metrics['level2']['balanced_accuracy']
        else:
            result['level2_accuracy'] = 0.0
            result['level2_balanced_accuracy'] = 0.0
            
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'temperature_sweep_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['temperature'], results_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(results_df['temperature'], results_df['balanced_accuracy'], 's-', label='Balanced Accuracy')
    plt.plot(results_df['temperature'], results_df['f1_macro'], '^-', label='F1 Score')
    
    # Add level-specific metrics
    plt.plot(results_df['temperature'], results_df['level1_balanced_accuracy'], 'x--', label='L1 Balanced Acc')
    plt.plot(results_df['temperature'], results_df['level2_balanced_accuracy'], '+--', label='L2 Balanced Acc')
    
    # Use log scale for temperature if range is wide
    if max(temp_values) / min(temp_values) > 10:
        plt.xscale('log')
    
    plt.xlabel('Bayesian Correction Temperature (0.0-1.0)')
    plt.ylabel('Metric Value')
    plt.title('Model Performance vs Bayesian Correction Temperature\n(Note: This is different from model calibration temperature)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path / 'bayesian_temperature_sweep_plot.png')
    plt.close()
    
    # Find best temperature for different metrics
    best_temp_acc = results_df.loc[results_df['accuracy'].idxmax(), 'temperature']
    best_temp_bal_acc = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'temperature']
    best_temp_f1 = results_df.loc[results_df['f1_macro'].idxmax(), 'temperature']
    
    # Get the recommended temperature based on the specified metric
    if optimize_metric == "accuracy":
        recommended_temp = best_temp_acc
        metric_name = "accuracy"
    elif optimize_metric == "balanced_accuracy":
        recommended_temp = best_temp_bal_acc
        metric_name = "balanced accuracy"
    else:  # Default to F1
        recommended_temp = best_temp_f1
        metric_name = "F1 score"
    
    print("\nBest temperatures:")
    print(f"  For accuracy: {best_temp_acc:.6f}")
    print(f"  For balanced accuracy: {best_temp_bal_acc:.6f}")
    print(f"  For F1 score: {best_temp_f1:.6f}")
    print(f"\nRecommended temperature (optimized for {metric_name}): {recommended_temp:.6f}")
    
    # Save this info to a file
    with open(output_path / 'best_temperatures.txt', 'w') as f:
        f.write(f"Best temperature for accuracy: {best_temp_acc:.6f}\n")
        f.write(f"Best temperature for balanced accuracy: {best_temp_bal_acc:.6f}\n")
        f.write(f"Best temperature for F1 score: {best_temp_f1:.6f}\n")
        f.write(f"\nRecommended temperature (optimized for {metric_name}): {recommended_temp:.6f}\n")
    
    return results_df, recommended_temp


def parse_arguments():
    """
    Parse command line arguments for the Bayesian Calibrated Predictor.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Bayesian Calibrated Predictor")
    
    # Model options
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        "--model_base_path", "-m", 
        default="models/hierarchical",
        help="Base path to hierarchical models"
    )
    model_group.add_argument(
        "--model_type",
        choices=["vit", "swin"],
        default="vit",
        help="Type of transformer model (vit or swin)"
    )
    model_group.add_argument(
        "--level2_model_path",
        help="Custom path to Level 2 model (overrides the default path)"
    )
    
    # Calibration options
    cal_group = parser.add_argument_group('Calibration')
    cal_group.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate model confidence calibration temperature before prediction"
    )
    cal_group.add_argument(
        "--level1_temperature", type=float, default=1.0,
        help="Confidence calibration temperature for Level 1 model (typically >1.0, overrides calibration)"
    )
    cal_group.add_argument(
        "--level2_temperature", type=float, default=1.0,
        help="Confidence calibration temperature for Level 2 model (typically >1.0, overrides calibration)"
    )
    cal_group.add_argument(
        "--bayesian_temperature", type=float, default=0.01,
        help="Bayesian correction temperature (0.0-1.0, where 0.0=no correction, 1.0=full correction)"
    )
    
    # Bayesian Temperature sweep options
    sweep_group = parser.add_argument_group('Bayesian Temperature Sweep')
    sweep_group.add_argument(
        "--sweep", action="store_true",
        help="Run a Bayesian temperature sweep to find optimal correction strength"
    )
    sweep_group.add_argument(
        "--min_temp", type=float, default=0.001,
        help="Minimum Bayesian temperature for sweep (0.0-1.0 range)"
    )
    sweep_group.add_argument(
        "--max_temp", type=float, default=0.1,
        help="Maximum Bayesian temperature for sweep (0.0-1.0 range)"
    )
    sweep_group.add_argument(
        "--num_temps", type=int, default=10,
        help="Number of Bayesian temperature values to evaluate"
    )
    sweep_group.add_argument(
        "--log_scale", action="store_true",
        help="Use logarithmic spacing for Bayesian temperature values"
    )
    sweep_group.add_argument(
        "--optimize_metric", choices=["accuracy", "balanced_accuracy", "f1"],
        default="f1",
        help="Metric to optimize for Bayesian temperature (default: f1)"
    )
    sweep_group.add_argument(
        "--use_best_temp", action="store_true",
        help="After sweep, use the best temperature for the chosen metric"
    )
    
    # Class weight options (rarely need to change these)
    weights_group = parser.add_argument_group('Class Weights')
    weights_group.add_argument(
        "--level1_weights",
        default=None,
        help="Comma-separated weights used for Level 1 training (0 vs 1+)"
    )
    weights_group.add_argument(
        "--level2_weights",
        default=None,
        help="Comma-separated weights used for Level 2 training (1 vs 2+)"
    )
    weights_group.add_argument(
        "--level1_freq",
        default=None,
        help="Comma-separated class frequencies for Level 1 (0 vs 1+)"
    )
    weights_group.add_argument(
        "--level2_freq",
        default=None,
        help="Comma-separated class frequencies for Level 2 (1 vs 2+)"
    )
    
    # Operation mode
    mode_group = parser.add_argument_group('Operation Mode')
    mode_group.add_argument(
        "--mode",
        choices=["predict", "evaluate"],
        default="predict",
        help="Operation mode: predict single image or evaluate on dataset"
    )
    
    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        "--image", "-i",
        help="Path to input image for prediction mode"
    )
    input_group.add_argument(
        "--test_csv",
        help="Path to CSV file with test data for evaluate mode"
    )
    input_group.add_argument(
        "--test_dir",
        help="Directory containing test images for evaluate mode"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        "--output_dir", "-o",
        default="evaluation/calibrated_hierarchical",
        help="Directory to save results"
    )
    
    # Processing options
    processing_group = parser.add_argument_group('Processing')
    processing_group.add_argument(
        "--no-enhance", action="store_true",
        help="Disable image enhancement before processing"
    )
    processing_group.add_argument(
        "--debug", action="store_true",
        help="Enable debug output"
    )
    processing_group.add_argument(
        "--use-cpu", action="store_true",
        help="Force CPU usage for all operations (helps avoid device mismatch errors)"
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the Bayesian Calibrated Predictor.
    """
    args = parse_arguments()

    # Parse class weights and frequencies if provided
    level1_weights = None
    level2_weights = None
    level1_freq = None
    level2_freq = None
    
    if args.level1_weights:
        level1_weights = [float(w) for w in args.level1_weights.split(',')]
        print(f"Using custom Level 1 weights: {level1_weights}")
    
    if args.level2_weights:
        level2_weights = [float(w) for w in args.level2_weights.split(',')]
        print(f"Using custom Level 2 weights: {level2_weights}")
        
    if args.level1_freq:
        level1_freq = [float(f) for f in args.level1_freq.split(',')]
        print(f"Using custom Level 1 frequencies: {level1_freq}")
        
    if args.level2_freq:
        level2_freq = [float(f) for f in args.level2_freq.split(',')]
        print(f"Using custom Level 2 frequencies: {level2_freq}")
    
    # Determine model paths
    model_base_path = Path(args.model_base_path)
    level1_model_path = model_base_path / "level1" / f"receipt_counter_{args.model_type}_best.pth"
    
    # Use custom Level 2 model if provided, otherwise use the default path
    if args.level2_model_path:
        level2_model_path = Path(args.level2_model_path)
        print(f"Using custom Level 2 model from: {level2_model_path}")
    else:
        level2_model_path = model_base_path / "level2" / f"receipt_counter_{args.model_type}_best.pth"
    
    # Check for training metadata
    metadata_path = model_base_path / "training_metadata.json"
    if not metadata_path.exists():
        print(f"Warning: No training metadata found at {metadata_path}")
        print("Will use provided or default class weights and frequencies")
    else:
        print(f"Found training metadata at {metadata_path}")
    
    # Validate required paths
    if not level1_model_path.exists():
        raise FileNotFoundError(f"Level 1 model not found at {level1_model_path}")
    
    # For level 2, check if it exists; if not, set to None and notify
    if not level2_model_path.exists():
        print(f"Warning: Level 2 model not found at {level2_model_path}")
        print("Running in simplified mode with only Level 1 predictions (0 vs 1+)")
        level2_model_path = None
    
    # Calibrate models if requested
    level1_temperature = args.level1_temperature
    level2_temperature = args.level2_temperature
    
    if args.calibrate:
        print("Calibrating model temperature parameters...")
        device = get_device()
        
        # Create validation datasets for calibration
        if not args.test_csv or not args.test_dir:
            raise ValueError("--test_csv and --test_dir are required for calibration")
            
        # Create Level 1 validation dataset
        level1_val_dataset = ReceiptDataset(
            csv_file=args.test_csv,
            img_dir=args.test_dir,
            augment=False,
            hierarchical_level="level1"
        )
        
        level1_val_loader = DataLoader(
            level1_val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
        
        # Calibrate Level 1 model
        print("Calibrating Level 1 model...")
        level1_temperature = calibrate_model(
            model_path=level1_model_path,
            val_loader=level1_val_loader,
            device=device
        )
        
        # Calibrate Level 2 model if available
        if level2_model_path is not None and level2_model_path.exists():
            # Create Level 2 validation dataset
            level2_val_dataset = ReceiptDataset(
                csv_file=args.test_csv,
                img_dir=args.test_dir,
                augment=False,
                hierarchical_level="level2"
            )
            
            level2_val_loader = DataLoader(
                level2_val_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=4
            )
            
            # Calibrate Level 2 model
            print("Calibrating Level 2 model...")
            level2_temperature = calibrate_model(
                model_path=level2_model_path,
                val_loader=level2_val_loader,
                device=device
            )
        
        # Save calibration parameters
        calibration_path = model_base_path / "calibration_params.json"
        with open(calibration_path, 'w') as f:
            json.dump({
                'level1_temperature': level1_temperature,
                'level2_temperature': level2_temperature
            }, f, indent=4)
            
        print(f"Calibration parameters saved to {calibration_path}")
    
    # Create Bayesian calibrated predictor
    predictor = BayesianCalibratedPredictor(
        level1_model_path=level1_model_path,
        level2_model_path=level2_model_path,
        model_type=args.model_type,
        level1_class_weights=level1_weights,
        level2_class_weights=level2_weights,
        level1_class_freq=level1_freq,
        level2_class_freq=level2_freq,
        metadata_path=metadata_path if metadata_path.exists() else None,
        level1_temperature=level1_temperature,
        level2_temperature=level2_temperature,
        bayesian_temperature=args.bayesian_temperature,
        use_cpu=args.use_cpu  # Pass the force CPU flag
    )
    
    # Run temperature sweep if requested
    if args.sweep:
        if not args.test_csv or not args.test_dir:
            raise ValueError("--test_csv and --test_dir are required for temperature sweep")
        
        # Generate temperature values
        if args.log_scale:
            from numpy import logspace, log10
            temp_values = logspace(log10(args.min_temp), log10(args.max_temp), args.num_temps)
        else:
            from numpy import linspace
            temp_values = linspace(args.min_temp, args.max_temp, args.num_temps)
        
        # Run sweep
        print(f"Running temperature sweep with {args.num_temps} values from {args.min_temp} to {args.max_temp}")
        print(f"Optimizing for metric: {args.optimize_metric}")
        
        # Run the sweep and get recommended temperature
        results_df, best_temp = run_temperature_sweep(
            predictor=predictor,
            csv_file=args.test_csv,
            image_dir=args.test_dir,
            output_dir=args.output_dir,
            temp_values=temp_values,
            enhance=not args.no_enhance,
            debug=args.debug,
            optimize_metric=args.optimize_metric
        )
        
        # If requested, use the best temperature for the specified metric
        if args.use_best_temp:
            print(f"\nUsing optimal temperature: {best_temp:.6f}")
            # Update the predictor with the optimal temperature
            predictor.bayesian_temperature = best_temp
            
            # Run evaluation with the optimal temperature
            print(f"\nEvaluating with optimal temperature ({best_temp:.6f}):")
            metrics = predictor.evaluate_on_dataset(
                csv_file=args.test_csv,
                image_dir=args.test_dir,
                output_dir=args.output_dir,
                enhance=(not args.no_enhance),
                debug=args.debug
            )
            
            # Print final results
            print("\nFinal evaluation with optimal temperature:")
            print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
            print(f"Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.4f}")
            print(f"F1 Score: {metrics['overall']['f1_macro']:.4f}")
        
        print("Temperature sweep complete!")
        return
    
    # Handle different modes
    if args.mode == "predict":
        if not args.image:
            raise ValueError("Please provide an image path with --image")
        
        # Run prediction
        predicted_count, confidence, confidences = predictor.predict(
            args.image, 
            enhance=(not args.no_enhance),
            return_confidences=True,
            debug=args.debug
        )
        
        print(f"\nPrediction for {args.image}:")
        print(f"Predicted receipt count: {predicted_count}")
        print(f"Confidence: {confidence:.4f}")
        
        # Print detailed confidences
        print("\nConfidence breakdown:")
        print("Level 1 (0 vs 1+):")
        print("  Raw probabilities (temperature-calibrated):")
        for label, conf in confidences['level1']['raw'].items():
            print(f"    {label}: {conf:.4f}")
        print("  Bayesian corrected probabilities:")
        for label, conf in confidences['level1']['corrected'].items():
            print(f"    {label}: {conf:.4f}")
        
        if 'level2' in confidences:
            print("\nLevel 2 (1 vs 2+):")
            print("  Raw probabilities (temperature-calibrated):")
            for label, conf in confidences['level2']['raw'].items():
                print(f"    {label}: {conf:.4f}")
            print("  Bayesian corrected probabilities:")
            for label, conf in confidences['level2']['corrected'].items():
                print(f"    {label}: {conf:.4f}")
        
    elif args.mode == "evaluate":
        if not args.test_csv or not args.test_dir:
            raise ValueError("Please provide test_csv and test_dir for evaluation mode")
        
        # Run evaluation
        metrics = predictor.evaluate_on_dataset(
            args.test_csv,
            args.test_dir,
            output_dir=args.output_dir,
            enhance=(not args.no_enhance),
            debug=args.debug
        )
        
        # Print overall summary
        print("\nEvaluation Summary:")
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.4f}")
        print(f"F1 Macro: {metrics['overall']['f1_macro']:.4f}")
        print(f"\nLevel 1 (0 vs 1+) Accuracy: {metrics['level1']['accuracy']:.4f}")
        print(f"Level 1 Balanced Accuracy: {metrics['level1']['balanced_accuracy']:.4f}")
        print(f"\nLevel 2 (1 vs 2+) Accuracy: {metrics['level2']['accuracy']:.4f}")
        print(f"Level 2 Balanced Accuracy: {metrics['level2']['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()