import torch
import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import shutil
from device_utils import get_device
from receipt_processor import ReceiptProcessor
from model_factory import ModelFactory
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def ensure_model_compatibility(model_path):
    """
    Ensure the model is in a compatible format for loading.
    If the model is a checkpoint with model_state_dict, extracts just the state dict.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Path to a compatible model file (same as input if already compatible)
    """
    try:
        # Try loading the model
        checkpoint = torch.load(model_path, weights_only=False)
        
        # Check if it's a checkpoint with model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f"Detected checkpoint format in {model_path}, converting to state dict format")
            # Create a temporary file for the converted model
            temp_path = Path(model_path).with_name(f"{Path(model_path).stem}_converted_temp.pth")
            # Save just the state dict
            torch.save(checkpoint['model_state_dict'], temp_path)
            return temp_path
        else:
            # It's already a state dict or compatible format
            return model_path
    except Exception as e:
        print(f"Warning: Could not check model format: {e}")
        # Return the original path and hope for the best
        return model_path


class HierarchicalPredictor:
    """
    Hierarchical model for receipt counting.
    
    This class implements a 2-level hierarchical approach:
    - Level 1: Binary classifier (0 vs 1+ receipts)
    - Level 2: Binary classifier (1 vs 2+ receipts)
    - Optional multiclass classifier for 2+ receipts
    """
    def __init__(
        self, 
        level1_model_path, 
        level2_model_path=None, 
        multiclass_model_path=None, 
        model_type="vit",
        device=None,
        lower_class1_threshold=None
    ):
        """
        Initialize the hierarchical predictor.
        
        Args:
            level1_model_path: Path to Level 1 model (0 vs 1+)
            level2_model_path: Path to Level 2 model (1 vs 2+)
            multiclass_model_path: Path to multiclass model (2+ receipts), optional
            model_type: Type of model ("vit" or "swin")
            device: Device to use for inference (if None, uses device_utils)
        """
        self.model_type = model_type
        self.device = device or get_device()
        self.processor = ReceiptProcessor()
        self.use_multiclass = multiclass_model_path is not None
        self.lower_class1_threshold = lower_class1_threshold
        
        # Store standard decision threshold for Bayesian correction
        self.standard_threshold = 0.5
        
        if self.lower_class1_threshold is not None:
            print(f"Using lower threshold ({self.lower_class1_threshold}) for class 1 in Level 1 model")
            print(f"Will apply Bayesian correction during inference to maintain calibration")
        
        print(f"Initializing hierarchical predictor with {model_type.upper()} models")
        print(f"Using device: {self.device}")
        
        # Ensure model compatibility and load Level 1 model (0 vs 1+)
        print(f"Loading Level 1 model from {level1_model_path}")
        compatible_level1_path = ensure_model_compatibility(level1_model_path)
        self.level1_model = ModelFactory.load_model(
            compatible_level1_path, 
            model_type=model_type,
            num_classes=2,
            mode="eval"
        ).to(self.device)
        self.level1_model.eval()
        
        # Clean up temporary file if created
        if compatible_level1_path != level1_model_path:
            try:
                os.remove(compatible_level1_path)
            except:
                pass
        
        # Load Level 2 model (1 vs 2+) if provided
        if level2_model_path is not None:
            print(f"Loading Level 2 model from {level2_model_path}")
            compatible_level2_path = ensure_model_compatibility(level2_model_path)
            try:
                self.level2_model = ModelFactory.load_model(
                    compatible_level2_path, 
                    model_type=model_type,
                    num_classes=2,
                    mode="eval"
                ).to(self.device)
                self.level2_model.eval()
                
                # Clean up temporary file if created
                if compatible_level2_path != level2_model_path:
                    try:
                        os.remove(compatible_level2_path)
                    except:
                        pass
            except Exception as e:
                print(f"Error loading Level 2 model: {e}")
                self.level2_model = None
                print("Falling back to Level 1 model only due to loading error.")
        else:
            self.level2_model = None
            print("Level 2 model not available. Using level 1 model only.")
        
        # Optionally load multiclass model (2+ receipts)
        if self.use_multiclass and multiclass_model_path is not None:
            print(f"Loading multiclass model from {multiclass_model_path}")
            compatible_multiclass_path = ensure_model_compatibility(multiclass_model_path)
            try:
                self.multiclass_model = ModelFactory.load_model(
                    compatible_multiclass_path, 
                    model_type=model_type,
                    num_classes=4,  # 2, 3, 4, 5 receipts
                    mode="eval"
                ).to(self.device)
                self.multiclass_model.eval()
                print("Using full hierarchical model with multiclass classifier")
                
                # Clean up temporary file if created
                if compatible_multiclass_path != multiclass_model_path:
                    try:
                        os.remove(compatible_multiclass_path)
                    except:
                        pass
            except Exception as e:
                print(f"Error loading multiclass model: {e}")
                self.multiclass_model = None
                print("Falling back to simplified model due to loading error.")
        else:
            self.multiclass_model = None
            print("Using simplified hierarchical model (0, 1, 2+)")
    
    def preprocess_image(self, image_path, enhance=True):
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to the image
            enhance: Whether to enhance the image
            
        Returns:
            Tensor ready for model input
        """
        # Enhance image if requested
        if enhance:
            try:
                enhanced_path = "enhanced_temp.jpg"
                self.processor.enhance_scan_quality(image_path, enhanced_path)
                image_path = enhanced_path
            except Exception as e:
                print(f"Warning: Image enhancement failed: {e}")
        
        # Load and preprocess image
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
    
    def predict(self, image_path, enhance=True, return_confidences=False):
        """
        Run hierarchical prediction on an image.
        
        Args:
            image_path: Path to the image
            enhance: Whether to enhance the image before processing
            return_confidences: Whether to return confidence scores
            
        Returns:
            predicted_count: Final receipt count prediction
            If return_confidences=True, also returns a dict of confidences at each level
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path, enhance)
        
        # Level 1: 0 vs 1+ receipts
        with torch.no_grad():
            level1_outputs = self.level1_model(img_tensor)
            if hasattr(level1_outputs, 'logits'):
                level1_logits = level1_outputs.logits
            else:
                level1_logits = level1_outputs
                
            level1_probs = torch.nn.functional.softmax(level1_logits, dim=1)
            
            # Apply Bayesian correction if using lower threshold
            if self.lower_class1_threshold is not None:
                # Extract original probabilities
                orig_prob_class0 = level1_probs[0, 0].item()  # P(class 0)
                orig_prob_class1 = level1_probs[0, 1].item()  # P(class 1) 
                
                # Apply Bayes' theorem to unbias the decision
                # The ratio of thresholds gives us the effective prior odds ratio change
                threshold_ratio = (1 - self.lower_class1_threshold) / self.lower_class1_threshold
                standard_ratio = (1 - self.standard_threshold) / self.standard_threshold
                prior_odds_adjustment = threshold_ratio / standard_ratio
                
                # Apply the correction to get posterior odds
                orig_odds = orig_prob_class1 / (orig_prob_class0 + 1e-10)  # Add epsilon to avoid division by zero
                corrected_odds = orig_odds * prior_odds_adjustment
                
                # Convert back to probability
                corrected_prob_class1 = corrected_odds / (1 + corrected_odds)
                corrected_prob_class0 = 1 - corrected_prob_class1
                
                # Store corrected probabilities
                corrected_probs = torch.tensor([[corrected_prob_class0, corrected_prob_class1]]).to(self.device)
                
                # Make prediction using lower threshold for class 1
                if corrected_prob_class1 >= self.lower_class1_threshold:
                    level1_prediction = 1
                    level1_confidence = corrected_prob_class1
                else:
                    level1_prediction = 0
                    level1_confidence = corrected_prob_class0
                    
                # Store both original and corrected probabilities for reporting
                level1_probs_orig = level1_probs.clone()
                level1_probs = corrected_probs
            else:
                # Standard prediction with threshold 0.5
                level1_prediction = torch.argmax(level1_probs, dim=1).item()
                level1_confidence = level1_probs[0, level1_prediction].item()
        
        # Store confidences at each level
        confidences = {
            'level1': {
                'has_receipts': level1_probs[0, 1].item(),
                'no_receipts': level1_probs[0, 0].item()
            }
        }
        
        # Include original probabilities if threshold was lowered
        if self.lower_class1_threshold is not None:
            confidences['level1_original'] = {
                'has_receipts': level1_probs_orig[0, 1].item(),
                'no_receipts': level1_probs_orig[0, 0].item(),
                'threshold_applied': self.lower_class1_threshold
            }
        
        # If prediction is 0 receipts, we're done
        if level1_prediction == 0:
            if return_confidences:
                return 0, level1_confidence, confidences
            else:
                return 0
        
        # Check if we have a level 2 model
        if self.level2_model is None:
            # If no level 2 model, return 1 (we already know it's not 0 from level 1)
            final_prediction = 1
            final_confidence = level1_confidence
            
            # Store placeholder level 2 confidences
            confidences['level2'] = {
                'one_receipt': 1.0,  # Assume it's 1 receipt with 100% confidence
                'multiple_receipts': 0.0
            }
            
            if return_confidences:
                return final_prediction, final_confidence, confidences
            else:
                return final_prediction
        
        # If we have a level 2 model, use it
        # Level 2: 1 vs 2+ receipts
        with torch.no_grad():
            level2_outputs = self.level2_model(img_tensor)
            if hasattr(level2_outputs, 'logits'):
                level2_logits = level2_outputs.logits
            else:
                level2_logits = level2_outputs
                
            level2_probs = torch.nn.functional.softmax(level2_logits, dim=1)
            level2_prediction = torch.argmax(level2_probs, dim=1).item()
            level2_confidence = level2_probs[0, level2_prediction].item()
        
        # Store level 2 confidences
        confidences['level2'] = {
            'one_receipt': level2_probs[0, 0].item(),
            'multiple_receipts': level2_probs[0, 1].item()
        }
        
        # If prediction is 1 receipt, we're done
        if level2_prediction == 0:  # 0 means class "1 receipt" in level 2
            if return_confidences:
                return 1, level2_confidence, confidences
            else:
                return 1
        
        # If 2+ receipts and we have a multiclass model, use it
        if self.use_multiclass:
            with torch.no_grad():
                multiclass_outputs = self.multiclass_model(img_tensor)
                if hasattr(multiclass_outputs, 'logits'):
                    multiclass_logits = multiclass_outputs.logits
                else:
                    multiclass_logits = multiclass_outputs
                    
                multiclass_probs = torch.nn.functional.softmax(multiclass_logits, dim=1)
                multiclass_prediction = torch.argmax(multiclass_probs, dim=1).item()
                multiclass_confidence = multiclass_probs[0, multiclass_prediction].item()
            
            # Store multiclass confidences
            confidences['multiclass'] = {
                '2_receipts': multiclass_probs[0, 0].item(),
                '3_receipts': multiclass_probs[0, 1].item(),
                '4_receipts': multiclass_probs[0, 2].item(),
                '5_receipts': multiclass_probs[0, 3].item()
            }
            
            # Map multiclass prediction (0-3) back to receipt count (2-5)
            final_prediction = multiclass_prediction + 2
            final_confidence = multiclass_confidence
        else:
            # If no multiclass model, predict 2+
            final_prediction = 2
            final_confidence = level2_confidence
        
        if return_confidences:
            return final_prediction, final_confidence, confidences
        else:
            return final_prediction
    
    def sort_by_count(self, image_paths, output_dir, enhance=True):
        """
        Process a list of images and sort them into folders by receipt count.
        
        Args:
            image_paths: List of paths to images
            output_dir: Base directory to create sorted folders
            enhance: Whether to enhance images before processing
            
        Returns:
            Dictionary with results for each image
        """
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        zero_dir = output_path / "0_receipts"
        one_dir = output_path / "1_receipt"
        multi_dir = output_path / "multiple_receipts"
        
        zero_dir.mkdir(exist_ok=True)
        one_dir.mkdir(exist_ok=True)
        multi_dir.mkdir(exist_ok=True)
        
        if self.use_multiclass:
            # Create subdirectories for each count if using multiclass
            for i in range(2, 6):
                (multi_dir / f"{i}_receipts").mkdir(exist_ok=True)
        
        # Process each image
        results = {}
        for img_path in tqdm(image_paths, desc="Sorting images"):
            try:
                # Get prediction and confidences
                prediction, confidence, confs = self.predict(img_path, enhance, return_confidences=True)
                
                # Get filename
                img_name = Path(img_path).name
                
                # Copy to appropriate folder
                if prediction == 0:
                    shutil.copy(img_path, zero_dir / img_name)
                    dest_folder = "0_receipts"
                elif prediction == 1:
                    shutil.copy(img_path, one_dir / img_name)
                    dest_folder = "1_receipt"
                else:
                    if self.use_multiclass:
                        # Copy to specific count folder
                        count_dir = multi_dir / f"{prediction}_receipts"
                        shutil.copy(img_path, count_dir / img_name)
                        dest_folder = f"multiple_receipts/{prediction}_receipts"
                    else:
                        # Just copy to multiple folder
                        shutil.copy(img_path, multi_dir / img_name)
                        dest_folder = "multiple_receipts"
                
                # Store results
                results[img_name] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "confidences": confs,
                    "destination": dest_folder
                }
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results[Path(img_path).name] = {
                    "error": str(e)
                }
        
        # Save results to CSV
        results_df = pd.DataFrame.from_dict(
            {
                k: {
                    "prediction": v.get("prediction", "error"),
                    "confidence": v.get("confidence", 0),
                    "destination": v.get("destination", "error")
                } 
                for k, v in results.items()
            },
            orient="index"
        )
        
        results_df.index.name = "filename"
        results_df.to_csv(output_path / "sorting_results.csv")
        
        # Print summary
        print("\nSorting complete!")
        print(f"Processed {len(image_paths)} images")
        print(f"0 receipts: {len(list(zero_dir.glob('*')))} images")
        print(f"1 receipt: {len(list(one_dir.glob('*')))} images")
        print(f"Multiple receipts: {len(list(multi_dir.rglob('*.*')))} images")
        
        return results
    
    def evaluate_on_dataset(self, csv_file, image_dir, output_dir=None, enhance=True):
        """
        Evaluate the hierarchical model on a dataset.
        
        Args:
            csv_file: Path to CSV file with filenames and ground truth labels
            image_dir: Directory containing the images
            output_dir: Directory to save evaluation results
            enhance: Whether to enhance images
            
        Returns:
            Dictionary with evaluation metrics
        """
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
        
        # Get device
        device = self.device
        
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
                true_count = row['receipt_count']  # Now comes pre-mapped from our processing above
                
                # Get prediction using hierarchical model - wrap in its own try block
                try:
                    result = self.predict(
                        image_path, 
                        enhance=enhance,
                        return_confidences=True
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
            results_df.to_csv(output_path / 'hierarchical_results.csv', index=False)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=range(6), yticklabels=range(6))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(output_path / 'confusion_matrix.png')
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
        
        # Level 2 performance (1 vs 2+)
        # Filter to only samples with 1+ receipts
        level2_indices = [i for i, t in enumerate(filtered_targets) if t > 0]
        if level2_indices:
            level2_targets = [0 if filtered_targets[i] == 1 else 1 for i in level2_indices]
            level2_preds = [0 if filtered_predictions[i] == 1 else 1 for i in level2_indices]
            level2_accuracy = accuracy_score(level2_targets, level2_preds)
            level2_balanced_accuracy = balanced_accuracy_score(level2_targets, level2_preds)
        else:
            level2_accuracy = 0
            level2_balanced_accuracy = 0
        
        # Multiclass performance (2-5 receipts) - only used with full multiclass model
        # Since we've mapped targets to 0/1/2, we use the original targets when evaluating multiclass
        # but only if multiclass predictions are available
        multiclass_indices = [i for i, t in enumerate(filtered_targets) if t > 1]
        if multiclass_indices and self.use_multiclass:
            # For multiclass evaluation, we need to use the original target values 
            # since we're specifically interested in the detailed counts
            multiclass_targets = [filtered_targets[i] for i in multiclass_indices]
            multiclass_preds = [filtered_predictions[i] for i in multiclass_indices]
            multiclass_accuracy = accuracy_score(multiclass_targets, multiclass_preds)
            multiclass_balanced_accuracy = balanced_accuracy_score(multiclass_targets, multiclass_preds)
        else:
            multiclass_accuracy = 0
            multiclass_balanced_accuracy = 0
        
        # Print hierarchical performance
        print("\nHierarchical Performance:")
        print(f"Level 1 (0 vs 1+): Accuracy={level1_accuracy:.4f}, Balanced Accuracy={level1_balanced_accuracy:.4f}")
        print(f"Level 2 (1 vs 2+): Accuracy={level2_accuracy:.4f}, Balanced Accuracy={level2_balanced_accuracy:.4f}")
        if self.use_multiclass:
            print(f"Multiclass (2-5): Accuracy={multiclass_accuracy:.4f}, Balanced Accuracy={multiclass_balanced_accuracy:.4f}")
        
        # Package all metrics
        if len(filtered_predictions) == 0:
            return {
                "error": "No valid predictions",
                "failed_images": failed_images
            }
        
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
        }
        
        if self.use_multiclass:
            hierarchical_metrics['multiclass'] = {
                'accuracy': multiclass_accuracy,
                'balanced_accuracy': multiclass_balanced_accuracy
            }
        
        if output_dir:
            hierarchical_metrics['results_csv'] = str(output_path / 'hierarchical_results.csv')
        
        return hierarchical_metrics


# Add a class for Level 2 dataset with proper target mapping
class Level2Dataset(Dataset):
    """Dataset specifically for Level 2 (1 vs 2+ receipts) classification"""
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        """
        Initialize a Level 2 dataset.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            augment: Whether to apply data augmentation (used for training)
        """
        from datasets import ReceiptDataset
        
        # Initialize the base dataset
        self.base_dataset = ReceiptDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=transform,
            augment=augment,
            binary=False,
            hierarchical_level=None
        )
        
        # Filter to only include samples with at least 1 receipt
        self.base_dataset.data = self.base_dataset.data[self.base_dataset.data['receipt_count'] > 0].reset_index(drop=True)
        print(f"Level2Dataset contains {len(self.base_dataset.data)} samples after filtering for 1+ receipts")
    
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


def train_level2_model(train_csv, train_dir, val_csv, val_dir, output_dir, model_type="swin", 
                       epochs=20, batch_size=16, learning_rate=2e-4, backbone_lr_multiplier=0.1,
                       weight_decay=0.01, grad_clip=1.0, seed=42, deterministic=True, num_workers=1):
    """
    Train a Level 2 model with balanced sampling.
    
    Args:
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file
        val_dir: Directory containing validation images
        output_dir: Directory to save trained model
        model_type: Type of model to train ("vit" or "swin")
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for classifier head
        backbone_lr_multiplier: Multiplier for backbone learning rate
        weight_decay: Weight decay for optimizer
        grad_clip: Gradient clipping max norm
        seed: Random seed for reproducibility
        deterministic: Whether to enable deterministic mode
        num_workers: Number of dataloader workers (set to 0 or 1 to avoid shared memory issues)
        
    Returns:
        Path to the trained model
    """
    import torch.nn as nn
    import torch.optim as optim
    from training_utils import validate
    from reproducibility import set_seed
    from pathlib import Path
    
    # Set random seed for reproducibility
    set_seed(seed, deterministic)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Loading training data...")
    train_dataset = Level2Dataset(
        csv_file=train_csv,
        img_dir=train_dir,
        augment=True
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
    sampler = WeightedRandomSampler(
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler for balanced training
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print worker count info
    if num_workers <= 1:
        print(f"Using {num_workers} dataloader workers - this should help avoid shared memory (shm) issues")
    else:
        print(f"Using {num_workers} dataloader workers")
    
    # Create model
    print(f"Creating {model_type.upper()} model for binary classification...")
    model = ModelFactory.create_transformer(
        model_type=model_type,
        num_classes=2,  # Binary classification
        pretrained=True
    )
    model = model.to(device)
    
    # Define loss and optimizer with different learning rates for backbone and classifier
    criterion = nn.CrossEntropyLoss()
    
    # Set up parameter groups with different learning rates
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
        {'params': backbone_params, 'lr': learning_rate * backbone_lr_multiplier},
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_balanced_acc = 0.0
    patience_counter = 0
    max_patience = 5
    best_model_path = output_dir / f"receipt_counter_{model_type}_best.pth"
    
    for epoch in range(epochs):
        # Train for one epoch
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")
        
        for images, targets in progress_bar:
            # Move to device
            images = images.to(device=device, dtype=torch.float32)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Handle different model output formats (HF vs. PyTorch)
            if hasattr(outputs, 'logits'):
                # HuggingFace transformer model output
                logits = outputs.logits
                loss = criterion(logits, targets)
                _, predicted = torch.max(logits, 1)
            else:
                # Standard PyTorch model output
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
            
            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({
                'loss': f'{train_loss / (progress_bar.n + 1):.4f}',
                'acc': f'{accuracy:.2f}%'
            })
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate train metrics
        train_accuracy = correct / total if total > 0 else 0
        train_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        train_f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': train_accuracy,
            'balanced_accuracy': train_balanced_acc,
            'f1_macro': train_f1_macro
        }
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1_macro']:.4f}, "
              f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        
        # Save checkpoint if validation accuracy improves
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            
            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics
            }
            
            # Save checkpoint
            torch.save(checkpoint, best_model_path)
            print(f"  New best model saved! Val Accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Create a state-dict-only version for compatibility
    try:
        checkpoint = torch.load(best_model_path, weights_only=False)
        converted_path = output_dir / f"receipt_counter_{model_type}_best_state_dict.pth"
        torch.save(checkpoint['model_state_dict'], converted_path)
        print(f"Created compatible state dict model at: {converted_path}")
        return converted_path
    except:
        # If conversion fails, return the original path
        return best_model_path


def parse_arguments():
    """
    Parse command line arguments for the Hierarchical Predictor.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Hierarchical receipt counting prediction"
    )
    
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
        "--use_multiclass", action="store_true",
        help="Use multiclass model for 2+ receipts instead of just grouping together"
    )
    model_group.add_argument(
        "--level2_model_path",
        help="Custom path to Level 2 model (overrides the default path)"
    )
    model_group.add_argument(
        "--lower_class1_threshold",
        type=float,
        help="Lower the threshold for class 1 in Level 1 model to reduce false negatives (e.g., 0.3 instead of 0.5)"
    )
    
    # Operation mode
    mode_group = parser.add_argument_group('Operation Mode')
    mode_group.add_argument(
        "--mode",
        choices=["predict", "evaluate", "sort", "train_level2"],
        default="predict",
        help="Operation mode: predict single image, evaluate on dataset, sort images, or train Level 2 model"
    )
    
    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        "--image", "-i",
        help="Path to input image for prediction mode"
    )
    input_group.add_argument(
        "--image_dir",
        help="Directory containing images to process for sort mode"
    )
    input_group.add_argument(
        "--test_csv",
        help="Path to CSV file with test data for evaluate mode"
    )
    input_group.add_argument(
        "--test_dir",
        help="Directory containing test images for evaluate mode"
    )
    input_group.add_argument(
        "--train_csv", "-tc",
        help="Path to CSV file with training data for train_level2 mode"
    )
    input_group.add_argument(
        "--train_dir", "-td",
        help="Directory containing training images for train_level2 mode"
    )
    input_group.add_argument(
        "--val_csv", "-vc",
        help="Path to CSV file with validation data for train_level2 mode"
    )
    input_group.add_argument(
        "--val_dir", "-vd",
        help="Directory containing validation images for train_level2 mode"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        "--output_dir", "-o",
        default="results",
        help="Directory to save results"
    )
    
    # Processing options
    processing_group = parser.add_argument_group('Processing')
    processing_group.add_argument(
        "--no-enhance", action="store_true",
        help="Disable image enhancement before processing"
    )
    processing_group.add_argument(
        "--workers", "-w", type=int, default=1,
        help="Number of dataloader workers (default: 1 to avoid shared memory issues)"
    )
    
    # Training options for Level 2 model
    training_group = parser.add_argument_group('Training (for train_level2 mode)')
    training_group.add_argument(
        "--epochs", "-e", type=int, default=20, 
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch_size", "-b", type=int, default=16, 
        help="Batch size for training"
    )
    training_group.add_argument(
        "--learning_rate", "-lr", type=float, default=2e-4, 
        help="Learning rate for classifier head"
    )
    training_group.add_argument(
        "--backbone_lr_multiplier", "-blrm", type=float, default=0.1, 
        help="Multiplier for backbone learning rate"
    )
    training_group.add_argument(
        "--weight_decay", "-wd", type=float, default=0.01, 
        help="Weight decay for optimizer"
    )
    training_group.add_argument(
        "--grad_clip", "-gc", type=float, default=1.0, 
        help="Gradient clipping max norm"
    )
    training_group.add_argument(
        "--seed", "-s", type=int, default=42, 
        help="Random seed for reproducibility"
    )
    training_group.add_argument(
        "--deterministic", "-d", action="store_true", 
        help="Enable deterministic mode"
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the Hierarchical Predictor.
    """
    args = parse_arguments()
    
    # Print information about lowered threshold if set
    if args.lower_class1_threshold is not None:
        if args.lower_class1_threshold <= 0 or args.lower_class1_threshold >= 1:
            raise ValueError("Lower class 1 threshold must be between 0 and 1")
        if args.lower_class1_threshold >= 0.5:
            print("Warning: Lower class 1 threshold should be less than 0.5 to reduce false negatives")
        print(f"\nUsing lowered threshold {args.lower_class1_threshold} for class 1 (has receipts) detection")
        print("This will increase sensitivity to detect receipts, reducing false negatives")
        print("Bayesian correction will be applied to maintain calibration\n")
    
    # Determine model paths
    model_base_path = Path(args.model_base_path)
    level1_model_path = model_base_path / "level1" / f"receipt_counter_{args.model_type}_best.pth"
    
    # Use custom Level 2 model if provided, otherwise use the default path
    if args.level2_model_path:
        level2_model_path = Path(args.level2_model_path)
        print(f"Using custom Level 2 model from: {level2_model_path}")
    else:
        level2_model_path = model_base_path / "level2" / f"receipt_counter_{args.model_type}_best.pth"
    
    # Check if multiclass model exists
    multiclass_model_path = None
    if args.use_multiclass:
        potential_multiclass_path = model_base_path / "multiclass" / f"receipt_counter_{args.model_type}_best.pth"
        if potential_multiclass_path.exists():
            multiclass_model_path = potential_multiclass_path
        else:
            print(f"Warning: Multiclass model not found at {potential_multiclass_path}")
            print("Falling back to simplified hierarchical model (0, 1, 2+)")
    
    # Validate required paths
    if not level1_model_path.exists():
        raise FileNotFoundError(f"Level 1 model not found at {level1_model_path}")
    
    # For level 2, check if it exists; if not, set to None and notify
    if not level2_model_path.exists():
        print(f"Warning: Level 2 model not found at {level2_model_path}")
        print("Running in simplified mode with only Level 1 predictions (0 vs 1+)")
        level2_model_path = None
    
    # Create hierarchical predictor
    predictor = HierarchicalPredictor(
        level1_model_path=level1_model_path,
        level2_model_path=level2_model_path,
        multiclass_model_path=multiclass_model_path,
        model_type=args.model_type,
        lower_class1_threshold=args.lower_class1_threshold
    )
    
    # Handle different modes
    if args.mode == "train_level2":
        # Only use the Level 2 training mode
        if not args.train_csv or not args.train_dir or not args.val_csv or not args.val_dir:
            raise ValueError("Please provide train_csv, train_dir, val_csv, and val_dir for training Level 2 model")
        
        # Train Level 2 model
        print("\nStarting Level 2 model training...")
        print("-----------------------------------")
        model_path = train_level2_model(
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
            num_workers=args.workers
        )
        
        print(f"\nLevel 2 model training complete!")
        print(f"Model saved to: {model_path}")
        print(f"\nTo use this model with the hierarchical predictor, run:")
        print(f"python hierarchical_predictor.py --mode evaluate --test_csv <test_csv> --test_dir <test_dir> --model_base_path {args.model_base_path} --model_type {args.model_type} --level2_model_path {model_path}")
        
        return
    
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("Please provide an image path with --image")
        
        # Run prediction
        predicted_count, confidence, confidences = predictor.predict(
            args.image, 
            enhance=(not args.no_enhance),
            return_confidences=True
        )
        
        print(f"\nPrediction for {args.image}:")
        print(f"Predicted receipt count: {predicted_count}")
        print(f"Confidence: {confidence:.4f}")
        
        # Print detailed confidences
        print("\nConfidence breakdown:")
        print("Level 1 (0 vs 1+):")
        for label, conf in confidences['level1'].items():
            print(f"  {label}: {conf:.4f}")
        
        print("\nLevel 2 (1 vs 2+):")
        for label, conf in confidences['level2'].items():
            print(f"  {label}: {conf:.4f}")
        
        if 'multiclass' in confidences:
            print("\nMulticlass (2-5):")
            for label, conf in confidences['multiclass'].items():
                print(f"  {label}: {conf:.4f}")
        
        # Save a copy of the image to the appropriate folder in output_dir
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if predicted_count == 0:
                folder = output_path / "0_receipts"
            elif predicted_count == 1:
                folder = output_path / "1_receipt"
            else:
                folder = output_path / "multiple_receipts"
                if 'multiclass' in confidences:
                    folder = folder / f"{predicted_count}_receipts"
            
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.image, folder / Path(args.image).name)
            print(f"\nImage saved to {folder / Path(args.image).name}")
            
            # Save confidences to JSON
            import json
            with open(output_path / f"{Path(args.image).stem}_confidences.json", 'w') as f:
                result = {
                    "prediction": predicted_count,
                    "confidence": confidence,
                    "confidences": confidences
                }
                json.dump(result, f, indent=2)
    
    elif args.mode == "evaluate":
        if not args.test_csv or not args.test_dir:
            raise ValueError("Please provide test_csv and test_dir for evaluation mode")
        
        # Run evaluation
        metrics = predictor.evaluate_on_dataset(
            args.test_csv,
            args.test_dir,
            output_dir=args.output_dir,
            enhance=(not args.no_enhance)
        )
        
        # Print overall summary
        print("\nEvaluation Summary:")
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.4f}")
        print(f"F1 Macro: {metrics['overall']['f1_macro']:.4f}")
    
    elif args.mode == "sort":
        if not args.image_dir:
            raise ValueError("Please provide image_dir for sort mode")
        
        # Get list of images to process
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
        
        if not image_paths:
            raise ValueError(f"No images found in {args.image_dir}")
        
        print(f"Found {len(image_paths)} images to process")
        
        # Sort images
        results = predictor.sort_by_count(
            image_paths,
            args.output_dir,
            enhance=(not args.no_enhance)
        )


if __name__ == "__main__":
    main()