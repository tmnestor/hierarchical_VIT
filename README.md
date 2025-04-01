# Hierarchical Vision Transformer with Calibrated Bayesian Correction

A hierarchical image classification system for receipt counting that combines Vision Transformers with temperature calibration and Bayesian correction techniques.

## Project Overview

This project implements a two-level hierarchical classification approach for receipt counting:

1. **Level 1**: Binary classification to determine if there are receipts in the image (0 vs 1+)
2. **Level 2**: Binary classification to determine if there is exactly one receipt or multiple receipts (1 vs 2+)

The key innovation is the use of temperature calibration and Bayesian correction to handle:
- Class imbalance during training
- Confidence calibration for reliable predictions
- Proper probability adjustment during inference

## Recent Updates

- **Enhanced Bayesian Correction**: Implemented comprehensive correction for both weighted loss function bias AND weighted sampling bias during inference
- **Sampling Weights Tracking**: Added explicit tracking of sampling weights in training metadata for more accurate inference correction
- **Code Organization Improvements**: Extracted command-line argument parsing into dedicated functions across multiple modules for better maintainability
- **Parameter Disambiguation**: Clearly differentiated between two different temperature parameters:
  - Confidence calibration temperature (typically >1.0) for scaling logits
  - Bayesian temperature (0.0-1.0) for controlling Bayesian correction strength
- **F1 Score Optimization**: Changed default optimization metric from balanced accuracy to F1 score
- **Device Compatibility**: Fixed device mismatch issues between MPS and CPU operations
- **Error Handling**: Improved error handling for missing keys in metrics dictionaries
- **Path Handling**: Enhanced compatibility with Path objects

## The Problem with Class Imbalance

When working with imbalanced datasets, common techniques include:
- Class weighting during training to emphasize rare classes
- Using weighted samplers to balance class distribution in batches

However, these techniques introduce a deliberate bias in the model's outputs. During inference, the raw model probabilities are influenced by these weights, making them no longer true probabilities.

## Our Solution

Our pipeline solves this problem in three stages:

1. **Training with Metadata**: We record the exact weights and class frequencies used during training in a metadata file
2. **Temperature Calibration**: We calibrate model confidence using temperature scaling on validation data
3. **Bayesian Correction**: We apply a controlled Bayesian correction during inference to counteract training bias, optimized for F1 score by default

According to Bayes' theorem:
```
P(class|data) ∝ P(data|class) × P(class)
```

Our enhanced Bayesian correction implements this by:
1. Dividing raw probabilities by loss function weights to obtain unbiased likelihoods
2. Further correcting for sampling bias introduced by weighted sampling
3. Applying true class frequencies as priors
4. Using temperature parameters to control correction strength

## Pipeline Components

### 1. Training with Bayesian Awareness

```bash
python train_bayesian_hierarchical.py \
    --train_csv receipt_dataset_swinv2/train.csv \
    --train_dir receipt_dataset_swinv2/train \
    --val_csv receipt_dataset_swinv2/val.csv \
    --val_dir receipt_dataset_swinv2/val \
    --model_type swin \
    --output_dir models/bayesian_hierarchical \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --backbone_lr_multiplier 0.1 \
    --look_at_misclassifications \
    --use_focal_loss --focal_gamma 2.0
```

This script:
- Records class weights and frequencies used during training
- Saves this information as metadata alongside trained models
- Implements weighted sampling for handling class imbalance
- Creates separate models for each hierarchy level

### 2. Temperature Calibration and Bayesian Correction

```bash
python bayesian_temperature_calibration.py \
    --calibrate \
    --mode evaluate \
    --model_base_path models/bayesian_hierarchical \
    --model_type swin \
    --test_csv receipt_dataset_swinv2/val.csv \
    --test_dir receipt_dataset_swinv2/val \
    --use-cpu
```

This script:
- Calibrates model confidence using temperature scaling
- Finds optimal calibration temperature parameters for each model
- Applies Bayesian correction during inference
- Supports customizable Bayesian temperature for controlling correction strength

### 3. Finding Optimal Bayesian Temperature with F1 Score Optimization

```bash
# Using the test_bayesian_calibration.py helper script
python test_bayesian_calibration.py \
    --model_type swin \
    --use-cpu \
    --sweep \
    --min_temp 0.001 \
    --max_temp 0.1 \
    --num_temps 5 \
    --optimize_metric f1

# Or using the full bayesian_temperature_calibration.py script
python bayesian_temperature_calibration.py \
    --sweep \
    --model_base_path models/bayesian_hierarchical \
    --model_type swin \
    --test_csv receipt_dataset_swinv2/test.csv \
    --test_dir receipt_dataset_swinv2/test \
    --output_dir evaluation/sweep \
    --min_temp 0.001 \
    --max_temp 0.1 \
    --num_temps 10 \
    --log_scale \
    --optimize_metric f1 \
    --use_best_temp
```

This command:
- Tests multiple Bayesian temperature values for correction
- Identifies the optimal temperature that maximizes F1 score (default) or other metrics
- Generates visualizations of model performance across temperature values
- Saves comprehensive metrics for all temperature settings
- Option to automatically apply the best temperature after sweep

### 4. Running the Complete Pipeline

For convenience, we also provide a simplified helper script for testing and evaluation:

```bash
# Simple test with default parameters
python test_bayesian_calibration.py --model_type swin --use-cpu

# Complete evaluation with F1 score optimization
python test_bayesian_calibration.py \
    --model_type swin \
    --test_csv receipt_dataset_swinv2/test.csv \
    --test_dir receipt_dataset_swinv2/test \
    --sweep \
    --min_temp 0.001 \
    --max_temp 0.1 \
    --num_temps 5 \
    --optimize_metric f1 \
    --use-cpu


python test_bayesian_calibration.py \
    --model_type swin \
    --test_csv receipt_dataset_swinv2/test.csv \
    --test_dir receipt_dataset_swinv2/test \
    --output_dir evaluation/calibrated_hierarchical \
    --mode bayesian_sweep \
    --bayesian_temps 0.001
```


### 5. Conduct Inference on a single image

```bash
python bayesian_temperature_calibration.py \
      --mode predict \
      --model_base_path models/bayesian_hierarchical \
      --model_type swin \
      --image receipt_dataset_swinv2/test/rectangle_0003_3.jpg \
      --bayesian_temperature 0.05 \
      --debug

```

### Analyse Model Errors

```bash
python analyze_model_errors.py --test_csv receipt_dataset_swinv2/test.csv \
                                --test_dir receipt_dataset_swinv2/test \
                                --model_base_path models/bayesian_hierarchical \
                                --model_type swin \
                                --output_dir error_analysis \
                                --use_calibrated
```

1. bayesian_sweep: Tests the model with Bayesian correction using the temperature values provided in
  --bayesian_temps. The script will evaluate the model once for each temperature value.
  2. threshold_sweep: Tests the model with different classification thresholds provided in --level1_thresholds
  and --level2_thresholds. This applies simple threshold adjustments rather than Bayesian correction.
  3. compare: Runs both threshold adjustment and Bayesian correction methods and compares their performance.
  Useful for seeing the difference between the approaches.
  4. all: Runs all of the above - threshold sweep, Bayesian sweep, and comparison analysis.

  In your case, since you want to evaluate specifically with the optimal Bayesian temperature of 0.05, you're
  using --mode bayesian_sweep with --bayesian_temps 0.05.

  If you only provide one value (0.05) for the Bayesian temperature, it will only run evaluation with that
  single value, which is exactly what you want. It's called a "sweep" because it can test multiple values, but
  it works fine with just one value too.

The helper script simplifies the workflow by:
1. Handling model loading and device compatibility
2. Supporting a simple temperature sweep process
3. Finding the optimal Bayesian temperature for F1 score by default
4. Providing a clean interface with sensible defaults
5. Running final evaluation with the best parameters when using --use_best_temp

## Understanding the Two Different Temperature Parameters

This project uses two distinct temperature parameters that should not be confused:

1. **Calibration Temperature** (typically >1.0):
   - Used for confidence calibration with temperature scaling
   - Applies to model logits before softmax
   - Higher values make confidence distributions more uniform
   - Improves reliability of confidence scores
   - Set during the calibration phase

2. **Bayesian Temperature** (range 0.0-1.0):
   - Controls the strength of Bayesian correction
   - Applies after softmax to adjust class probabilities
   - Lower values apply less correction
   - Helps counteract training bias from class weighting
   - Optimized during the temperature sweep phase

## Bayesian Temperature Explained

The Bayesian temperature parameter controls the strength of the correction:

- `bayesian_temperature=0.0`: No correction, uses raw model outputs
- `bayesian_temperature=1.0`: Full correction, completely reverses training weights
- `bayesian_temperature=0.01-0.1`: Typically optimal range, applies mild correction

Finding the right temperature is critical because:
- Too much correction can over-compensate and introduce new biases
- Too little correction doesn't address the training bias
- The optimal value depends on your specific dataset and model

## Results Interpretation

The pipeline generates several key outputs:

1. **Confusion Matrices**: Show classification patterns at each temperature level
2. **Performance Curves**: Visualize metrics vs temperature parameter
3. **Confidence Distributions**: Show how confident the model is in its predictions
4. **Method Comparison**: Compare baseline, calibrated, and confident approaches

## Usage Guidelines

- **For new datasets**: Run the complete pipeline with temperature sweep and F1 optimization
- **For quick testing**: Use test_bayesian_calibration.py with default parameters and --use-cpu flag
- **For production**: Use the identified optimal temperature from F1-optimized sweep results
- **For maximum reliability**: Consider the confident predictor approach which can abstain from low-confidence predictions
- **For Apple Silicon Macs**: Always include the --use-cpu flag to avoid device mismatch errors

## Mathematical Foundation

Our enhanced Bayesian correction applies the following formulas:

1. Loss function bias correction:
```
P_loss_corrected(data|class) = P_model(data|class) / loss_weights
```

2. Sampling bias correction:
```
P_sampling_corrected(data|class) = P_loss_corrected(data|class) / sampling_weights
```

3. Application of class frequency priors:
```
P(class|data) ∝ P_sampling_corrected(data|class) × frequencies
```

The Bayesian temperature parameter interpolates between uniform and corrected probabilities at each step:

```
P_temp(class|data) = (1-t) × uniform + t × P(class|data)
```

Where:
- `loss_weights` are the class weights used in the loss function during training
- `sampling_weights` are the weights used for weighted random sampling during training
- `frequencies` are the true class frequencies in training data
- `t` is the Bayesian temperature parameter (0.0-1.0)

By combining confidence calibration with this comprehensive Bayesian correction approach, we achieve well-calibrated confidence scores, accurate class boundaries, and properly unbiased predictions that address both forms of training bias.

## Optimize for F1 Score vs Balanced Accuracy

The system now optimizes for F1 score by default, but you can choose different metrics:

```bash
# Optimize for F1 score (default)
python bayesian_temperature_calibration.py --sweep --optimize_metric f1

# Optimize for balanced accuracy
python bayesian_temperature_calibration.py --sweep --optimize_metric balanced_accuracy  

# Optimize for standard accuracy
python bayesian_temperature_calibration.py --sweep --optimize_metric accuracy
```

F1 score optimization is particularly valuable when both precision and recall are important and your dataset has class imbalance. After finding the optimal temperature for your chosen metric, you can automatically apply it using the `--use_best_temp` flag.

## Technical Notes

- For Apple Silicon (M1/M2) Macs, use the `--use-cpu` flag to avoid device mismatch errors
- The system supports both image and directory modes for batch processing 
- All results include detailed metrics for each hierarchical level (level1 and level2)
- For deeper insight, examine the confidence distributions in the output JSON files
- The codebase has been refactored to follow best practices with dedicated argument parsing functions



## Latest Updates

### New Additions (March 2025)

- **Batch Processing System**: Added `batch_processor.py` for efficient batch processing of image directories
- **Simple Rectangle Dataset Creator**: Added `create_simple_rectangle_dataset.py` for generating synthetic datasets
- **Enhanced Temperature Calibration**: Improved Bayesian temperature calibration with weighted sampling correction
- **Device Compatibility Fixes**: Fixed MPS and CPU compatibility issues for Apple Silicon Macs
- **Refactored CLI Arguments**: Standardized command-line argument parsing across all scripts
- **Improved Error Handling**: More robust error handling for missing files and invalid parameters
- **Documentation Updates**: Extended README with detailed mathematical explanations and usage guidelines
- **Performance Optimization**: Reduced memory usage during training and evaluation

### Bayesian Temperature Calibration Usage

```bash
# Quick test with default parameters (uses rectangle_dataset for testing)
python test_bayesian_calibration.py --model_type swin

# Run Bayesian temperature sweep 
python test_bayesian_calibration.py --model_type swin \
    --mode bayesian_sweep \
    --bayesian_temps 0.001,0.01,0.05,0.1,0.5

# Test with custom dataset
python test_bayesian_calibration.py --model_type swin \
    --test_csv receipt_dataset/test.csv \
    --test_dir receipt_dataset/test \
    --mode bayesian_sweep

# Compare different methods (threshold vs Bayesian)
python test_bayesian_calibration.py --model_type swin \
    --mode compare \
    --output_dir evaluation/calibration_comparison

# Run all test modes with custom parameters
python test_bayesian_calibration.py --model_type swin \
    --mode all \
    --level1_thresholds 0.3,0.5,0.7 \
    --level2_thresholds 0.3,0.5,0.7 \
    --bayesian_temps 0.001,0.01,0.1
```

> **Note for Apple Silicon Mac Users**: The code now automatically detects Apple Silicon and forces CPU mode to prevent device mismatch errors. We've also fixed string formatting errors in the results display and initialized missing temperature attributes.

## Python Modules (Core Files)

### Core Architecture

- model_factory.py - Central factory for model creation and loading
- transformer_vit.py - Vision Transformer implementation
- transformer_swin.py - Swin Transformer implementation
- datasets.py - Dataset handling for receipt images
- training_utils.py - Shared training utilities
- evaluation.py - Evaluation metrics and functions
- device_utils.py - Device abstraction (CPU/GPU/MPS)
- reproducibility.py - Seed and deterministic settings
- config.py - Configuration management

### Hierarchical Model Components

- hierarchical_predictor.py - Base hierarchical prediction
- train_bayesian_hierarchical.py - Training with Bayesian awareness
- bayesian_temperature_calibration.py - Temperature calibration and Bayesian correction
- test_bayesian_calibration.py - Helper script for testing calibration
- train_hierarchical_model.py - Training hierarchical models
- run_calibration_pipeline.py - Full calibration pipeline

### Training Scripts

- train_vit_classification.py - Train ViT models
- train_swin_classification.py - Train Swin models
- temperature_calibration.py - Base temperature scaling

### Utilities

- batch_processor.py - Batch processing utilities
- receipt_processor.py - Receipt-specific processing
- hierarchical_demo.py - Visual demo for hierarchical models
- create_simple_rectangle_dataset.py - Generate synthetic rectangle datasets

### Data Directories

Dataset Structure

- receipt_dataset/ or rectangle_dataset/ - Main dataset directory
  - train/ - Training images
  - val/ - Validation images
  - test/ - Test images
  - train.csv - Training labels
  - val.csv - Validation labels
  - test.csv - Test labels

### Testing Data

- receipt_collages/ - Generated collages for testing
- test_images/ - Test images directory

### Model Directories

Model Storage

- models/ - Parent directory for all models
  - models/bayesian_hierarchical/ - Bayesian hierarchical models
      - level1/ - Level 1 model files
    - level2/ - Level 2 model files
    - training_metadata.json - Metadata with class weights and frequencies
    - calibration_params.json - Temperature calibration parameters

### Evaluation Outputs

- evaluation/ - Root evaluation directory
  - evaluation/sweep/ - Temperature sweep results
  - evaluation/hierarchical/ - Hierarchical model evaluation
  - evaluation/calibrated_hierarchical/ - Calibrated model evaluation


python train_bayesian_hierarchical.py \
    --train_csv rectangle_dataset/train.csv \
    --train_dir rectangle_dataset/train \
    --val_csv rectangle_dataset/val.csv \
    --val_dir rectangle_dataset/val \
    --model_type swin \
    --output_dir models/bayesian_hierarchical \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --backbone_lr_multiplier 0.1 \
    --look_at_misclassifications \
    --use_focal_loss --focal_gamma 2.0

### SWIN2

python train_swin_classification.py --offline \
  --train_csv receipt_dataset_swinv2/train.csv \
  --train_dir receipt_dataset_swinv2/train \
  --val_csv receipt_dataset_swinv2/val.csv \
  --val_dir receipt_dataset_swinv2/val \
  --output_dir models/swinv2 \
  --epochs 20 \
  --batch_size 16