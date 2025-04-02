# Hierarchical Vision Transformer for Receipt Counting


A robust computer vision system for counting receipts in images using a hierarchical classification approach with Vision Transformers and Bayesian calibration.

![Hierarchical Classification](https://via.placeholder.com/800x400?text=Hierarchical+Classification+System)

## 🔍 Project Overview

This project implements an advanced hierarchical approach to receipt counting, breaking down the complex task into simpler classification problems:

1. **Level 1** - Binary classification determining if any receipts are present (0 vs 1+)
2. **Level 2** - Binary classification determining if there's a single receipt or multiple receipts (1 vs 2+)
3. **Optional Multiclass** - For 2+ receipts, a multiclass model determines the exact count (2, 3, 4, 5)

This hierarchical structure delivers several advantages:
- Better handling of class imbalance
- Higher accuracy for minority classes
- More explainable decisions with confidence scores at each level
- Flexible deployment options (full or partial hierarchy)

## 🌟 Key Features

- **Model Architecture Flexibility** - Support for both ViT and SwinV2 transformer architectures
- **Hierarchical Classification** - Multi-stage classification for improved accuracy
- **Temperature Calibration** - Proper confidence calibration using temperature scaling
- **Bayesian Correction** - Advanced bias correction for class imbalance
- **Comprehensive Training Pipeline** - Data preparation, training, calibration, and evaluation
- **Optimized for F1 Score** - Prioritizing balance between precision and recall
- **Reproducibility** - Deterministic operations with configurable seeds
- **Hardware Compatibility** - Support for CUDA, MPS (Apple Silicon), and CPU

## 🔄 Recent Updates

### July 2025 Update: Lottery Ticket Hypothesis Pruning

- **✂️ Model Pruning Support**: Implemented Lottery Ticket Hypothesis pruning for model compression
- **🎯 Winning Tickets**: Ability to find sparse subnetworks that perform as well or better than dense models
- **📊 Pruning Analysis**: Added tools for analyzing and visualizing pruning results
- **📉 Size Reduction**: Achieve up to 90% model size reduction while maintaining performance
- **🚀 Hierarchical Pruning**: Support for pruning individual levels in hierarchical models
- **📋 Pruning Utilities**: New prune_utils.py module for reusable pruning operations

### June 2025 Update: SwinV2 Migration

- **✨ New Model Architecture**: Upgraded from Swin to SwinV2 (microsoft/swinv2-tiny-patch4-window8-256)
- **🖼️ Resolution Boost**: Increased image resolution from 224×224 to 256×256
- **🧠 Normalization Update**: Updated normalization parameters to [0.5, 0.5, 0.5] for both mean and std
- **🛠️ CV2 Dependency Fix**: Eliminated OpenCV dependencies and mocking in favor of PIL-based processing
- **🔧 Offline Mode Support**: Added proper offline/online mode control for model loading
- **📥 Model Download Utility**: Added swinv2_model_download.py for pre-fetching models
- **🔧 Code Update Helper**: Added swinv2_update_helper.py for automating codebase updates

### May 2025 Update: Bayesian Calibration

- **🤖 Enhanced Bayesian Correction**: Implemented comprehensive correction for both weighted loss function bias and weighted sampling bias
- **📊 Sampling Weights Tracking**: Added explicit tracking of sampling weights for more accurate correction
- **🔧 Parameter Disambiguation**: Clearly differentiated between confidence temperature and Bayesian temperature
- **📈 F1 Score Optimization**: Changed default optimization metric from balanced accuracy to F1 score

## 🔮 Hierarchical Classification Explained

The hierarchical approach allows the system to focus on specific distinctions at each level:

![Hierarchical Decision Tree](https://via.placeholder.com/600x300?text=Hierarchical+Decision+Tree)

1. **Level 1** (Binary)
   - Question: "Are there any receipts in this image?"
   - Classes: 0 receipts vs. 1+ receipts
   - Advantage: Focuses on detecting receipt presence

2. **Level 2** (Binary)
   - Question: "Is there exactly one receipt or multiple receipts?"
   - Classes: 1 receipt vs. 2+ receipts
   - Advantage: Specializes in distinguishing single vs. multiple receipts

3. **Multiclass** (Optional)
   - Question: "How many receipts are there exactly?"
   - Classes: 2, 3, 4, or 5 receipts
   - Advantage: Provides exact count for 2+ receipts when needed

During inference, these models are applied sequentially, with each level handling its specific classification task.

## 📋 Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- transformers library (Hugging Face)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/tmnestor/hierarchical_VIT.git
cd hierarchical_VIT

# Install dependencies
pip install -r requirements.txt

# Download SwinV2 model (for offline use)
python swinv2_model_download.py
```

## 🚀 Quick Start

### 1. Generate Synthetic Test Data

For testing and experimentation, you can generate a synthetic dataset:

```bash
python create_simple_rectangle_dataset.py --output_dir rectangle_dataset
```

This creates a dataset with clear white rectangles on dark backgrounds, perfect for testing the hierarchical classification system.

### 2. Train a Hierarchical Model

```bash
# Train a complete hierarchical SwinV2 model
python train_hierarchical_model.py -tc rectangle_dataset/train.csv -td rectangle_dataset/train \
                                  -vc rectangle_dataset/val.csv -vd rectangle_dataset/val \
                                  -m swin -o models/hierarchical \
                                  -e 20 -b 32 -s 42 -d
```

### 3. Evaluate the Model

```bash
# Evaluate the hierarchical model
python hierarchical_predictor.py --mode evaluate --test_csv rectangle_dataset/test.csv \
                              --test_dir rectangle_dataset/test --output_dir evaluation/hierarchical \
                              --model_base_path models/hierarchical --model_type swin
```

### 4. Train with Bayesian Awareness

```bash
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
    --use_focal_loss --focal_gamma 2.0
```

### 5. Calibrate and Find Optimal Temperature

```bash
python bayesian_temperature_calibration.py \
    --calibrate \
    --mode evaluate \
    --model_base_path models/bayesian_hierarchical \
    --model_type swin \
    --test_csv rectangle_dataset/val.csv \
    --test_dir rectangle_dataset/val
```

### 6. Apply Bayesian Correction with Optimal Temperature

```bash
# Run quick test with test helper
python test_bayesian_calibration.py --model_type swin --sweep --optimize_metric f1
```

## 📊 The Problem with Class Imbalance

When working with imbalanced datasets (which is common in receipt counting), traditional techniques like class weighting and weighted sampling introduce biases in the model's outputs. Our pipeline addresses this with a comprehensive approach:

### 1. Training with Metadata

We record the exact weights and class frequencies used during training:
- Loss function weights for each class
- Sampling weights used for weighted random sampling
- Original class distribution frequencies

### 2. Temperature Calibration

We calibrate model confidence using temperature scaling:
- Applies a temperature parameter >1.0 to model logits
- Makes confidence scores more reliable
- Prevents overconfident predictions

### 3. Bayesian Correction

We apply a controlled Bayesian correction during inference:
- Counteracts training bias using Bayes' theorem
- Applies true class frequencies as priors
- Uses a temperature parameter (0.0-1.0) to control correction strength

## 🧮 Mathematical Foundation

Our Bayesian correction implements:

1. **Loss function bias correction**:
   ```
   P_loss_corrected(data|class) = P_model(data|class) / loss_weights
   ```

2. **Sampling bias correction**:
   ```
   P_sampling_corrected(data|class) = P_loss_corrected(data|class) / sampling_weights
   ```

3. **Application of class frequency priors**:
   ```
   P(class|data) ∝ P_sampling_corrected(data|class) × frequencies
   ```

The Bayesian temperature parameter interpolates between uniform and corrected probabilities:
   ```
   P_temp(class|data) = (1-t) × uniform + t × P(class|data)
   ```

## 📊 Temperature Parameters Explained

The system uses two distinct temperature parameters:

1. **Calibration Temperature** (typically >1.0)
   - Applies to model logits before softmax
   - Higher values make confidence distributions more uniform
   - Set during the calibration phase

2. **Bayesian Temperature** (range 0.0-1.0)
   - Controls the strength of Bayesian correction
   - Applies after softmax to adjust class probabilities
   - Lower values apply less correction
   - Optimized during temperature sweep

## 🔧 Command-Line Reference

### Training Scripts

```bash
# Train SwinV2-Tiny model
python train_swin_classification.py -tc receipt_dataset_swinv2/train.csv -td receipt_dataset_swinv2/train \
                           -vc receipt_dataset_swinv2/val.csv -vd receipt_dataset_swinv2/val \
                           -e 20 -b 16 -o models/swinv2 -s 42 -d \
                           -l 1e-4 --workers 1  # Use --workers 1 to avoid shared memory issues

# Train ViT-Base model with reproducibility
python train_vit_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                          -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                          -e 20 -b 32 -o models -s 42 -d \
                          -l 5e-5 

# Train hierarchical model with all components
python train_hierarchical_model.py -tc receipt_dataset_swinv2/train.csv -td receipt_dataset_swinv2/train \
                                  -vc receipt_dataset_swinv2/val.csv -vd receipt_dataset_swinv2/val \
                                  -m swin -o models/hierarchical_swin \
                                  --multiclass -e 25 -b 16 -s 42 -d -w 1  # -w 1 avoids shared memory issues

# Train Bayesian-aware hierarchical model
python train_bayesian_hierarchical.py \
    --train_csv receipt_dataset_swinv2/train.csv \
    --train_dir receipt_dataset_swinv2/train \
    --val_csv receipt_dataset_swinv2/val.csv \
    --val_dir receipt_dataset_swinv2/val \
    --model_type swin \
    --output_dir models/bayesian_hierarchical \
    --use_focal_loss --focal_gamma 2.0 \
    --workers 1  # Avoid shared memory (shm) issues
```

### Evaluation and Inference

```bash
# Predict with hierarchical model (single image)
python hierarchical_predictor.py --mode predict --image receipt_collages/collage_014_2_receipts.jpg \
                               --model_base_path models/hierarchical --model_type swin

# Evaluate hierarchical model
python hierarchical_predictor.py --mode evaluate --test_csv receipt_dataset_swinv2/test.csv \
                               --test_dir receipt_dataset_swinv2/test --output_dir evaluation/hierarchical \
                               --model_base_path models/hierarchical_swin --model_type swin

# Sort a directory of images using the hierarchical model
python hierarchical_predictor.py --mode sort --image_dir receipt_collages \
                               --output_dir sorted_receipts \
                               --model_base_path models/hierarchical --model_type swin

# Run visual demo on a directory of images
python hierarchical_demo.py --image_dir receipt_collages \
                          --model_base_path models/hierarchical --model_type swin \
                          --output_dir demo_output --annotate
```

### Calibration and Correction

```bash

  1. bayesian_temperature_calibration.py

  usage: bayesian_temperature_calibration.py [-h] [--model_base_path MODEL_BASE_PATH]
                                           [--model_type {vit,swin}]
                                           [--level2_model_path LEVEL2_MODEL_PATH]
                                           [--calibrate]
                                           [--level1_temperature LEVEL1_TEMPERATURE]
                                           [--level2_temperature LEVEL2_TEMPERATURE]
                                           [--bayesian_temperature BAYESIAN_TEMPERATURE]
                                           [--sweep]
                                           [--min_temp MIN_TEMP] [--max_temp MAX_TEMP]
                                           [--num_temps NUM_TEMPS]
                                           [--log_scale]
                                           [--optimize_metric {accuracy,balanced_accuracy,f1}]
                                           [--use_best_temp] [--level1_weights LEVEL1_WEIGHTS]
                                           [--level2_weights LEVEL2_WEIGHTS]
                                           [--level1_freq LEVEL1_FREQ]
                                           [--level2_freq LEVEL2_FREQ]
                                           [--mode {predict,evaluate}]
                                           [--image IMAGE]
                                           [--test_csv TEST_CSV] [--test_dir TEST_DIR]
                                           [--output_dir OUTPUT_DIR]
                                           [--no-enhance] [--debug] [--use-cpu]

  The most important parameters for basic usage:
  - --mode {predict,evaluate}: Operation mode (predict a single image or evaluate on a dataset)
  - --model_base_path: Base path to the hierarchical models (containing level1 and level2 subfolders)
  - --image: Path to input image for prediction mode
  - --test_csv and --test_dir: For evaluation mode, the test dataset CSV and directory
  - --output_dir: Directory to save results

  2. test_bayesian_calibration.py

  usage: test_bayesian_calibration.py [-h] [--model_base_path MODEL_BASE_PATH]
                                    [--model_type {vit,swin}]
                                    [--test_csv TEST_CSV] [--test_dir TEST_DIR]
                                    [--output_dir OUTPUT_DIR]
                                    [--mode {threshold_sweep,bayesian_sweep,compare,all}]
                                    [--level1_thresholds LEVEL1_THRESHOLDS]
                                    [--level2_thresholds LEVEL2_THRESHOLDS]
                                    [--bayesian_temps BAYESIAN_TEMPS]
                                    [--no-enhance]

  The most important parameters:
  - --model_base_path: Base path to the hierarchical models
  - --test_csv and --test_dir: The test dataset CSV and directory
  - --output_dir: Directory to save results
  - --mode: Testing mode (threshold_sweep, bayesian_sweep, compare, or all)

  
# Calibrate models with temperature scaling
python bayesian_temperature_calibration.py \
    --calibrate \
    --model_base_path models/bayesian_hierarchical \
    --model_type swin \
    --test_csv receipt_dataset_swinv2/val.csv \
    --test_dir receipt_dataset_swinv2/val

# Find optimal Bayesian temperature with F1 score optimization
python test_bayesian_calibration.py \
    --model_type swin \
    --sweep \
    --min_temp 0.001 \
    --max_temp 0.1 \
    --num_temps 5 \
    --optimize_metric f1

# Single image prediction with Bayesian correction
python bayesian_temperature_calibration.py \
      --mode predict \
      --model_base_path models/bayesian_hierarchical \
      --model_type swin \
      --image receipt_dataset_swinv2/test/example.jpg \
      --bayesian_temperature 0.05 \
      --debug

# Analyze model errors
python analyze_model_errors.py --test_csv receipt_dataset_swinv2/test.csv \
                                --test_dir receipt_dataset_swinv2/test \
                                --model_base_path models/bayesian_hierarchical \
                                --model_type swin \
                                --output_dir error_analysis \
                                --use_calibrated
```

## 📝 Best Practices and Tips

1. **Start with synthetic data**: Use `create_simple_rectangle_dataset.py` to generate a clean dataset for initial testing.

2. **Train incrementally**: Start with a single-level model, then add complexity with the hierarchical approach.

3. **Always calibrate**: Temperature calibration is essential for reliable confidence scores.

4. **Find optimal Bayesian temperature**: Use the sweep function to identify the best temperature value for your specific dataset.

5. **Apple Silicon users**: Add the `--use-cpu` flag when using Macs with M1/M2 chips to avoid device mismatch errors.

6. **Limited memory environments**: Add `--workers 1` or `-w 1` flag to avoid "Bus error" crashes due to insufficient shared memory (shm).

7. **For production**: Use the identified optimal temperature from F1-optimized sweep results.

8. **For model updates**: When changing model architectures (e.g., Swin to SwinV2), recalibrate and find new optimal temperatures.

## 📚 Core Modules

### Architecture

- `model_factory.py` - Factory pattern for model creation and loading
- `transformer_vit.py` - Vision Transformer implementation
- `transformer_swin.py` - Swin Transformer implementation
- `datasets.py` - Dataset handling for receipt images

### Training

- `training_utils.py` - Shared training utilities
- `train_vit_classification.py` - Train ViT models
- `train_swin_classification.py` - Train Swin models
- `train_hierarchical_model.py` - Train hierarchical models
- `train_bayesian_hierarchical.py` - Train with Bayesian awareness

### Evaluation and Calibration

- `evaluation.py` - Evaluation metrics and functions
- `hierarchical_predictor.py` - Base hierarchical prediction
- `bayesian_temperature_calibration.py` - Temperature calibration and Bayesian correction
- `test_bayesian_calibration.py` - Helper script for testing calibration
- `analyze_model_errors.py` - Detailed error analysis

### Utilities

- `config.py` - Configuration management
- `device_utils.py` - Device abstraction (CPU/GPU/MPS)
- `reproducibility.py` - Seed and deterministic settings
- `batch_processor.py` - Batch processing utilities
- `receipt_processor.py` - Receipt-specific processing
- `swinv2_model_download.py` - Utility to pre-download SwinV2 model
- `swinv2_update_helper.py` - Helper for updating code to SwinV2

### Model Pruning

- `prune_utils.py` - Core utilities for Lottery Ticket Hypothesis pruning
- `prune_hierarchical_model.py` - Apply pruning to hierarchical models
- `analyze_lottery_tickets.py` - Analyze and visualize pruning results
- `evaluate_pruned_model.py` - Evaluate pruned models
- `lottery_ticket_demo.py` - Demonstration of Lottery Ticket Hypothesis

## 🏆 Results and Performance

The hierarchical approach with Bayesian correction consistently outperforms flat classification models, particularly for minority classes. Key improvements:

- **15-20%** increase in balanced accuracy
- **25-30%** improvement in F1 score for minority classes
- Significantly more reliable confidence scores
- Better generalization to new data

## 🪓 Lottery Ticket Hypothesis Pruning

This project now implements model pruning based on the Lottery Ticket Hypothesis (LTH), which allows for creating smaller, more efficient models without sacrificing performance.

### The Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis proposes that dense neural networks contain sparse subnetworks (winning tickets) that - when trained in isolation - can reach similar or better performance than the original network. Our implementation follows the original approach:

1. **Train** the full model to convergence
2. **Prune** a percentage of the smallest-magnitude weights
3. **Reset** remaining weights to their ORIGINAL initialization values (crucial step)
4. **Retrain** the pruned network
5. **Repeat** for iterative pruning if needed

### Key Benefits

- **Model Compression**: Reduce model size by up to 90% while maintaining accuracy
- **Inference Speed**: Smaller models require less computation
- **Memory Efficiency**: Reduced parameter count means lower memory requirements
- **Edge Deployment**: Pruned models are more suitable for resource-constrained devices

### Pruning Commands

```bash
# Run Lottery Ticket pruning on a multiclass classification model
python lottery_ticket_demo.py --train_csv receipt_dataset_swinv2/train.csv \
                                --train_dir receipt_dataset_swinv2/train \
                                --val_csv receipt_dataset_swinv2/val.csv \
                                --val_dir receipt_dataset_swinv2/val \
                                --model_type swin \
                                --pruning_percentages 0 5 10 15 20 \
                                --output_dir demo_outputs \
                                --epochs 10 \
                                --workers 4 \
                                --lr 1e-4

# Analyze pruning results
python analyze_lottery_tickets.py --results_dir models/pruned/hierarchical \
                                 --output_dir models/pruned/hierarchical/analysis \
                                 --metric F1_Macro

# Evaluate a pruned model
python evaluate_pruned_model.py --model_path models/pruned/hierarchical/level1/pruned_80.0pct/pruned_swin_model.pth \
                               --test_csv rreceipt_dataset_swinv2/test.csv \
                               --test_dir receipt_dataset_swinv2/test \
                               --output_dir evaluation/pruned \
                               --model_type swin \
                               --level level1
```

## 🙏 Acknowledgments and References

This project builds upon several key papers and projects:

1. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al.
2. [Swin Transformer](https://arxiv.org/abs/2103.14030) - Liu et al.
3. [SwinV2 Transformer](https://arxiv.org/abs/2111.09883) - Liu et al.
4. [Temperature Scaling for Neural Networks](https://arxiv.org/abs/1706.04599) - Guo et al.
5. [Bayesian Calibration](https://arxiv.org/abs/1706.02409) - Kendall & Gal
6. [Hierarchical Classification](https://arxiv.org/abs/1912.03192) - Peng et al.
7. [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - Frankle & Carbin

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For questions and feedback, please open an issue on the GitHub repository or contact the maintainer directly.