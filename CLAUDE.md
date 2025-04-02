# Vision Transformer Receipt Counter Project

## Important Instructions
- Always ask before running any model training or inference commands
- The user can run models for free, so let them handle the actual execution of training/evaluation
- Always clean up temporary test directories after testing is completed
- Never create directories for testing purposes without deleting them afterward

## Hierarchical Classification

This project now implements a hierarchical classification approach that breaks down the receipt counting task into two distinct levels:

1. **Level 1**: Binary classification to determine if there are receipts in the image (0 vs 1+)
2. **Level 2**: Binary classification to determine if there is exactly one receipt or multiple receipts (1 vs 2+)
3. **Optional Multiclass**: If more granularity is needed, a third multiclass model can determine exact counts for 2+ receipts

This hierarchical approach helps address class imbalance and allows for specialized models at each level. The system can be used to classify images into three main categories:
- No receipts (0)
- Single receipt (1)
- Multiple receipts (2+)

### Hierarchical Commands

```bash
# Train a complete hierarchical model (VIT)
python train_hierarchical_model.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                                  -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                                  -m vit -o models/hierarchical \
                                  -e 20 -b 32 -s 42 -d

# Train a hierarchical model with multiclass component (Swin)
python train_hierarchical_model.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                                  -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                                  -m swin -o models/hierarchical_swin \
                                  --use_multiclass -e 25 -b 16 -s 42 -d

# Predict with hierarchical model (single image)
python hierarchical_predictor.py --mode predict --image receipt_collages/collage_014_2_receipts.jpg \
                               --model_base_path models/hierarchical --model_type vit

# Evaluate hierarchical model
python hierarchical_predictor.py --mode evaluate --test_csv receipt_dataset/test.csv \
                               --test_dir receipt_dataset/test --output_dir evaluation/hierarchical \
                               --model_base_path models/hierarchical --model_type vit

# Sort a directory of images using the hierarchical model
python hierarchical_predictor.py --mode sort --image_dir receipt_collages \
                               --output_dir sorted_receipts \
                               --model_base_path models/hierarchical --model_type vit

# Run visual demo on a directory of images
python hierarchical_demo.py --image_dir receipt_collages \
                          --model_base_path models/hierarchical --model_type vit \
                          --output_dir demo_output --annotate
```

## Key Commands

### Training Commands

```bash
# Train ViT-Base model with reproducibility
python train_vit_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                          -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                          -e 20 -b 32 -o models -s 42 -d \
                          -l 5e-5 

# Train Swin-Tiny model with reproducibility
python train_swin_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                           -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                           -e 20 -b 32 -o models -s 42 -d \
                           -l 5e-5

# Resume training from a checkpoint
python train_vit_classification.py -r models/receipt_counter_vit_best.pth \
                          -e 10 -b 32 -o models

# Dry run to validate configuration
python train_swin_classification.py --dry-run -e 30 -b 16 -s 42
```

### CLI Options

The training scripts now have an improved CLI with argument groups:

#### Data Options
- `-tc/--train_csv`: Path to training CSV file
- `-td/--train_dir`: Directory containing training images
- `-vc/--val_csv`: Path to validation CSV file
- `-vd/--val_dir`: Directory containing validation images
- `--no-augment`: Disable data augmentation during training

#### Training Options
- `-e/--epochs`: Number of training epochs
- `-b/--batch_size`: Batch size for training
- `-l/--lr`: Learning rate for classifier head
- `-blrm/--backbone_lr_multiplier`: Multiplier for backbone learning rate (default: 0.1)
- `-gc/--grad_clip`: Gradient clipping max norm (default: 1.0)
- `-wd/--weight_decay`: Weight decay for optimizer
- `-ls/--label_smoothing`: Label smoothing factor
- `-o/--output_dir`: Directory to save trained model
- `-c/--config`: Path to configuration JSON file
- `-r/--resume`: Resume training from checkpoint file
- `-bin/--binary`: Train as binary classification
- `--dry-run`: Validate configuration without training
- `--class_dist`: Comma-separated class distribution

#### Reproducibility Options
- `-s/--seed`: Random seed for reproducibility
- `-d/--deterministic`: Enable deterministic mode

### Evaluation Commands

```bash
# Evaluate ViT model
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth \
                             --test_csv receipt_dataset/val.csv \
                             --test_dir receipt_dataset/val \
                             --output_dir evaluation/vit_base

# Evaluate Swin model
python evaluate_swin_counter.py --model models/receipt_counter_swin_best.pth \
                                 --test_csv receipt_dataset/val.csv \
                                 --test_dir receipt_dataset/val \
                                 --output_dir evaluation/swin_tiny
```

### Testing on Individual Images

```bash
# Test a single image with ViT model
python individual_image_tester.py --image receipt_collages/collage_014_2_receipts.jpg --model models/receipt_counter_vit_best.pth

# Test a single image with Swin model
python individual_image_tester.py --image receipt_collages/collage_014_2_receipts.jpg --model models/receipt_counter_swin_best.pth
```

### Testing on Multiple Images

```bash
# Show sample images
python test_images_demo.py --image_dir receipt_collages --samples 4 --mode show

# Process sample images with Swin model
python test_images_demo.py --image_dir receipt_collages --samples 4 --mode process --model models/receipt_counter_swin_best.pth
```

## Project Structure

### Core Modules:
- `model_factory.py` - Factory pattern for creating and loading models
- `datasets.py` - Unified dataset implementation 
- `training_utils.py` - Shared training, validation, and evaluation utilities
- `config.py` - Centralized configuration system
- `device_utils.py` - Device abstraction for hardware acceleration 
- `evaluation.py` - Unified evaluation functionality
- `reproducibility.py` - Seed control and deterministic behavior

### Training Scripts:
- `train_vit_classification.py` - Train the ViT-Base model
- `train_swin_classification.py` - Train the Swin-Tiny model
- `train_hierarchical_model.py` - Train hierarchical models for improved receipt counting

### Evaluation Scripts:
- `hierarchical_predictor.py` - Predict, evaluate and sort using hierarchical models

### Testing Scripts:
- `hierarchical_predictor.py` - Supports multiple modes for hierarchical model testing
- `hierarchical_demo.py` - Visual demonstration of hierarchical prediction

### Legacy Scripts (in legacy/ directory):
- `evaluate_vit_counter.py` - Original ViT model evaluation
- `evaluate_swin_counter.py` - Original Swin model evaluation
- `individual_image_tester.py` - Original single image tester
- `test_images_demo.py` - Original multiple image tester

### Data Generation:
- `create_receipt_collages.py` - Generate synthetic receipt collages
- `prepare_collage_dataset.py` - Prepare datasets from collage images

## Recent Refactoring

The codebase has been refactored to:
1. Implement a model factory pattern in `model_factory.py`
2. Unify dataset handling in `datasets.py` with a single `ReceiptDataset` class
3. Extract training utilities into `training_utils.py`
4. Standardize validation, checkpointing, and early stopping logic 
5. Implement consistent dictionary-based metrics return values
6. Use `pathlib.Path` for modern path handling instead of `os.path`
7. Add device abstraction with `device_utils.py` for consistent hardware acceleration
8. Implement reproducibility with `reproducibility.py` and `set_seed()` function
9. Improve CLI interfaces with argument groups and shorthand options
10. Add support for checkpoint resuming via the `--resume` parameter

## Environment Variables

You can use these environment variables to override default configurations:

```bash
# Class distribution
export RECEIPT_CLASS_DIST="0.4,0.2,0.2,0.1,0.1"

# Model parameters
export RECEIPT_IMAGE_SIZE="256"
export RECEIPT_BATCH_SIZE="32" 
export RECEIPT_LEARNING_RATE="1e-5"
export RECEIPT_NUM_WORKERS="8"
export RECEIPT_WEIGHT_DECAY="0.005"
export RECEIPT_LABEL_SMOOTHING="0.05"

# Reproducibility settings
export RECEIPT_RANDOM_SEED="42"
export RECEIPT_DETERMINISTIC_MODE="true"
```

## Reproducibility

The project now includes a centralized reproducibility module for consistent results:

- Set random seeds for all libraries (Python, NumPy, PyTorch)
- Optional deterministic mode for complete reproducibility  
- Command-line options via `--seed` and `--deterministic` flags
- Environment variable configuration via `RECEIPT_RANDOM_SEED` and `RECEIPT_DETERMINISTIC_MODE`
- Default seed is 42 if not specified

Note: Full deterministic mode may impact performance, especially on GPUs.

## Hierarchical Classification Details

The hierarchical classification approach provides several advantages for the receipt counting task:

### Advantages
- Each level handles a more balanced classification task
- Level 1 focuses on the crucial distinction between no receipts and having receipts
- Level 2 specializes in distinguishing between single and multiple receipts
- Better handling of class imbalance compared to a flat multiclass approach
- Improved accuracy for minority classes
- Modular approach allows retraining individual levels independently

### Implementation Details
- The hierarchical model consists of 2-3 separate models:
  - Level 1: Binary classifier (0 vs 1+)
  - Level 2: Binary classifier (1 vs 2+)
  - Optional: Multiclass classifier (2, 3, 4, 5)
- Each model has its own directory in the output folder
- During inference, models are applied sequentially
- Confidence scores are provided for each level of classification
- Images can be automatically sorted into appropriate folders


### Usage
The hierarchical approach is ideal when:
- The dataset has imbalanced classes
- You need high accuracy in distinguishing between key categories
- A simplified output (none/single/multiple) is sufficient
- Computational efficiency is important (models can be smaller)