"""
Utility script to pre-download SwinV2 model weights and configuration.
This allows the model to be used in offline mode.
"""

import os
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor

def download_model(model_name, output_dir=None):
    """
    Download model weights, configuration, and processor for offline use.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save a copy of the cached files (optional)
    """
    print(f"Downloading {model_name}...")
    
    # Download the model config
    print("Downloading model configuration...")
    config = AutoConfig.from_pretrained(model_name)
    print("Model configuration downloaded successfully")
    
    # Download the model weights
    print("Downloading model weights...")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    print("Model weights downloaded successfully")
    
    # Download the image processor
    print("Downloading image processor...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    print("Image processor downloaded successfully")
    
    # If output directory is specified, save the model files there as well
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {output_path}...")
        model.save_pretrained(output_path)
        processor.save_pretrained(output_path)
        print(f"Model saved to {output_path}")
    
    # Print cache location
    print("\nModel is now cached locally. You can use --offline mode with the training script.")
    
    # Get the cache directory path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    print(f"Default cache location: {cache_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download SwinV2 model for offline use")
    parser.add_argument(
        "--model_name",
        default="microsoft/swinv2-tiny-patch4-window8-256",
        help="HuggingFace model name to download (default: microsoft/swinv2-tiny-patch4-window8-256)"
    )
    parser.add_argument(
        "--output_dir",
        help="Optional directory to save model files"
    )
    
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir)
    
    # Print instructions for offline use
    print("\nTo use the downloaded model in offline mode:")
    print(f"python train_swin_classification.py --offline --train_csv receipt_dataset_swinv2/train.csv --train_dir receipt_dataset_swinv2/train \\\n"
          f"                              --val_csv receipt_dataset_swinv2/val.csv --val_dir receipt_dataset_swinv2/val \\\n"
          f"                              --output_dir models/swinv2 --epochs 20 --batch_size 16")

if __name__ == "__main__":
    main()