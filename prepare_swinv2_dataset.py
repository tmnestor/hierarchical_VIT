"""
Dataset preparation script specifically for SwinV2 model.
This script regenerates datasets with SwinV2-appropriate resolution and normalization.
"""

import os
import shutil
import argparse
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def convert_dataset_to_swinv2(input_dir, output_dir, image_size=256):
    """
    Convert an existing dataset to use SwinV2 image size (256x256).
    
    Args:
        input_dir: Directory containing the existing dataset
        output_dir: Directory to save the converted dataset
        image_size: Size to resize images to (default: 256 for SwinV2)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        # Copy CSV files directly
        shutil.copy2(os.path.join(input_dir, csv_file), os.path.join(output_dir, csv_file))
        print(f"Copied {csv_file} to output directory")
    
    # Process subdirectories (train, val, test)
    for subdir in ['train', 'val', 'test']:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        if not os.path.exists(input_subdir):
            print(f"Skipping {subdir} - directory not found")
            continue
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process all images in the subdirectory
        image_files = [f for f in os.listdir(input_subdir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Converting {len(image_files)} images in {subdir} directory...")
        
        for img_file in tqdm(image_files):
            input_path = os.path.join(input_subdir, img_file)
            output_path = os.path.join(output_subdir, img_file)
            
            # Open and resize the image
            try:
                img = Image.open(input_path).convert('RGB')
                img_resized = img.resize((image_size, image_size), Image.BICUBIC)
                img_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"Dataset conversion complete. SwinV2-ready dataset saved to {output_dir}")

def verify_dataset_counts(original_dir, converted_dir):
    """
    Verify that the original and converted datasets have the same number of files.
    """
    for subdir in ['train', 'val', 'test']:
        orig_path = os.path.join(original_dir, subdir)
        conv_path = os.path.join(converted_dir, subdir)
        
        if not os.path.exists(orig_path) or not os.path.exists(conv_path):
            continue
        
        orig_count = len([f for f in os.listdir(orig_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        conv_count = len([f for f in os.listdir(conv_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"{subdir}: Original={orig_count}, Converted={conv_count}")
        if orig_count != conv_count:
            print(f"WARNING: Count mismatch in {subdir} directory!")

def main():
    parser = argparse.ArgumentParser(description="Convert an existing dataset for SwinV2 model")
    parser.add_argument("--input_dir", required=True, 
                        help="Directory containing the existing dataset")
    parser.add_argument("--output_dir", required=True, 
                        help="Directory to save the converted dataset")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to resize images to (default: 256 for SwinV2)")
    
    args = parser.parse_args()
    
    convert_dataset_to_swinv2(args.input_dir, args.output_dir, args.image_size)
    verify_dataset_counts(args.input_dir, args.output_dir)
    
    # Print command to use the new dataset
    print("\nTo train with the new SwinV2 dataset, run:")
    train_csv = os.path.join(args.output_dir, 'train.csv')
    train_dir = os.path.join(args.output_dir, 'train')
    val_csv = os.path.join(args.output_dir, 'val.csv')
    val_dir = os.path.join(args.output_dir, 'val')
    
    print(f"python train_swin_classification.py --train_csv {train_csv} --train_dir {train_dir} \\\n"
          f"                              --val_csv {val_csv} --val_dir {val_dir} \\\n"
          f"                              --output_dir models/swinv2 --epochs 20 --batch_size 16")

if __name__ == "__main__":
    main()