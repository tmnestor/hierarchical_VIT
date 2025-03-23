#!/usr/bin/env python3
"""
Generate a simplified synthetic dataset with clearly defined rectangles as receipts.
This script creates images with 0-5 white rectangles on a dark background.
The goal is to create a clean dataset to debug the hierarchical classification system.
"""

import os
import random
import argparse
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from config import get_config

def create_rectangle_image(canvas_size=(512, 512), receipt_count=None, bg_color=(50, 50, 50)):
    """
    Create an image with the specified number of white rectangles on a dark background.
    
    Args:
        canvas_size: Size of the output canvas (width, height)
        receipt_count: Number of rectangles to include
        bg_color: Background color of the canvas
        
    Returns:
        image: PIL Image of the synthetic data
        actual_count: Number of rectangles in the image
    """
    # Create a canvas with dark background
    canvas = Image.new('RGB', canvas_size, color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # For 0 rectangles, just return the canvas
    if receipt_count == 0:
        return canvas, 0
    
    # Calculate grid for placing rectangles
    grid_columns = 3
    grid_rows = 2
    
    # Calculate cell dimensions
    cell_width = canvas_size[0] // grid_columns
    cell_height = canvas_size[1] // grid_rows
    
    # Keep track of which grid cells are used
    grid_used = [[False for _ in range(grid_columns)] for _ in range(grid_rows)]
    
    # Function to get unused grid cell
    def get_unused_cell():
        unused_cells = [(r, c) for r in range(grid_rows) for c in range(grid_columns) 
                      if not grid_used[r][c]]
        if not unused_cells:
            return None
        return random.choice(unused_cells)
    
    # Place each rectangle
    actual_count = 0
    
    for _ in range(receipt_count):
        # Get an unused cell
        cell = get_unused_cell()
        if cell is None:
            break  # No more cells available
            
        row, col = cell
        grid_used[row][col] = True
        
        # Calculate the cell boundaries
        cell_x = col * cell_width
        cell_y = row * cell_height
        
        # Calculate rectangle size (smaller than cell)
        width = random.randint(cell_width // 3, cell_width * 2 // 3)
        height = random.randint(cell_height // 2, cell_height * 3 // 4)
        
        # Calculate position within cell (centered with slight variation)
        pos_x = cell_x + (cell_width - width) // 2 + random.randint(-10, 10)
        pos_y = cell_y + (cell_height - height) // 2 + random.randint(-10, 10)
        
        # Apply a slight rotation
        rotation = random.uniform(-15, 15)
        
        # Create a new image for the rotated rectangle
        rect_img = Image.new('RGBA', (width + 20, height + 20), (0, 0, 0, 0))  # Transparent
        rect_draw = ImageDraw.Draw(rect_img)
        
        # Draw white rectangle
        rect_draw.rectangle([(10, 10), (width + 10, height + 10)], fill=(255, 255, 255))
        
        # Rotate the rectangle
        rotated_rect = rect_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        # Paste the rotated rectangle onto the canvas
        canvas.paste(rotated_rect, (pos_x, pos_y), rotated_rect)
        actual_count += 1
    
    return canvas, actual_count

def main():
    """
    Generate a synthetic dataset with rectangles for hierarchical classification debugging.
    """
    parser = argparse.ArgumentParser(description="Create synthetic rectangle dataset for hierarchical classification debugging")
    parser.add_argument("--output_dir", default="rectangle_dataset", 
                        help="Directory to save generated images")
    parser.add_argument("--train_size", type=int, default=300,
                        help="Number of training images to create")
    parser.add_argument("--val_size", type=int, default=100,
                        help="Number of validation images to create")
    parser.add_argument("--test_size", type=int, default=100,
                        help="Number of test images to create")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size of the output images (square)")
    parser.add_argument("--count_probs", type=str, default="0.2,0.2,0.2,0.2,0.1,0.1",
                        help="Comma-separated probabilities for 0,1,2,3,4,5 rectangles")
    
    args = parser.parse_args()
    
    # Parse probability distribution
    try:
        count_probs = [float(p) for p in args.count_probs.split(',')]
        # Normalize to ensure they sum to 1
        prob_sum = sum(count_probs)
        if prob_sum <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        count_probs = [p / prob_sum for p in count_probs]
    except ValueError as e:
        print(f"Warning: Invalid probability format: {e}")
        count_probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # Default distribution
    
    print(f"Using rectangle count distribution: {count_probs}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    # Create directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Function to generate dataset split
    def generate_split(directory, num_images, split_name):
        csv_data = [["filename", "receipt_count"]]
        
        for i in tqdm(range(num_images), desc=f"Creating {split_name} set"):
            # Select rectangle count based on probability distribution
            receipt_count = random.choices(list(range(len(count_probs))), weights=count_probs)[0]
            
            # Create image
            canvas_size = (args.image_size, args.image_size)
            image, actual_count = create_rectangle_image(
                canvas_size=canvas_size,
                receipt_count=receipt_count
            )
            
            # Generate filename
            filename = f"rectangle_{i:04d}_{actual_count}.jpg"
            file_path = directory / filename
            
            # Save the image
            image.save(file_path, "JPEG", quality=95)
            
            # Add to CSV data
            csv_data.append([filename, actual_count])
        
        # Write CSV file
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        print(f"Created {num_images} images in {directory}")
        print(f"CSV file saved to {csv_path}")
    
    # Generate datasets
    generate_split(train_dir, args.train_size, "train")
    generate_split(val_dir, args.val_size, "val")
    generate_split(test_dir, args.test_size, "test")
    
    # Create hierarchical dataset splits
    def create_hierarchical_split(csv_path, output_prefix):
        # Read original CSV
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        
        # Level 1: 0 vs 1+ receipts
        level1_data = [header]
        for row in rows:
            filename, count = row
            level1_data.append([filename, 0 if int(count) == 0 else 1])
        
        # Write Level 1 CSV
        level1_path = output_dir / f"{output_prefix}_level1.csv"
        with open(level1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(level1_data)
        
        # Level 2: 1 vs 2+ receipts (filter out 0 receipts)
        level2_data = [header]
        for row in rows:
            filename, count = row
            count = int(count)
            if count > 0:
                level2_data.append([filename, 0 if count == 1 else 1])
        
        # Write Level 2 CSV
        level2_path = output_dir / f"{output_prefix}_level2.csv"
        with open(level2_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(level2_data)
        
        # Multiclass: 2-5 receipts (filter out 0-1 receipts)
        multiclass_data = [header]
        for row in rows:
            filename, count = row
            count = int(count)
            if count >= 2:
                multiclass_data.append([filename, count - 2])  # Adjust to 0-3 classes
        
        # Write Multiclass CSV
        multiclass_path = output_dir / f"{output_prefix}_multiclass.csv"
        with open(multiclass_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(multiclass_data)
            
        print(f"Hierarchical CSVs created for {output_prefix}")
    
    # Create hierarchical splits
    create_hierarchical_split(output_dir / "train.csv", "train")
    create_hierarchical_split(output_dir / "val.csv", "val")
    create_hierarchical_split(output_dir / "test.csv", "test")
    
    print("\nDataset generation complete!")
    print(f"Full dataset saved to {output_dir}")
    print("Directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/ ({args.train_size} images)")
    print(f"  ├── val/ ({args.val_size} images)")
    print(f"  ├── test/ ({args.test_size} images)")
    print(f"  ├── train.csv, val.csv, test.csv (original data)")
    print(f"  └── train_level1.csv, train_level2.csv, etc. (hierarchical data)")

if __name__ == "__main__":
    main()