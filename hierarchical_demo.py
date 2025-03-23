#!/usr/bin/env python3
"""
Demo script to showcase the hierarchical receipt counting system.
This script demonstrates the hierarchical model's ability to classify
receipts into none/single/multiple categories.
"""

import os
import argparse
import glob
from pathlib import Path
from hierarchical_predictor import HierarchicalPredictor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm


def create_demo_grid(images, predictions, confidences, output_path, title="Hierarchical Receipt Counting"):
    """Create a grid of images with their predictions."""
    # Maximum number of images to display
    max_images = min(len(images), 12)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(max_images)))
    rows = cols = grid_size
    
    # Sample images if we have more than max_images
    if len(images) > max_images:
        indices = random.sample(range(len(images)), max_images)
        selected_images = [images[i] for i in indices]
        selected_predictions = [predictions[i] for i in indices]
        selected_confidences = [confidences[i] for i in indices]
    else:
        selected_images = images[:max_images]
        selected_predictions = predictions[:max_images]
        selected_confidences = confidences[:max_images]
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes array if it's multi-dimensional
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    
    # Plot each image
    for i, (img_path, pred, conf) in enumerate(zip(selected_images, selected_predictions, selected_confidences)):
        if i < len(axes):
            # Load image
            img = Image.open(img_path)
            
            # Get an axis to plot on
            ax = axes[i] if rows > 1 or cols > 1 else axes
            
            # Display image
            ax.imshow(img)
            
            # Add prediction as title
            if pred == 0:
                pred_text = "No Receipts"
                color = 'red'
            elif pred == 1:
                pred_text = "1 Receipt"
                color = 'green'
            else:
                pred_text = f"{pred}+ Receipts"
                color = 'blue'
            
            ax.set_title(f"{pred_text}\nConf: {conf:.2f}", color=color)
            ax.axis('off')
    
    # Hide any unused axes
    for i in range(len(selected_images), len(axes)):
        if rows > 1 or cols > 1:
            axes[i].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Demo grid saved to {output_path}")


def create_annotated_images(images, predictions, confidences, output_dir):
    """Create annotated versions of images with prediction overlays."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path, pred, conf in tqdm(zip(images, predictions, confidences), total=len(images), desc="Creating annotated images"):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Determine text and color
            if pred == 0:
                pred_text = "No Receipts"
                color = (255, 0, 0)  # Red
            elif pred == 1:
                pred_text = "1 Receipt"
                color = (0, 255, 0)  # Green
            else:
                pred_text = f"{pred}+ Receipts"
                color = (0, 0, 255)  # Blue
            
            # Draw a semi-transparent rectangle at the top
            rect_size = (img.width, 40)
            overlay = Image.new('RGBA', rect_size, color + (150,))  # Semi-transparent
            img.paste(overlay, (0, 0), overlay)
            
            # Add text
            try:
                # Try to use a TrueType font if available
                font = ImageFont.truetype("Arial", 24)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
            
            text = f"{pred_text} (Conf: {conf:.2f})"
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            
            # Save annotated image
            output_file = output_path / f"annotated_{Path(img_path).name}"
            img.save(output_file)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Annotated images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Receipt Counting Demo")
    
    parser.add_argument(
        "--image_dir", "-i",
        required=True,
        help="Directory containing images to process"
    )
    
    parser.add_argument(
        "--model_base_path", "-m",
        default="models/hierarchical",
        help="Base path to hierarchical models"
    )
    
    parser.add_argument(
        "--model_type", "-t",
        choices=["vit", "swin"],
        default="vit",
        help="Type of transformer model (vit or swin)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        default="demo_output",
        help="Directory to save demo outputs"
    )
    
    parser.add_argument(
        "--annotate", "-a",
        action="store_true",
        help="Create annotated versions of all images"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    image_dir = Path(args.image_dir)
    image_files = (
        list(image_dir.glob("*.jpg")) + 
        list(image_dir.glob("*.jpeg")) + 
        list(image_dir.glob("*.png"))
    )
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Determine model paths
    model_base_path = Path(args.model_base_path)
    level1_model_path = model_base_path / "level1" / f"receipt_counter_{args.model_type}_best.pth"
    level2_model_path = model_base_path / "level2" / f"receipt_counter_{args.model_type}_best.pth"
    
    # Check if multiclass model exists
    multiclass_model_path = model_base_path / "multiclass" / f"receipt_counter_{args.model_type}_best.pth"
    if multiclass_model_path.exists():
        print(f"Using multiclass model: {multiclass_model_path}")
    else:
        multiclass_model_path = None
        print("Multiclass model not found, using binary classification only")
    
    # Initialize hierarchical predictor
    predictor = HierarchicalPredictor(
        level1_model_path=level1_model_path,
        level2_model_path=level2_model_path,
        multiclass_model_path=multiclass_model_path,
        model_type=args.model_type
    )
    
    # Process all images
    predictions = []
    confidences = []
    processed_images = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get prediction
            pred, conf, _ = predictor.predict(img_path, return_confidences=True)
            predictions.append(pred)
            confidences.append(conf)
            processed_images.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Create demo grid
    grid_path = Path(args.output_dir) / "demo_grid.png"
    create_demo_grid(processed_images, predictions, confidences, grid_path)
    
    # Create annotated images if requested
    if args.annotate:
        create_annotated_images(
            processed_images, 
            predictions, 
            confidences, 
            Path(args.output_dir) / "annotated"
        )
    
    # Print summary
    counts = {
        0: predictions.count(0),
        1: predictions.count(1),
        "multi": sum(1 for p in predictions if p > 1)
    }
    
    print("\nResults Summary:")
    print(f"Processed {len(processed_images)} images")
    print(f"No receipts: {counts[0]} images ({counts[0]/len(predictions)*100:.1f}%)")
    print(f"1 receipt: {counts[1]} images ({counts[1]/len(predictions)*100:.1f}%)")
    print(f"Multiple receipts: {counts['multi']} images ({counts['multi']/len(predictions)*100:.1f}%)")
    
    # Also, sort the images if requested
    sort_dir = Path(args.output_dir) / "sorted"
    if not sort_dir.exists():
        print("\nSorting images to categories...")
        predictor.sort_by_count(processed_images, sort_dir)
    

if __name__ == "__main__":
    main()