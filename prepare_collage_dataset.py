import os
import pandas as pd
import re
import shutil
from sklearn.model_selection import train_test_split
import argparse

def extract_receipt_count(filename):
    """Extract the receipt count from the collage filename."""
    match = re.search(r'_(\d+)_receipts', filename)
    if match:
        return int(match.group(1))
    return 0

def prepare_dataset(collage_dir, output_dir, train_split=0.7, val_split=0.15, random_state=42):
    """
    Prepare a dataset from collage images for receipt counting with train/val/test splits.
    
    Args:
        collage_dir: Directory containing collage images
        output_dir: Directory to save the organized dataset
        train_split: Proportion of data to use for training (default: 0.7)
        val_split: Proportion of data to use for validation (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        train_csv_path, val_csv_path, test_csv_path: Paths to the created CSV files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Collect image paths and receipt counts
    data = []
    for filename in os.listdir(collage_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        receipt_count = extract_receipt_count(filename)
        data.append((filename, receipt_count))
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['filename', 'receipt_count'])
    
    # Calculate test split size
    test_split = 1.0 - train_split - val_split
    
    # First split to separate train from the rest
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_split + test_split), 
        random_state=random_state,
        stratify=df['receipt_count'] if len(df['receipt_count'].unique()) < 10 else None
    )
    
    # Then split the remaining data into validation and test sets
    # Calculate the relative proportion for validation from the remaining data
    relative_val_split = val_split / (val_split + test_split)
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_split),
        random_state=random_state,
        stratify=temp_df['receipt_count'] if len(temp_df['receipt_count'].unique()) < 10 else None
    )
    
    # Copy files and create CSVs
    for df_subset, target_dir, name in [
        (train_df, train_dir, 'train'), 
        (val_df, val_dir, 'val'),
        (test_df, test_dir, 'test')
    ]:
        # Copy images
        for _, row in df_subset.iterrows():
            src = os.path.join(collage_dir, row['filename'])
            dst = os.path.join(target_dir, row['filename'])
            shutil.copy2(src, dst)
        
        # Save CSV
        csv_path = os.path.join(output_dir, f'{name}.csv')
        df_subset.to_csv(csv_path, index=False)
        print(f"Created {name} dataset with {len(df_subset)} images")
    
    return (
        os.path.join(output_dir, 'train.csv'), 
        os.path.join(output_dir, 'val.csv'),
        os.path.join(output_dir, 'test.csv')
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset from collage images")
    parser.add_argument("--collage_dir", default="receipt_collages", 
                        help="Directory containing collage images")
    parser.add_argument("--output_dir", default="receipts", 
                        help="Directory to save the organized dataset")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Proportion of data to use for training (default: 0.7)")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Proportion of data to use for validation (default: 0.15)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Check that splits add up to less than 1.0
    total_split = args.train_split + args.val_split
    if total_split >= 1.0:
        raise ValueError(f"Train split ({args.train_split}) + validation split ({args.val_split}) must be less than 1.0")
    
    train_csv, val_csv, test_csv = prepare_dataset(
        args.collage_dir, 
        args.output_dir, 
        train_split=args.train_split,
        val_split=args.val_split,
        random_state=args.random_seed
    )
    
    print(f"\nDataset preparation complete.")
    print(f"Train CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")
    print(f"Test CSV: {test_csv}")
    
    print("\nTo train the hierarchical model, run:")
    print(f"python train_hierarchical_model.py --train_csv {train_csv} --train_dir {os.path.join(args.output_dir, 'train')} --val_csv {val_csv} --val_dir {os.path.join(args.output_dir, 'val')} --model_type vit --output_dir models/hierarchical")

if __name__ == "__main__":
    main()