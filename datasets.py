import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from receipt_processor import ReceiptProcessor
from config import get_config


class ReceiptDataset(Dataset):
    """
    Unified dataset class for receipt counting. 
    Handles both regular receipt images and collage images consistently.
    Supports hierarchical classification modes.
    """
    def __init__(self, csv_file, img_dir, transform=None, augment=False, binary=False, hierarchical_level=None):
        """
        Initialize a receipt dataset.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            augment: Whether to apply data augmentation (used for training)
            binary: Whether to use binary classification mode (0 vs 1+ receipts)
            hierarchical_level: Which level of hierarchical classification to use
                - None: standard classification (0-5 receipts) or binary if binary=True
                - "level1": Binary classification (0 vs 1+ receipts)
                - "level2": Binary classification (1 vs 2+ receipts)
                - "multiclass": Just classes 2-5
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.root_dir = os.path.dirname(self.img_dir)
        self.binary = binary  # Flag for binary classification
        self.hierarchical_level = hierarchical_level  # Hierarchical level
        
        # Get configuration parameters
        config = get_config()
        self.image_size = config.get_model_param("image_size", 256)
        
        # Get normalization parameters from config
        self.mean = np.array(config.get_model_param("normalization_mean", [0.485, 0.456, 0.406]))
        self.std = np.array(config.get_model_param("normalization_std", [0.229, 0.224, 0.225]))
        
        # Use provided transform or create from receipt processor
        self.transform = transform or ReceiptProcessor(augment=augment).transform
        
        # Filter data based on hierarchical level if specified
        if hierarchical_level == "level1":
            # Level 1: Binary 0 vs 1+
            self.data['label'] = self.data['receipt_count'].apply(lambda x: 0 if x == 0 else 1)
            print("Using hierarchical Level 1: 0 vs 1+ receipts")
        elif hierarchical_level == "level2":
            # Level 2: Binary 1 vs 2+ (filter out 0 receipts)
            self.data = self.data[self.data['receipt_count'] > 0].copy()
            self.data['label'] = self.data['receipt_count'].apply(lambda x: 0 if x == 1 else 1)
            print("Using hierarchical Level 2: 1 vs 2+ receipts")
        elif hierarchical_level == "multiclass" and not binary:
            # Only 2+ receipts for multiclass (optional)
            self.data = self.data[self.data['receipt_count'] > 1].copy()
            print("Using multiclass for 2+ receipts only")
        
        # Print the first few file names in the dataset for debugging
        print(f"First few files in dataset: {self.data.iloc[:5, 0].tolist() if len(self.data) > 0 else '(empty)'}")
        print(f"Checking for image files in: {self.img_dir} and parent dir")
        if binary and not hierarchical_level:
            print("Using binary classification mode (0 vs 1+ receipts)")
        
        print(f"Dataset contains {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        
        # Try multiple potential locations for the image
        potential_paths = [
            os.path.join(self.img_dir, filename),                # Primary location
            os.path.join(self.root_dir, 'train', filename),      # Alternative in train dir
            os.path.join(self.root_dir, 'val', filename),        # Alternative in val dir
            os.path.join(self.root_dir, filename)                # Alternative in root dir
        ]
        
        # Try each potential path
        image = None
        for path in potential_paths:
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                break
        
        if image is None:
            # Fallback to using a blank image rather than crashing
            print(f"Warning: Could not find image {filename} in any potential location.")
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Apply transformations if available - using PIL-based transforms now
        if self.transform:
            # The torchvision transforms take the PIL image directly
            image_tensor = self.transform(image)
        else:
            # Manual resize and normalization if no transform provided
            # For SwinV2, use BICUBIC interpolation for better quality
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = (image_np - self.mean) / self.std
            image_tensor = torch.tensor(image_np).permute(2, 0, 1)

        # Handle different label scenarios based on hierarchical level
        if self.hierarchical_level in ["level1", "level2"]:
            # For hierarchical levels, use the 'label' column that was prepared in __init__
            label = int(self.data.iloc[idx, self.data.columns.get_loc('label')])
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            # For standard modes, use the receipt count
            count = int(self.data.iloc[idx, 1])
            
            if self.binary:
                # Convert to binary classification (0 vs 1+ receipts)
                binary_label = 1 if count > 0 else 0
                return image_tensor, torch.tensor(binary_label, dtype=torch.long)
            else:
                # Original multiclass classification (0-5 receipts)
                # For multiclass mode with filtered data (2+ receipts), we still return the original count
                # This ensures that when we predict "2", it means 2 receipts, not class index 0
                return image_tensor, torch.tensor(count, dtype=torch.long)


def prepare_hierarchical_datasets(dataset_path, csv_file, output_path=None):
    """
    Prepare datasets for each level of the hierarchy.
    
    Args:
        dataset_path: Path to the directory containing the CSV file
        csv_file: Name of the CSV file
        output_path: Directory to save the prepared CSV files (if None, uses dataset_path)
        
    Returns:
        dict: Dictionary with created datasets for each level
    """
    import os
    import pandas as pd
    from pathlib import Path
    
    # Determine output path
    if output_path is None:
        output_path = dataset_path
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Load original dataset
    csv_path = os.path.join(dataset_path, csv_file)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Level 1: 0 vs 1+ receipts
    df_level1 = df.copy()
    df_level1['label'] = df_level1['receipt_count'].apply(lambda x: 0 if x == 0 else 1)
    level1_path = os.path.join(output_path, f'level1_{csv_file}')
    df_level1.to_csv(level1_path, index=False)
    print(f"Created Level 1 dataset with {len(df_level1)} samples")
    
    # Level 2: 1 vs 2+ receipts (filter out 0 receipts)
    df_level2 = df[df['receipt_count'] > 0].copy()
    df_level2['label'] = df_level2['receipt_count'].apply(lambda x: 0 if x == 1 else 1)
    level2_path = os.path.join(output_path, f'level2_{csv_file}')
    df_level2.to_csv(level2_path, index=False)
    print(f"Created Level 2 dataset with {len(df_level2)} samples")
    
    # Multiclass data for 2+ receipts (optional)
    df_multiclass = df[df['receipt_count'] > 1].copy()
    multiclass_path = os.path.join(output_path, f'multiclass_{csv_file}')
    df_multiclass.to_csv(multiclass_path, index=False)
    print(f"Created multiclass dataset with {len(df_multiclass)} samples")
    
    # Print class distribution for each level
    print("\nClass distribution:")
    print("Level 1 (0 vs 1+):")
    level1_counts = df_level1['label'].value_counts().sort_index()
    for label, count in level1_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df_level1)*100:.1f}%)")
    
    print("\nLevel 2 (1 vs 2+):")
    level2_counts = df_level2['label'].value_counts().sort_index()
    for label, count in level2_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df_level2)*100:.1f}%)")
    
    print("\nMulticlass (2-5):")
    multiclass_counts = df_multiclass['receipt_count'].value_counts().sort_index()
    for label, count in multiclass_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df_multiclass)*100:.1f}%)")
    
    # Return the created datasets and their paths
    return {
        'level1': {
            'df': df_level1,
            'path': level1_path
        },
        'level2': {
            'df': df_level2,
            'path': level2_path
        },
        'multiclass': {
            'df': df_multiclass,
            'path': multiclass_path
        }
    }


def create_data_loaders(
    train_csv, 
    train_dir, 
    val_csv=None, 
    val_dir=None, 
    batch_size=16, 
    augment_train=True, 
    binary=False,
    hierarchical_level=None,
    train_val_split=0.8,
    num_workers=None
):
    """
    Create data loaders for training and validation.
    
    Args:
        train_csv: Path to CSV file containing training data
        train_dir: Directory containing training images
        val_csv: Path to CSV file containing validation data (optional)
        val_dir: Directory containing validation images (optional)
        batch_size: Batch size for training and validation
        augment_train: Whether to apply data augmentation to training set
        binary: Whether to use binary classification mode
        hierarchical_level: Which level of hierarchical classification to use
            - None: standard classification (0-5 receipts) or binary if binary=True
            - "level1": Binary classification (0 vs 1+ receipts)
            - "level2": Binary classification (1 vs 2+ receipts)
            - "multiclass": Just classes 2-5
        train_val_split: Proportion of training data to use for training (if no val_csv provided)
        num_workers: Number of worker processes for data loading (if None, uses config value)
            - Set to 0 or 1 to avoid shared memory (shm) issues on systems with limited memory
        
    Returns:
        tuple: (train_loader, val_loader, num_train_samples, num_val_samples)
    """
    # Get configuration
    config = get_config()
    # Use provided num_workers or get from config
    if num_workers is None:
        num_workers = config.get_model_param("num_workers", 4)
    
    # Print warning if using low number of workers to help avoid shared memory issues
    if num_workers <= 1:
        print(f"Using {num_workers} dataloader workers - this should help avoid shared memory (shm) issues")
    else:
        print(f"Using {num_workers} dataloader workers")
    
    # Ensure dataset directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if val_dir and not os.path.exists(val_dir):
        print(f"Warning: Validation directory not found: {val_dir}. Using training directory split instead.")
        val_dir = None
        val_csv = None  # Reset val_csv to force train/val split
    
    # If separate validation set is provided
    if val_csv and val_dir and os.path.exists(val_csv):
        # Create datasets with appropriate augmentation settings
        train_dataset = ReceiptDataset(
            train_csv, 
            train_dir, 
            augment=augment_train, 
            binary=binary, 
            hierarchical_level=hierarchical_level
        )
        
        val_dataset = ReceiptDataset(
            val_csv, 
            val_dir, 
            augment=False, 
            binary=binary, 
            hierarchical_level=hierarchical_level
        )  # No augmentation for validation
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        return train_loader, val_loader, len(train_dataset), len(val_dataset)
    
    # If no separate validation set, split the training set
    else:
        # Create datasets with and without augmentation
        train_data_with_aug = ReceiptDataset(
            train_csv, 
            train_dir, 
            augment=augment_train, 
            binary=binary, 
            hierarchical_level=hierarchical_level
        )
        
        val_data_no_aug = ReceiptDataset(
            train_csv, 
            train_dir, 
            augment=False, 
            binary=binary, 
            hierarchical_level=hierarchical_level
        )
        
        # Calculate split sizes
        dataset_size = len(train_data_with_aug)
        train_size = int(train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        # Generate random indices for the split and ensure they don't overlap
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create data loaders with appropriate samplers
        train_loader = DataLoader(
            train_data_with_aug, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data_no_aug, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
        )
        
        print(f"Split {dataset_size} samples into {train_size} training and {val_size} validation samples")
        return train_loader, val_loader, train_size, val_size