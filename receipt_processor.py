import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from config import get_config

class ReceiptProcessor:
    def __init__(self, img_size=None, augment=False):
        # Get configuration
        config = get_config()
        
        # Use config value or override with provided value
        self.img_size = img_size or config.get_model_param("image_size", 224)
        
        # Get normalization parameters from config
        mean = config.get_model_param("normalization_mean", [0.485, 0.456, 0.406])
        std = config.get_model_param("normalization_std", [0.229, 0.224, 0.225])
        
        if augment:
            # Enhanced transform with augmentations for training
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                ], p=0.8),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=0.6),
                    A.GaussNoise(p=0.4),  # Simplified GaussNoise
                ], p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(0, 0.1), rotate=(-15, 15), p=0.7),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    A.OpticalDistortion(distort_limit=0.3, p=0.5),
                ], p=0.4),
                A.CoarseDropout(p=0.5),  # Simplified CoarseDropout with default parameters
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            # Standard transform for evaluation
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        
    def preprocess_image(self, image_path):
        """Process a scanned document image for receipt counting."""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        preprocessed = self.transform(image=img)["image"]
        
        # Add batch dimension
        return preprocessed.unsqueeze(0)
        
    def enhance_scan_quality(self, image_path, output_path=None):
        """Enhance scanned image quality for better receipt detection."""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        if output_path:
            cv2.imwrite(output_path, opening)
            
        return opening