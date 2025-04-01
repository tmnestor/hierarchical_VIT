import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from torchvision import transforms
from config import get_config

class ReceiptProcessor:
    def __init__(self, img_size=None, augment=False):
        # Get configuration
        config = get_config()
        
        # Use config value or override with provided value
        self.img_size = img_size or config.get_model_param("image_size", 256)
        
        # Get normalization parameters for SwinV2 (different from ImageNet defaults)
        # These values are the default for SwinV2 according to the model card
        mean = config.get_model_param("normalization_mean", [0.5, 0.5, 0.5])
        std = config.get_model_param("normalization_std", [0.5, 0.5, 0.5])
        
        if augment:
            # Basic transform with limited augmentations using torchvision
            # For SwinV2, we ensure images are resized to 256x256 and use center cropping
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            # Standard transform for evaluation
            # For SwinV2, we use the same preprocessing as used during model training
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
    def preprocess_image(self, image_path):
        """Process a scanned document image for receipt counting."""
        # Read image using PIL
        img = Image.open(image_path).convert('RGB')
        
        # Apply preprocessing
        preprocessed = self.transform(img)
        
        # Add batch dimension
        return preprocessed.unsqueeze(0)
        
    def enhance_scan_quality(self, image_path, output_path=None):
        """
        Basic image enhancement using PIL.
        Note: This is a simplified version without adaptive thresholding
        that was available in OpenCV.
        """
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        # Convert to grayscale
        gray = img.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Apply some sharpening
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        # Apply threshold to make it more binary-like
        enhanced = enhanced.point(lambda x: 0 if x < 128 else 255, '1')
        
        if output_path:
            enhanced.save(output_path)
            
        return np.array(enhanced)