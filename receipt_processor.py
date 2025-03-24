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
        self.img_size = img_size or config.get_model_param("image_size", 224)
        
        # Get normalization parameters from config
        mean = config.get_model_param("normalization_mean", [0.485, 0.456, 0.406])
        std = config.get_model_param("normalization_std", [0.229, 0.224, 0.225])
        
        if augment:
            # Basic transform with limited augmentations using torchvision
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            # Standard transform for evaluation
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
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