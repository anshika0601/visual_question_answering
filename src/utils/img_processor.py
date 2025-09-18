# src/utils/image_processor.py
from PIL import Image
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
    def load_image(self, image_path):
        """Load image from path and validate"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
            
    def validate_image(self, image):
        """Basic image validation"""
        if not image:
            return False, "No image provided"
        if min(image.size) < 50:
            return False, "Image too small"
        return True, "Valid image"

    def preprocess_for_model(self, image, target_size=(384, 384)):
        """Preprocess image for model input"""
        return image.resize(target_size, Image.Resampling.LANCZOS)