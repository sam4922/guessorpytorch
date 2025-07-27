"""
Data preprocessing module for GeoGuessr dataset.
Handles image preprocessing, data validation, and dataset preparation.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.db import Database


class ImagePreprocessor:
    """Handles image preprocessing for the GeoGuessr dataset."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db = Database(config_path)
        self.images_dir = self.config['database']['images_dir']
        
        # Default preprocessing parameters
        self.target_size = (224, 224)  # Standard input size for many models
        self.normalize_mean = [0.485, 0.456, 0.406]  # ImageNet means
        self.normalize_std = [0.229, 0.224, 0.225]   # ImageNet stds
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate the collected dataset for completeness and quality.
        
        Returns:
            Dictionary with validation results
        """
        print("Validating dataset...")
        
        images_metadata = self.db.get_all_images()
        validation_results = {
            'total_images': len(images_metadata),
            'valid_images': 0,
            'corrupted_images': 0,
            'missing_files': 0,
            'invalid_coordinates': 0,
            'size_issues': 0,
            'errors': []
        }
        
        for img_meta in images_metadata:
            try:
                # Check if file exists
                image_path = os.path.join(self.images_dir, img_meta['filename'])
                if not os.path.exists(image_path):
                    validation_results['missing_files'] += 1
                    validation_results['errors'].append(f"Missing file: {img_meta['filename']}")
                    continue
                
                # Check coordinates
                lat = img_meta.get('lat')
                lng = img_meta.get('lng')
                if lat is None or lng is None or not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    validation_results['invalid_coordinates'] += 1
                    validation_results['errors'].append(f"Invalid coordinates for {img_meta['filename']}: ({lat}, {lng})")
                    continue
                
                # Try to open and validate image
                try:
                    with Image.open(image_path) as img:
                        # Check image properties
                        if img.format not in ['JPEG', 'PNG']:
                            validation_results['corrupted_images'] += 1
                            validation_results['errors'].append(f"Unsupported format for {img_meta['filename']}: {img.format}")
                            continue
                        
                        # Check image size
                        if img.size[0] < 100 or img.size[1] < 100:
                            validation_results['size_issues'] += 1
                            validation_results['errors'].append(f"Image too small: {img_meta['filename']} - {img.size}")
                            continue
                        
                        validation_results['valid_images'] += 1
                
                except Exception as e:
                    validation_results['corrupted_images'] += 1
                    validation_results['errors'].append(f"Cannot open {img_meta['filename']}: {str(e)}")
            
            except Exception as e:
                validation_results['errors'].append(f"Error processing {img_meta.get('filename', 'unknown')}: {str(e)}")
        
        # Calculate success rate
        if validation_results['total_images'] > 0:
            success_rate = (validation_results['valid_images'] / validation_results['total_images']) * 100
            validation_results['success_rate'] = round(success_rate, 2)
        
        print(f"Validation complete: {validation_results['valid_images']}/{validation_results['total_images']} images valid")
        return validation_results
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Preprocess a single image for model training.
        
        Args:
            image_path: Path to the image file
            target_size: Target size tuple (width, height)
            
        Returns:
            Preprocessed image as numpy array or None if processing fails
        """
        if target_size is None:
            target_size = self.target_size
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                # Apply ImageNet normalization
                for i in range(3):
                    img_array[:, :, i] = (img_array[:, :, i] - self.normalize_mean[i]) / self.normalize_std[i]
                
                # Convert to CHW format (channels first)
                img_array = np.transpose(img_array, (2, 0, 1))
                
                return img_array
        
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def create_dataset_splits(self, train_ratio: float = 0.8, 
                            val_ratio: float = 0.1, 
                            test_ratio: float = 0.1,
                            seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split the dataset into train/validation/test sets.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducible splits
            
        Returns:
            Dictionary with 'train', 'val', and 'test' keys
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Get all valid images
        all_images = self.db.get_all_images()
        
        # Filter out any images without required fields
        valid_images = []
        for img in all_images:
            if all(key in img for key in ['filename', 'lat', 'lng']):
                image_path = os.path.join(self.images_dir, img['filename'])
                if os.path.exists(image_path):
                    valid_images.append(img)
        
        print(f"Found {len(valid_images)} valid images for splitting")
        
        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(valid_images)
        
        # Calculate split indices
        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # Create splits
        splits = {
            'train': valid_images[:n_train],
            'val': valid_images[n_train:n_train + n_val],
            'test': valid_images[n_train + n_val:]
        }
        
        print(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        # Save splits to files
        for split_name, split_data in splits.items():
            split_file = f"database/{split_name}_split.json"
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {split_name} split to {split_file}")
        
        return splits
    
    def calculate_coordinate_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about the geographic distribution of the dataset.
        
        Returns:
            Dictionary with coordinate statistics
        """
        images = self.db.get_all_images()
        
        if not images:
            return {'error': 'No images found'}
        
        latitudes = [img['lat'] for img in images if 'lat' in img]
        longitudes = [img['lng'] for img in images if 'lng' in img]
        
        if not latitudes or not longitudes:
            return {'error': 'No valid coordinates found'}
        
        stats = {
            'total_images': len(images),
            'coordinates_count': len(latitudes),
            'latitude': {
                'min': min(latitudes),
                'max': max(latitudes),
                'mean': sum(latitudes) / len(latitudes),
                'range': max(latitudes) - min(latitudes)
            },
            'longitude': {
                'min': min(longitudes),
                'max': max(longitudes),
                'mean': sum(longitudes) / len(longitudes),
                'range': max(longitudes) - min(longitudes)
            },
            'bounding_box': {
                'north': max(latitudes),
                'south': min(latitudes),
                'east': max(longitudes),
                'west': min(longitudes)
            }
        }
        
        return stats
    
    def prepare_training_data(self, batch_size: int = 32, 
                            target_size: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Prepare preprocessed data for training.
        This creates a summary of the preprocessing steps needed.
        
        Args:
            batch_size: Batch size for training
            target_size: Target image size
            
        Returns:
            Dictionary with preprocessing configuration
        """
        if target_size is None:
            target_size = self.target_size
        
        # Get dataset statistics
        coord_stats = self.calculate_coordinate_statistics()
        validation_results = self.validate_dataset()
        
        preprocessing_config = {
            'image_preprocessing': {
                'target_size': target_size,
                'normalization': {
                    'mean': self.normalize_mean,
                    'std': self.normalize_std
                },
                'format': 'CHW',  # Channels, Height, Width
                'dtype': 'float32'
            },
            'dataset_info': {
                'total_images': validation_results['valid_images'],
                'batch_size': batch_size,
                'coordinate_range': coord_stats.get('bounding_box', {}),
                'geographic_coverage': {
                    'lat_span': coord_stats.get('latitude', {}).get('range', 0),
                    'lng_span': coord_stats.get('longitude', {}).get('range', 0)
                }
            },
            'validation_results': validation_results
        }
        
        # Save preprocessing config
        config_file = "database/preprocessing_config.json"
        with open(config_file, 'w') as f:
            json.dump(preprocessing_config, f, indent=2)
        
        print(f"Preprocessing configuration saved to {config_file}")
        return preprocessing_config


def main():
    """Main preprocessing workflow."""
    print("Starting data preprocessing...")
    
    preprocessor = ImagePreprocessor()
    
    # Step 1: Validate dataset
    print("\n=== Step 1: Dataset Validation ===")
    validation_results = preprocessor.validate_dataset()
    
    if validation_results['valid_images'] == 0:
        print("No valid images found! Run data collection first.")
        return
    
    # Step 2: Calculate statistics
    print("\n=== Step 2: Dataset Statistics ===")
    coord_stats = preprocessor.calculate_coordinate_statistics()
    print(f"Geographic coverage:")
    print(f"  Latitude: {coord_stats['latitude']['min']:.3f} to {coord_stats['latitude']['max']:.3f}")
    print(f"  Longitude: {coord_stats['longitude']['min']:.3f} to {coord_stats['longitude']['max']:.3f}")
    print(f"  Total span: {coord_stats['latitude']['range']:.3f}° lat, {coord_stats['longitude']['range']:.3f}° lng")
    
    # Step 3: Create dataset splits
    print("\n=== Step 3: Creating Dataset Splits ===")
    splits = preprocessor.create_dataset_splits()
    
    # Step 4: Prepare training configuration
    print("\n=== Step 4: Preparing Training Configuration ===")
    preprocessing_config = preprocessor.prepare_training_data()
    
    print("\n=== Preprocessing Complete ===")
    print(f"Dataset ready for training with {validation_results['valid_images']} images")
    print(f"Training: {len(splits['train'])} images")
    print(f"Validation: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")


if __name__ == "__main__":
    main()