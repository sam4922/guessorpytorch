"""
Database utilities for storing and managing GeoGuessr dataset.
Uses JSON-based storage instead of SQL for simplicity.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime


class Database:
    """Simple directory-based database for storing panorama data."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.images_dir = self.config['database']['images_dir']
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
    
    def save_panorama(self, panorama_id: str, lat: float, lng: float, images_data: List[Dict]) -> str:
        """
        Save a panorama with multiple heading images.
        
        Args:
            panorama_id: Unique identifier for the panorama
            lat: Latitude (not saved in manifest)
            lng: Longitude (not saved in manifest) 
            images_data: List of dicts with 'heading', 'image', 'filename' keys
            
        Returns:
            Path to the created panorama directory
        """
        # Create panorama directory
        panorama_dir = os.path.join(self.images_dir, panorama_id)
        os.makedirs(panorama_dir, exist_ok=True)
        
        # Save images
        for image_data in images_data:
            image_path = os.path.join(panorama_dir, image_data['filename'])
            image_data['image'].save(image_path, 'JPEG', quality=95)
        
        # Create manifest with coordinates for training/scoring
        manifest = {
            'panorama_id': panorama_id,
            'created_at': datetime.now().isoformat(),
            'lat': lat,
            'lng': lng,
            'images': [
                {
                    'filename': img['filename'],
                    'heading': img['heading'],
                    'size': self.config['data_collection']['image_size'],
                    'fov': self.config['data_collection']['fov'],
                    'pitch': self.config['data_collection']['pitch']
                }
                for img in images_data
            ]
        }
        
        # Save manifest
        manifest_path = os.path.join(panorama_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return panorama_dir
    
    
    def get_all_panoramas(self) -> List[Dict[str, Any]]:
        """Get all panorama directories and their manifests."""
        panoramas = []
        
        if not os.path.exists(self.images_dir):
            return panoramas
            
        for item in os.listdir(self.images_dir):
            panorama_dir = os.path.join(self.images_dir, item)
            if os.path.isdir(panorama_dir):
                manifest_path = os.path.join(panorama_dir, 'manifest.json')
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            panoramas.append(manifest)
                    except json.JSONDecodeError:
                        continue
        
        return panoramas
    
    def get_panorama_by_id(self, panorama_id: str) -> Optional[Dict[str, Any]]:
        """Get manifest for a specific panorama by ID."""
        panorama_dir = os.path.join(self.images_dir, panorama_id)
        manifest_path = os.path.join(panorama_dir, 'manifest.json')
        
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        panoramas = self.get_all_panoramas()
        
        total_images = 0
        total_size_mb = 0.0
        
        for panorama in panoramas:
            images = panorama.get('images', [])
            total_images += len(images)
            
            # Calculate size for each panorama directory
            panorama_dir = os.path.join(self.images_dir, panorama['panorama_id'])
            for image_info in images:
                image_path = os.path.join(panorama_dir, image_info['filename'])
                if os.path.exists(image_path):
                    size_bytes = os.path.getsize(image_path)
                    total_size_mb += size_bytes / (1024 * 1024)
        
        return {
            'total_panoramas': len(panoramas),
            'total_images': total_images,
            'total_size_mb': round(total_size_mb, 2),
            'images_per_panorama': len(self.config['data_collection']['headings'])
        }
    
    def cleanup_missing_panoramas(self):
        """Remove panorama directories that are incomplete or corrupted."""
        removed_count = 0
        
        if not os.path.exists(self.images_dir):
            return removed_count
            
        for item in os.listdir(self.images_dir):
            panorama_dir = os.path.join(self.images_dir, item)
            if os.path.isdir(panorama_dir):
                manifest_path = os.path.join(panorama_dir, 'manifest.json')
                
                # Check if manifest exists and is valid
                valid = False
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            # Check if all images exist
                            images = manifest.get('images', [])
                            all_images_exist = all(
                                os.path.exists(os.path.join(panorama_dir, img['filename']))
                                for img in images
                            )
                            if all_images_exist and len(images) > 0:
                                valid = True
                    except json.JSONDecodeError:
                        pass
                
                if not valid:
                    import shutil
                    shutil.rmtree(panorama_dir)
                    removed_count += 1
        
        if removed_count > 0:
            print(f"Removed {removed_count} incomplete panorama directories")
        
        return removed_count
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed dataset statistics including file sizes and directory info."""
        panoramas = self.get_all_panoramas()
        
        total_images = 0
        total_size_bytes = 0
        total_directories = 0
        largest_panorama = 0
        smallest_panorama = float('inf')
        
        # Scan all panorama directories
        if os.path.exists(self.images_dir):
            for item in os.listdir(self.images_dir):
                item_path = os.path.join(self.images_dir, item)
                if os.path.isdir(item_path):
                    total_directories += 1
                    panorama_size = 0
                    panorama_images = 0
                    
                    # Calculate size of this panorama directory
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            panorama_size += file_size
                            total_size_bytes += file_size
                            if file.endswith(('.jpg', '.jpeg', '.png')):
                                panorama_images += 1
                                total_images += 1
                    
                    if panorama_size > largest_panorama:
                        largest_panorama = panorama_size
                    if panorama_size < smallest_panorama and panorama_size > 0:
                        smallest_panorama = panorama_size
        
        if smallest_panorama == float('inf'):
            smallest_panorama = 0
        
        return {
            'total_panoramas': len(panoramas),
            'total_directories': total_directories,
            'total_images': total_images,
            'total_size_bytes': total_size_bytes,
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'total_size_gb': total_size_bytes / (1024 * 1024 * 1024),
            'average_panorama_size_mb': (total_size_bytes / len(panoramas) / (1024 * 1024)) if len(panoramas) > 0 else 0,
            'largest_panorama_mb': largest_panorama / (1024 * 1024),
            'smallest_panorama_mb': smallest_panorama / (1024 * 1024),
            'images_per_panorama': len(self.config['data_collection']['headings'])
        }