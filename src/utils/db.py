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
        
        # Create manifest with only headings (no lat/lng)
        manifest = {
            'panorama_id': panorama_id,
            'created_at': datetime.now().isoformat(),
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