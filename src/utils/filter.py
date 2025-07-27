"""
Image filtering utilities to detect and remove indoor Street View images.
"""

import os
import json
import sys
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageStat
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.db import Database


class ImageFilter:
    """Filter out indoor and low-quality Street View images."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db = Database(config_path)
        self.images_dir = self.config['database']['images_dir']
    
    def analyze_image_for_outdoor_features(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image to detect if it's likely outdoors.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image statistics
                stat = ImageStat.Stat(img)
                
                # Convert to numpy for analysis
                img_array = np.array(img)
                height, width, channels = img_array.shape
                
                analysis = {
                    'brightness_mean': sum(stat.mean) / 3,
                    'brightness_std': sum(stat.stddev) / 3,
                    'color_variance': self._calculate_color_variance(img_array),
                    'sky_likelihood': self._detect_sky_region(img_array),
                    'horizon_likelihood': self._detect_horizon(img_array),
                    'uniformity_score': self._calculate_uniformity(img_array),
                    'is_likely_outdoor': False
                }
                
                # Decision logic for outdoor detection
                analysis['is_likely_outdoor'] = self._classify_outdoor(analysis)
                
                return analysis
                
        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return {'error': str(e), 'is_likely_outdoor': False}
    
    def _calculate_color_variance(self, img_array: np.ndarray) -> float:
        """Calculate overall color variance in the image."""
        # Higher variance suggests more diverse outdoor scenery
        return float(np.var(img_array))
    
    def _detect_sky_region(self, img_array: np.ndarray) -> float:
        """Detect if upper portion of image contains sky-like colors."""
        height, width, channels = img_array.shape
        
        # Check upper 30% of image
        upper_region = img_array[:int(height * 0.3), :, :]
        
        # Sky is typically blue-ish and bright
        blue_channel = upper_region[:, :, 2]  # Blue channel
        
        # Calculate how much of upper region is blue-ish and bright
        bright_blue_pixels = np.sum((blue_channel > 100) & (blue_channel > upper_region[:, :, 0]) & (blue_channel > upper_region[:, :, 1]))
        total_pixels = upper_region.shape[0] * upper_region.shape[1]
        
        return bright_blue_pixels / total_pixels
    
    def _detect_horizon(self, img_array: np.ndarray) -> float:
        """Detect horizontal lines that might indicate horizon."""
        # Simple edge detection for horizontal features
        height, width, channels = img_array.shape
        
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)
        
        # Look for horizontal edges in middle portion of image
        middle_third = gray[int(height * 0.3):int(height * 0.7), :]
        
        # Calculate horizontal gradient
        horizontal_edges = np.abs(np.diff(middle_third, axis=0))
        
        # Strong horizontal features suggest outdoor scenes
        return float(np.mean(horizontal_edges))
    
    def _calculate_uniformity(self, img_array: np.ndarray) -> float:
        """Calculate how uniform/repetitive the image is."""
        # Indoor spaces often have repetitive patterns (tiles, walls, etc.)
        gray = np.mean(img_array, axis=2)
        
        # Calculate local variance
        # More uniform = more likely indoor
        local_variance = np.var(gray)
        
        return float(local_variance)
    
    def _classify_outdoor(self, analysis: Dict[str, Any]) -> bool:
        """Classify if image is likely outdoor based on features."""
        score = 0
        
        # Sky detection (strong indicator)
        if analysis['sky_likelihood'] > 0.1:
            score += 3
        
        # Color variance (outdoor scenes more varied)
        if analysis['color_variance'] > 2000:
            score += 2
        
        # Brightness (outdoor usually brighter)
        if analysis['brightness_mean'] > 120:
            score += 1
        
        # Uniformity (outdoor less uniform)
        if analysis['uniformity_score'] > 1000:
            score += 1
        
        # Horizon detection
        if analysis['horizon_likelihood'] > 50:
            score += 1
        
        # Threshold for outdoor classification
        return score >= 3
    
    def filter_outdoor_images(self, min_outdoor_score: int = 3) -> Dict[str, Any]:
        """
        Filter dataset to keep only likely outdoor images.
        
        Args:
            min_outdoor_score: Minimum score to consider outdoor
            
        Returns:
            Dictionary with filtering results
        """
        print("🔍 Analyzing images for outdoor content...")
        
        all_images = self.db.get_all_images()
        outdoor_images = []
        indoor_images = []
        analysis_results = []
        
        for img_meta in all_images:
            image_path = os.path.join(self.images_dir, img_meta['filename'])
            
            if not os.path.exists(image_path):
                print(f"⚠️  Missing image: {img_meta['filename']}")
                continue
            
            print(f"   Analyzing: {img_meta['filename']}")
            analysis = self.analyze_image_for_outdoor_features(image_path)
            
            # Add image metadata to analysis
            analysis['image_metadata'] = img_meta
            analysis_results.append(analysis)
            
            if analysis.get('is_likely_outdoor', False):
                outdoor_images.append(img_meta)
                print(f"   ✅ Outdoor: {img_meta['filename']}")
            else:
                indoor_images.append(img_meta)
                print(f"   🏠 Indoor: {img_meta['filename']}")
        
        results = {
            'total_analyzed': len(analysis_results),
            'outdoor_images': outdoor_images,
            'indoor_images': indoor_images,
            'outdoor_count': len(outdoor_images),
            'indoor_count': len(indoor_images),
            'analysis_details': analysis_results
        }
        
        print(f"\n📊 Filtering Results:")
        print(f"   Total images: {results['total_analyzed']}")
        print(f"   Outdoor images: {results['outdoor_count']}")
        print(f"   Indoor images: {results['indoor_count']}")
        
        return results
    
    def remove_indoor_images(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove indoor images from dataset.
        
        Args:
            dry_run: If True, only show what would be removed
            
        Returns:
            Dictionary with removal results
        """
        filter_results = self.filter_outdoor_images()
        
        if dry_run:
            print(f"\n🔍 DRY RUN - Would remove {filter_results['indoor_count']} indoor images:")
            for img in filter_results['indoor_images']:
                print(f"   - {img['filename']} (lat: {img['lat']:.4f}, lng: {img['lng']:.4f})")
            print("\nRun with dry_run=False to actually remove these images")
            return filter_results
        
        # Actually remove indoor images
        removed_count = 0
        for img_meta in filter_results['indoor_images']:
            try:
                # Remove image file
                image_path = os.path.join(self.images_dir, img_meta['filename'])
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"🗑️  Removed: {img_meta['filename']}")
                    removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {img_meta['filename']}: {e}")
        
        # Update metadata to keep only outdoor images
        if removed_count > 0:
            self._update_metadata_with_outdoor_only(filter_results['outdoor_images'])
            print(f"✅ Updated metadata - kept {len(filter_results['outdoor_images'])} outdoor images")
        
        filter_results['actually_removed'] = removed_count
        return filter_results
    
    def _update_metadata_with_outdoor_only(self, outdoor_images: List[Dict[str, Any]]):
        """Update metadata file to contain only outdoor images."""
        metadata = self.db.load_metadata()
        metadata['images'] = outdoor_images
        metadata['statistics']['total_images'] = len(outdoor_images)
        metadata['statistics']['total_size_mb'] = round(
            sum(img.get('size_mb', 0) for img in outdoor_images), 2
        )
        self.db.save_metadata(metadata)
    
    def show_image_analysis_details(self):
        """Show detailed analysis for each image."""
        filter_results = self.filter_outdoor_images()
        
        print(f"\n📋 Detailed Analysis:")
        for analysis in filter_results['analysis_details']:
            if 'image_metadata' in analysis:
                img = analysis['image_metadata']
                print(f"\n🖼️  {img['filename']}")
                print(f"   Location: {img['lat']:.4f}, {img['lng']:.4f}")
                print(f"   Outdoor likelihood: {'✅ YES' if analysis['is_likely_outdoor'] else '❌ NO'}")
                print(f"   Sky detection: {analysis.get('sky_likelihood', 0):.3f}")
                print(f"   Color variance: {analysis.get('color_variance', 0):.0f}")
                print(f"   Brightness: {analysis.get('brightness_mean', 0):.1f}")
                print(f"   Uniformity: {analysis.get('uniformity_score', 0):.0f}")


def main():
    """Main filtering workflow."""
    print("🔍 Starting image filtering analysis...")
    
    filter_tool = ImageFilter()
    
    # Show detailed analysis
    filter_tool.show_image_analysis_details()
    
    # Show what would be removed (dry run)
    print(f"\n" + "="*50)
    removal_results = filter_tool.remove_indoor_images(dry_run=True)
    
    # Ask user if they want to proceed
    if removal_results['indoor_count'] > 0:
        print(f"\n⚠️  Found {removal_results['indoor_count']} likely indoor images.")
        response = input("Remove them? (y/N): ").lower().strip()
        
        if response == 'y':
            print("🗑️  Removing indoor images...")
            actual_results = filter_tool.remove_indoor_images(dry_run=False)
            print(f"✅ Removed {actual_results['actually_removed']} indoor images")
        else:
            print("👍 Keeping all images")
    else:
        print("✅ All images appear to be outdoor scenes!")


if __name__ == "__main__":
    main()
