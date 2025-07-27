"""
Data collection module for GeoGuessr training dataset.
Uses Google Street View Coverage API to find valid locations and collect images.
"""

import os
import json
import random
import uuid
import time
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.api import StreetViewAPI, TileCoverage
from src.utils.db import Database


class CoverageCollector:
    """Collects Street View coverage data using the tile-based approach."""
    
    def __init__(self, config_path: str = "src/config.json"):
        self.api = StreetViewAPI(config_path)
        self.tile_coverage = TileCoverage(config_path)
        self.db = Database(config_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def discover_coverage_tiles(self, zoom_level: int = None, bounds: Dict = None) -> List[Tuple[int, int, int]]:
        """
        Skip coverage API (not working) and go directly to random sampling.
        
        Args:
            zoom_level: Target zoom level (uses config max if None)
            bounds: Geographic bounds dict with north, south, east, west keys
            
        Returns:
            List of (x, y, zoom) tuples for random sampling
        """
        if zoom_level is None:
            zoom_level = self.config['data_collection']['max_zoom_level']
        
        if bounds is None:
            bounds = self.config['data_collection']['default_bounds']
        
        print(f"⚠️  Coverage API not available - using direct coordinate sampling")
        print(f"Geographic bounds: N={bounds['north']}, S={bounds['south']}, E={bounds['east']}, W={bounds['west']}")
        
        # Go directly to random coordinate sampling
        return self._fallback_random_tiles(zoom_level, bounds)
    
    def _get_tiles_in_bounds(self, zoom: int, bounds: Dict) -> List[Tuple[int, int, int]]:
        """Get tiles that intersect with geographic bounds."""
        import mercantile
        
        # Get tiles that cover the bounding box
        tiles = list(mercantile.tiles(
            bounds['west'], bounds['south'], bounds['east'], bounds['north'], zoom
        ))
        
        return [(tile.x, tile.y, tile.z) for tile in tiles]
    
    def _fallback_random_tiles(self, zoom_level: int, bounds: Dict) -> List[Tuple[int, int, int]]:
        """Fallback method: generate random tiles within bounds."""
        import mercantile
        
        print(f"Generating random tiles at zoom {zoom_level} within bounds...")
        
        # Get all tiles in the bounding box at target zoom
        tiles = list(mercantile.tiles(
            bounds['west'], bounds['south'], bounds['east'], bounds['north'], zoom_level
        ))
        
        # Limit to a larger number for better success rate
        max_tiles = min(20, len(tiles))  # Increased from 10 to 20
        selected_tiles = random.sample(tiles, max_tiles)
        
        result = [(tile.x, tile.y, tile.z) for tile in selected_tiles]
        print(f"Generated {len(result)} random tiles for testing")
        return result
    
    def sample_coordinates_from_tile(self, x: int, y: int, zoom: int, 
                                   num_samples: int = 5) -> List[Tuple[float, float]]:
        """
        Sample random coordinates within a tile's bounds.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            num_samples: Number of coordinate pairs to sample
            
        Returns:
            List of (lat, lng) coordinate pairs
        """
        west, south, east, north = self.tile_coverage.get_tile_bounds(x, y, zoom)
        
        coordinates = []
        for _ in range(num_samples):
            lat = random.uniform(south, north)
            lng = random.uniform(west, east)
            coordinates.append((lat, lng))
        
        return coordinates


class ImageCollector:
    """Collects Street View images from discovered locations."""
    
    def __init__(self, config_path: str = "src/config.json"):
        self.api = StreetViewAPI(config_path)
        self.db = Database(config_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Ensure images directory exists
        images_dir = self.config['database']['images_dir']
        os.makedirs(images_dir, exist_ok=True)
    
    def collect_images_from_coordinates(self, coordinates: List[Tuple[float, float]], 
                                      max_panoramas: int = None) -> Dict[str, int]:
        """
        Collect Street View panoramas from coordinate list using parallel processing.
        
        Args:
            coordinates: List of (lat, lng) tuples
            max_panoramas: Maximum number of panoramas to collect (each has 6 images)
            
        Returns:
            Dictionary with collection statistics
        """
        if max_panoramas is None:
            max_panoramas = self.config['data_collection']['images_per_session']
        
        collected = 0
        failed = 0
        skipped = 0
        
        # Shuffle coordinates for random sampling
        shuffled_coords = coordinates.copy()
        random.shuffle(shuffled_coords)
        
        # Process in batches for better efficiency
        batch_size = min(6, max_panoramas)  # Process up to 6 at a time
        pbar = tqdm(total=max_panoramas, desc="Collecting panoramas")
        
        try:
            for i in range(0, len(shuffled_coords), batch_size):
                if collected >= max_panoramas:
                    break
                
                # Get batch of coordinates
                batch_coords = shuffled_coords[i:i + batch_size]
                remaining_needed = max_panoramas - collected
                batch_coords = batch_coords[:remaining_needed]
                
                # Process batch in parallel
                results = self.api.get_streetview_panorama_parallel(
                    batch_coords, max_workers=3
                )
                
                # Save successful results
                for j, images_data in enumerate(results):
                    if images_data and collected < max_panoramas:
                        # Generate unique panorama ID
                        panorama_id = str(uuid.uuid4())
                        lat, lng = batch_coords[j] if j < len(batch_coords) else (0, 0)
                        
                        # Save panorama
                        self.db.save_panorama(panorama_id, lat, lng, images_data)
                        collected += 1
                        pbar.update(1)
                
                # Count failures
                failed += len(batch_coords) - len(results)
                
                # Brief pause between batches to respect rate limits
                if i + batch_size < len(shuffled_coords) and collected < max_panoramas:
                    time.sleep(0.5)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Panorama collection stopped by user after {collected} panoramas")
        
        finally:
            pbar.close()
        
        stats = {
            'collected': collected,
            'failed': failed,
            'skipped': skipped,
            'total_attempted': collected + failed + skipped
        }
        
        print(f"Collection complete: {stats}")
        return stats
    
    def collect_images_from_tiles(self, covered_tiles: List[Tuple[int, int, int]], 
                                max_panoramas: int = None,
                                samples_per_tile: int = 1) -> Dict[str, Any]:
        """
        Collect panoramas by sampling coordinates from covered tiles.
        
        Args:
            covered_tiles: List of (x, y, zoom) tuples with coverage
            max_panoramas: Maximum number of panoramas to collect
            samples_per_tile: Number of coordinate samples per tile
            
        Returns:
            Dictionary with collection statistics
        """
        print(f"Collecting panoramas from {len(covered_tiles)} tiles")
        
        # Sample coordinates from tiles, but limit total coordinates to max_panoramas * 4
        # Generate more coordinates to reduce failed attempts
        if max_panoramas:
            target_coords = max_panoramas * 4  # Generate 4x coordinates as buffer (increased from 2x)
            samples_per_tile = max(2, target_coords // len(covered_tiles))  # At least 2 per tile
        else:
            samples_per_tile = 3  # Default increased from 1
        
        all_coordinates = []
        for x, y, zoom in covered_tiles:
            coords = CoverageCollector().sample_coordinates_from_tile(
                x, y, zoom, samples_per_tile
            )
            all_coordinates.extend(coords)
            
            # Stop if we have enough coordinates
            if max_panoramas and len(all_coordinates) >= max_panoramas * 4:
                break
        
        print(f"Generated {len(all_coordinates)} coordinate samples")
        
        # Collect panoramas from coordinates
        return self.collect_images_from_coordinates(all_coordinates, max_panoramas)


def main(max_images: int = None, bounds: Dict = None):
    """Main data collection workflow."""
    print("Starting GeoGuessr data collection...")
    
    try:
        # max_images now represents max_panoramas directly
        max_panoramas = max_images
        
        # Step 1: Discover coverage tiles
        print("\n=== Step 1: Discovering Street View coverage ===")
        coverage_collector = CoverageCollector()
        covered_tiles = coverage_collector.discover_coverage_tiles(bounds=bounds)
        
        if not covered_tiles:
            print("No covered tiles found! Check your API key and configuration.")
            return
        
        # Step 2: Collect panoramas from covered areas
        print("\n=== Step 2: Collecting Street View panoramas ===")
        image_collector = ImageCollector()
        stats = image_collector.collect_images_from_tiles(covered_tiles, max_panoramas=max_panoramas)
        
        print(f"\n=== Collection Summary ===")
        print(f"Panoramas collected: {stats['collected']}")
        print(f"Total images: {stats['collected'] * 6}")  # 6 headings per panorama
        print(f"Failed requests: {stats['failed']}")
        print(f"Total attempts: {stats['total_attempted']}")
        
        if stats['collected'] > 0:
            success_rate = (stats['collected'] / stats['total_attempted']) * 100
            print(f"Success rate: {success_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⏹️  Data collection stopped by user")
        print("You can run 'python3 collect_data.py stats' to see collected data")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()