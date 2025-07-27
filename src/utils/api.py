"""
Google Street View API utilities for GeoGuessr data collection.
"""

import os
import time
import json
import requests
from typing import Optional, Tuple, List
from PIL import Image
import io
import mercantile
from dotenv import load_dotenv

load_dotenv()


class StreetViewAPI:
    """Handler for Google Street View API operations."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.api_key = os.getenv('GOOGLE_STREET_VIEW_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_STREET_VIEW_API_KEY not found in environment variables")
        
        self.coverage_url = self.config['api']['coverage_base_url']
        self.streetview_url = self.config['api']['streetview_base_url']
        self.timeout = self.config['api']['request_timeout']
        self.rate_limit_delay = self.config['api']['rate_limit_delay']
    
    def get_coverage_tile(self, x: int, y: int, zoom: int) -> Optional[Image.Image]:
        """
        Get coverage tile from Google Street View Coverage API.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate  
            zoom: Zoom level
            
        Returns:
            PIL Image of coverage mask or None if request fails
        """
        params = {
            'x': x,
            'y': y,
            'zoom': zoom,
            'key': self.api_key
        }
        
        try:
            response = requests.get(
                self.coverage_url, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse PNG response
            image = Image.open(io.BytesIO(response.content))
            return image
            
        except requests.RequestException as e:
            print(f"Error fetching coverage tile ({x}, {y}, {zoom}): {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)
    
    def has_coverage(self, coverage_image: Image.Image) -> bool:
        """
        Check if coverage tile has any Street View data.
        
        Args:
            coverage_image: PIL Image from coverage API
            
        Returns:
            True if tile has coverage, False otherwise
        """
        if coverage_image.mode != 'RGBA':
            coverage_image = coverage_image.convert('RGBA')
        
        # Check for non-transparent pixels
        pixels = list(coverage_image.getdata())
        for pixel in pixels:
            if len(pixel) == 4 and pixel[3] > 0:  # Alpha > 0 means coverage
                return True
        return False
    
    def get_streetview_panorama(self, lat: float, lng: float, 
                              headings: List[int] = None) -> List[Dict]:
        """
        Download multiple Street View images for different headings at same location.
        
        Args:
            lat: Latitude
            lng: Longitude
            headings: List of headings to capture (uses config default if None)
            
        Returns:
            List of dicts with 'heading', 'image', 'filename' keys
        """
        if headings is None:
            headings = self.config['data_collection']['headings']
        
        results = []
        
        for heading in headings:
            image = self.get_streetview_image(lat, lng, heading=heading)
            if image is not None:
                filename = f"heading_{heading:03d}.jpg"
                results.append({
                    'heading': heading,
                    'image': image,
                    'filename': filename
                })
            else:
                # If any heading fails, the whole panorama is invalid
                return []
        
        return results
    
    def get_streetview_image(self, lat: float, lng: float, 
                           size: Optional[int] = None,
                           fov: Optional[int] = None,
                           heading: Optional[int] = None,
                           pitch: Optional[int] = None) -> Optional[Image.Image]:
        """
        Download Street View image for given coordinates.
        
        Args:
            lat: Latitude
            lng: Longitude
            size: Image size (uses config default if None)
            fov: Field of view (uses config default if None)
            heading: Camera heading (uses config default if None)
            pitch: Camera pitch (uses config default if None)
            
        Returns:
            PIL Image or None if request fails
        """
        params = {
            'location': f"{lat},{lng}",
            'size': f"{size or self.config['data_collection']['image_size']}x{size or self.config['data_collection']['image_size']}",
            'fov': fov or self.config['data_collection']['fov'],
            'heading': heading if heading is not None else 0,
            'pitch': pitch or self.config['data_collection']['pitch'],
            'key': self.api_key
        }
        
        try:
            response = requests.get(
                self.streetview_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Check if we got an actual image (not an error image)
            image = Image.open(io.BytesIO(response.content))
            
            # Google returns a generic "no image available" image for invalid locations
            # We can detect this by checking image properties or doing a simple hash check
            if self._is_no_image_available(image):
                return None
                
            return image
            
        except requests.RequestException as e:
            print(f"Error fetching Street View image for ({lat}, {lng}): {e}")
            return None
        
        finally:
            time.sleep(self.rate_limit_delay)
    
    def _is_no_image_available(self, image: Image.Image) -> bool:
        """
        Check if image is the generic "no image available" placeholder.
        This is a simple heuristic - could be improved with better detection.
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check if image is mostly gray (typical for no-image placeholder)
        pixels = list(image.getdata())
        gray_count = 0
        total_pixels = len(pixels)
        
        for pixel in pixels[:1000]:  # Sample first 1000 pixels
            r, g, b = pixel[:3]
            # Check if pixel is grayish
            if abs(r - g) < 10 and abs(g - b) < 10 and abs(r - b) < 10:
                gray_count += 1
        
        # If more than 80% of sampled pixels are gray, likely no image
        return (gray_count / min(1000, total_pixels)) > 0.8


class TileCoverage:
    """Utility for working with map tiles and coverage detection."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.min_zoom = self.config['data_collection']['min_zoom_level']
        self.max_zoom = self.config['data_collection']['max_zoom_level']
    
    def get_tile_bounds(self, x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
        """
        Get geographic bounds of a tile.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Tuple of (west, south, east, north) coordinates
        """
        tile = mercantile.Tile(x, y, zoom)
        bounds = mercantile.bounds(tile)
        return bounds.west, bounds.south, bounds.east, bounds.north
    
    def get_children_tiles(self, x: int, y: int, zoom: int) -> List[Tuple[int, int, int]]:
        """
        Get the four child tiles at the next zoom level.
        
        Args:
            x: Parent tile X coordinate
            y: Parent tile Y coordinate
            zoom: Parent zoom level
            
        Returns:
            List of (x, y, zoom+1) tuples for child tiles
        """
        if zoom >= self.max_zoom:
            return []
        
        child_zoom = zoom + 1
        return [
            (2 * x, 2 * y, child_zoom),
            (2 * x + 1, 2 * y, child_zoom),
            (2 * x, 2 * y + 1, child_zoom),
            (2 * x + 1, 2 * y + 1, child_zoom)
        ]
    
    def tiles_at_zoom(self, zoom: int) -> List[Tuple[int, int, int]]:
        """
        Get all possible tiles at a given zoom level.
        
        Args:
            zoom: Zoom level
            
        Returns:
            List of (x, y, zoom) tuples
        """
        tiles = []
        max_coord = 2 ** zoom
        for x in range(max_coord):
            for y in range(max_coord):
                tiles.append((x, y, zoom))
        return tiles