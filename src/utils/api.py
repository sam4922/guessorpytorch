"""
High-performance async Google Street View API utilities for fast data collection.
Optimized for collecting 100k+ panoramas efficiently.
"""

import os
import time
import json
import asyncio
import aiohttp
import aiofiles
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
import io
import uuid
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()


@dataclass
class PanoramaResult:
    """Result from panorama collection."""
    success: bool
    panorama_id: str = None
    lat: float = None
    lng: float = None
    images: List[Dict] = None
    error: str = None



class FastCollector:
    """Ultra-fast async Street View API client optimized for bulk collection."""
    
    def __init__(self, config_path: str = "src/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Support multiple API keys for higher quota
        self.api_keys = []
        primary_key = os.getenv('GOOGLE_STREET_VIEW_API_KEY')
        if primary_key:
            self.api_keys.append(primary_key)
        
        # Check for additional API keys
        for i in range(2, 6):  # Support up to 5 API keys
            key = os.getenv(f'GOOGLE_STREET_VIEW_API_KEY_{i}')
            if key:
                self.api_keys.append(key)
        
        if not self.api_keys:
            raise ValueError("No GOOGLE_STREET_VIEW_API_KEY found in environment variables")
        
        print(f"🔑 Using {len(self.api_keys)} API key(s) for collection")
        
        self.current_key_index = 0
        self.key_request_counts = [0] * len(self.api_keys)
        self.key_daily_limits = [25000] * len(self.api_keys)  # Default daily limit per key
        
        self.streetview_url = self.config['api']['streetview_base_url']
        self.timeout = self.config['api']['request_timeout']
        self.max_workers = self.config['api']['max_workers']
        self.batch_size = self.config['api']['batch_size']
        self.requests_per_second = self.config['api']['requests_per_second']
        self.headings = self.config['data_collection']['headings']
        self.images_dir = self.config['database']['images_dir']
        
        # Ensure images directory exists
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.rate_limiter = asyncio.Semaphore(self.requests_per_second)
        
        # API quota tracking
        self.requests_made = 0
        self.quota_limit = 25000  # Google's daily limit
        self.consecutive_failures = 0
        
    async def download_image_async(self, session: aiohttp.ClientSession, 
                                 lat: float, lng: float, heading: int, retry_count: int = 0) -> Optional[bytes]:
        """Download a single Street View image asynchronously with retry logic."""
        params = {
            'size': f"{self.config['data_collection']['image_size']}x{self.config['data_collection']['image_size']}",
            'location': f"{lat},{lng}",
            'heading': heading,
            'fov': self.config['data_collection']['fov'],
            'pitch': self.config['data_collection']['pitch'],
            'key': self.api_key
        }
        
        async with self.rate_limiter:
            try:
                # Add exponential backoff for retries
                if retry_count > 0:
                    delay = min(2 ** retry_count, 30)  # Max 30 second delay
                    await asyncio.sleep(delay)
                
                self.requests_made += 1
                
                async with session.get(self.streetview_url, params=params, 
                                     timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Quick validation - check if it's a valid image
                        if len(content) > 1000:  # Minimum size check
                            self.consecutive_failures = 0  # Reset failure counter
                            return content
                    elif response.status == 429:  # Rate limited
                        print(f"⚠️  Rate limit hit at {self.requests_made} requests")
                        if retry_count < 3:
                            return await self.download_image_async(session, lat, lng, heading, retry_count + 1)
                    elif response.status == 403:  # Quota exceeded
                        print(f"❌ API quota exceeded at {self.requests_made} requests")
                        return None
                    
                    self.consecutive_failures += 1
                    return None
            except Exception as e:
                self.consecutive_failures += 1
                if "quota" in str(e).lower() or "limit" in str(e).lower():
                    print(f"❌ API quota/limit error: {e}")
                    return None
                    
                print(f"Error downloading image at {lat},{lng} heading {heading}: {e}")
                
                # Retry on network errors
                if retry_count < 2:
                    return await self.download_image_async(session, lat, lng, heading, retry_count + 1)
                    
                return None
    
    async def collect_panorama_async(self, session: aiohttp.ClientSession,
                                   lat: float, lng: float) -> PanoramaResult:
        """Collect a complete panorama (6 images) asynchronously."""
        async with self.semaphore:
            try:
                # Download all 6 images concurrently
                tasks = [
                    self.download_image_async(session, lat, lng, heading)
                    for heading in self.headings
                ]
                
                images_data = await asyncio.gather(*tasks)
                
                # Check if we got valid images
                valid_images = [img for img in images_data if img is not None]
                
                if len(valid_images) >= 4:  # Require at least 4 out of 6 images
                    panorama_id = str(uuid.uuid4())
                    images = []
                    
                    for i, (heading, image_data) in enumerate(zip(self.headings, images_data)):
                        if image_data:
                            images.append({
                                'filename': f'heading_{heading:03d}.jpg',
                                'heading': heading,
                                'size': self.config['data_collection']['image_size'],
                                'fov': self.config['data_collection']['fov'],
                                'pitch': self.config['data_collection']['pitch'],
                                'data': image_data
                            })
                    
                    return PanoramaResult(
                        success=True,
                        panorama_id=panorama_id,
                        lat=lat,
                        lng=lng,
                        images=images
                    )
                else:
                    return PanoramaResult(
                        success=False,
                        error=f"Only {len(valid_images)}/6 images available"
                    )
                    
            except Exception as e:
                return PanoramaResult(
                    success=False,
                    error=str(e)
                )
    
    async def collect_batch_async(self, coordinates: List[Tuple[float, float]]) -> List[PanoramaResult]:
        """Collect a batch of panoramas asynchronously."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.collect_panorama_async(session, lat, lng)
                for lat, lng in coordinates
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(PanoramaResult(
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
    
    async def save_panorama_async(self, result: PanoramaResult, images_dir: str):
        """Save panorama to disk asynchronously."""
        if not result.success:
            return False
        
        # Create panorama directory
        panorama_dir = os.path.join(images_dir, result.panorama_id)
        os.makedirs(panorama_dir, exist_ok=True)
        
        # Save images
        for image_info in result.images:
            image_path = os.path.join(panorama_dir, image_info['filename'])
            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(image_info['data'])
        
        # Create manifest
        manifest = {
            'panorama_id': result.panorama_id,
            'created_at': datetime.now().isoformat(),
            'lat': result.lat,
            'lng': result.lng,
            'images': [
                {k: v for k, v in img.items() if k != 'data'}
                for img in result.images
            ]
        }
        
        manifest_path = os.path.join(panorama_dir, 'manifest.json')
        async with aiofiles.open(manifest_path, 'w') as f:
            await f.write(json.dumps(manifest, indent=2))
        
        return True
    
    async def collect_panoramas_fast(self, coordinates: List[Tuple[float, float]], 
                                   max_panoramas: int = None) -> Dict[str, int]:
        """
        Collect panoramas at maximum speed using async processing.
        
        Args:
            coordinates: List of (lat, lng) tuples
            max_panoramas: Maximum number of panoramas to collect
            
        Returns:
            Collection statistics
        """
        if max_panoramas is None:
            max_panoramas = len(coordinates)
        
        # Limit coordinates to max needed
        coords_to_process = coordinates[:max_panoramas * 2]  # Get 2x to account for failures
        batch_size = self.batch_size
        
        collected = 0
        failed = 0
        total_batches = (len(coords_to_process) + batch_size - 1) // batch_size
        
        print(f"🚀 Starting high-speed collection of up to {max_panoramas} panoramas")
        print(f"📊 Processing {len(coords_to_process)} coordinates in {total_batches} batches")
        print(f"📦 Batch configuration: {batch_size} coords/batch, {self.max_workers} workers, {self.requests_per_second} req/s limit")
        print()
        
        start_time = time.time()
        
        # Create progress bar for overall progress
        with tqdm(total=max_panoramas, desc="🌍 Collecting panoramas", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for batch_num in range(total_batches):
                if collected >= max_panoramas:
                    break
                    
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(coords_to_process))
                batch_coords = coords_to_process[batch_start:batch_end]
                
                # Update batch info in progress bar description
                pbar.set_description(f"🌍 Batch {batch_num + 1}/{total_batches} ({len(batch_coords)} coords)")
                
                # Collect batch
                batch_start_time = time.time()
                results = await self.collect_batch_async(batch_coords)
                
                # Save successful results
                save_tasks = []
                batch_collected = 0
                batch_failed = 0
                
                for result in results:
                    if result.success and collected < max_panoramas:
                        save_tasks.append(self.save_panorama_async(result, self.images_dir))
                        batch_collected += 1
                        collected += 1
                    elif not result.success:
                        batch_failed += 1
                        failed += 1
                
                # Save all files in parallel
                if save_tasks:
                    await asyncio.gather(*save_tasks)
                
                # Update progress bar
                pbar.update(batch_collected)
                
                # Print batch summary
                batch_time = time.time() - batch_start_time
                rate = len(batch_coords) / batch_time if batch_time > 0 else 0
                success_rate = (batch_collected / len(batch_coords) * 100) if len(batch_coords) > 0 else 0
                
                tqdm.write(f"  📈 Batch {batch_num + 1}: {batch_collected}/{len(batch_coords)} collected "
                          f"({success_rate:.1f}% success, {rate:.1f} req/s) [Total API calls: {self.requests_made}]")
                
                # Check for quota issues
                if self.consecutive_failures > 20:
                    tqdm.write(f"⚠️  {self.consecutive_failures} consecutive failures - possible quota/rate limit hit")
                    tqdm.write(f"💡 Consider reducing --max-images or using multiple API keys")
                    
                    # Add longer delay between batches when hitting limits
                    if batch_num < total_batches - 1:  # Don't delay after last batch
                        tqdm.write(f"⏸️  Adding 30 second cooldown...")
                        await asyncio.sleep(30)
                
                # Stop if we hit too many consecutive failures (likely quota exceeded)
                if self.consecutive_failures > 100:
                    tqdm.write(f"🛑 Stopping collection due to persistent failures (likely quota exceeded)")
                    tqdm.write(f"📊 Collected {collected} panoramas before hitting limits")
                    break
                
                if collected >= max_panoramas:
                    break
        
        total_time = time.time() - start_time
        overall_rate = (collected + failed) / total_time if total_time > 0 else 0
        
        # Print final summary
        print(f"\n🎯 Collection complete!")
        print(f"✅ Collected: {collected} panoramas ({collected * 6} images)")
        print(f"❌ Failed: {failed} attempts")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"🚀 Average rate: {overall_rate:.1f} requests/second")
        print(f"📊 API calls made: {self.requests_made}")
        
        if self.requests_made > 20000:
            print(f"⚠️  Warning: High API usage ({self.requests_made} calls)")
            print(f"💡 Google's daily limit is 25,000 - consider spreading collection across multiple days")
        
        stats = {
            'collected': collected,
            'failed': failed,
            'total_time': total_time,
            'requests_per_second': overall_rate,
            'api_calls_made': self.requests_made
        }
        
    
        
        return stats
