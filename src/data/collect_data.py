#!/usr/bin/env python3
"""
GeoGuessr Data Collection Tool
Command line interface for collecting Street View panoramas.
"""

import os
import json
import sys
import argparse
import subprocess
import asyncio
from pathlib import Path
from PIL import Image
import io
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def check_api_key():
    """Check if API key is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_STREET_VIEW_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            print("⚠️  API key not configured!")
            print("Please edit .env file and set your Google Street View API key.")
            print("Get your API key from: https://console.cloud.google.com/apis/credentials")
            print("Make sure to enable the Street View Static API and Maps Tile API")
            return False
        return True
    except Exception as e:
        print(f"Error checking API key: {e}")
        return False


def show_stats():
    """Show dataset statistics."""
    try:
        from src.utils.db import Database
        db = Database()
        stats = db.get_statistics()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total panoramas: {stats.get('total_panoramas', 0)}")
        print(f"   Total images: {stats.get('total_images', 0)}")
        print(f"   Total size: {stats.get('total_size_mb', 0):.1f} MB")
        print(f"   Images per panorama: {stats.get('images_per_panorama', 6)}")
        
    except Exception as e:
        print(f"Error getting statistics: {e}")


def collect_main(max_images: int = 100, bounds: Dict = None, 
                batch_size: int = None, max_workers: int = None, 
                requests_per_second: int = None, pre_filter: bool = True):
    """Run the collection process."""
    try:
        # Check for required packages first
        try:
            import aiohttp
            import aiofiles
        except ImportError:
            print("📦 Installing required packages...")
            subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "aiofiles", "python-dotenv"])
            import aiohttp
            import aiofiles
        
        # Now import the async collector
        import asyncio
        from src.utils.api import FastCollector
        
        async def run_collection():
            # Load config
            config_path = 'src/config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Override config with command line arguments
            if batch_size is not None:
                config['api']['batch_size'] = batch_size
            if max_workers is not None:
                config['api']['max_workers'] = max_workers
            if requests_per_second is not None:
                config['api']['requests_per_second'] = requests_per_second
            
            # Use provided bounds or default
            if bounds is None:
                bounds_to_use = config['data_collection']['default_bounds']
            else:
                bounds_to_use = bounds
            
            # Generate coordinates
            import random
            coordinates = []
            # Generate more coordinates when pre-filtering is enabled
            coordinate_buffer = max_images * (5 if pre_filter else 2)
            
            for _ in range(coordinate_buffer):
                lat = random.uniform(bounds_to_use['south'], bounds_to_use['north'])
                lng = random.uniform(bounds_to_use['west'], bounds_to_use['east'])
                coordinates.append((lat, lng))
            
            # Initialize collector and run
            collector = FastCollector(config_path)
            # Apply runtime config overrides
            if batch_size is not None:
                collector.batch_size = batch_size
            if max_workers is not None:
                collector.max_workers = max_workers
                collector.semaphore = asyncio.Semaphore(max_workers)
            if requests_per_second is not None:
                collector.requests_per_second = requests_per_second
                collector.rate_limiter = asyncio.Semaphore(requests_per_second)
            
            print(f"🔍 Pre-filtering: {'enabled' if pre_filter else 'disabled'}")
            stats = await collector.collect_panoramas_fast(coordinates, max_images, pre_filter)
            
            print(f"\n✅ Collection complete!")
            print(f"   Collected: {stats['collected']} panoramas")
            print(f"   Failed: {stats['failed']} attempts")
            print(f"   Total time: {stats['total_time']:.1f} seconds")
            if 'metadata_calls_made' in stats:
                print(f"   Metadata checks: {stats['metadata_calls_made']}")
        
        # Run the async collection
        asyncio.run(run_collection())
        
    except Exception as e:
        print(f"❌ Collection error: {e}")
        raise


def preprocess_main():
    """Run the preprocessing workflow."""
    from src.data.preprocess import main as preprocess_main_func
    preprocess_main_func()


def test_single_image(image_path: str):
    """Test image validation on a single image file."""
    try:
        from src.utils.api import FastCollector
        collector = FastCollector()
        
        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Test our validation
        is_valid = collector.validate_image_content(image_data)
        
        # Also get basic image info
        image = Image.open(io.BytesIO(image_data))
        
        print(f"\n📸 Image: {os.path.basename(image_path)}")
        print(f"   Size: {image.size[0]}x{image.size[1]}")
        print(f"   Mode: {image.mode}")
        print(f"   File size: {len(image_data)} bytes")
        print(f"   Validation result: {'✅ VALID' if is_valid else '❌ INVALID (likely no imagery placeholder)'}")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")
        return None


def test_images_in_directory(directory: str):
    """Test all images in a directory."""
    image_dir = Path(directory)
    if not image_dir.exists():
        print(f"❌ Directory {directory} does not exist")
        return
    
    print(f"🔍 Testing images in: {directory}")
    
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f'*{ext}'))
        image_files.extend(image_dir.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No image files found in {directory}")
        return
    
    print(f"📊 Found {len(image_files)} image files to test")
    
    for image_file in image_files:
        result = test_single_image(str(image_file))
        if result is True:
            valid_count += 1
        elif result is False:
            invalid_count += 1
        else:
            error_count += 1
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Valid images: {valid_count}")
    print(f"   ❌ Invalid images (placeholders): {invalid_count}")
    print(f"   ⚠️  Errors: {error_count}")
    success_rate = 100 * valid_count/(valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0
    print(f"   📊 Success rate: {valid_count}/{valid_count + invalid_count} ({success_rate:.1f}%)")


async def test_live_coordinates():
    """Test the improved collection on a few sample coordinates."""
    print("🌐 Testing live coordinate validation...")
    
    # Some test coordinates - mix of good and bad locations
    test_coords = [
        (40.7589, -73.9851),  # Times Square, NYC - should have imagery
        (37.7749, -122.4194), # San Francisco - should have imagery  
        (71.0, -8.0),         # Svalbard - likely no imagery
        (0.0, 0.0),           # Null Island - definitely no imagery
        (48.8566, 2.3522),    # Paris - should have imagery
    ]
    
    try:
        from src.utils.api import FastCollector
        collector = FastCollector()
        
        # Test the new pre-filtering
        print(f"🔍 Testing metadata pre-filtering on {len(test_coords)} coordinates...")
        filtered_coords = await collector.batch_check_imagery_available(test_coords)
        
        print(f"✅ Filtered coordinates: {len(filtered_coords)}/{len(test_coords)} have available imagery")
        for i, (lat, lng) in enumerate(test_coords):
            has_imagery = (lat, lng) in filtered_coords
            print(f"   {i+1}. ({lat}, {lng}): {'✅ Available' if has_imagery else '❌ No imagery'}")
        
    except Exception as e:
        print(f"❌ Error testing live coordinates: {e}")


def validate_main():
    """Test image validation functionality."""
    print("🧪 Image Validation Test Tool")
    print("=" * 50)
    
    # Test existing collected images if any
    images_dir = "database/images"
    if os.path.exists(images_dir):
        print("🔍 Testing existing collected images...")
        test_images_in_directory(images_dir)
    else:
        print("📁 No existing images found. Run collection first.")
    
    # Test live coordinates (requires API key)
    if check_api_key():
        try:
            asyncio.run(test_live_coordinates())
        except Exception as e:
            print(f"⚠️  Could not test live coordinates: {e}")
            print("💡 Make sure your .env file has GOOGLE_STREET_VIEW_API_KEY set")
    else:
        print("⚠️  Skipping live coordinate test - API key not configured")


def test_image_validation():
    """Test the image validation with sample images."""
    try:
        from src.utils.api import FastCollector
        collector = FastCollector()
        
        print('🧪 Testing image validation algorithm...')
        
        # Create a test gray image (simulate no-imagery placeholder)
        test_image = Image.new('RGB', (640, 640), color=(229, 227, 223))  # Typical placeholder color
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        gray_image_data = buffer.getvalue()
        
        # Test validation
        is_valid_gray = collector.validate_image_content(gray_image_data)
        print(f'📸 Gray placeholder test: {"❌ Correctly rejected" if not is_valid_gray else "⚠️ Incorrectly accepted"}')
        
        # Create a colorful test image  
        colorful_image = Image.new('RGB', (640, 640))
        pixels = []
        for y in range(640):
            for x in range(640):
                # Create a gradient pattern with varied colors
                r = (x * 255) // 640
                g = (y * 255) // 640 
                b = ((x + y) * 255) // 1280
                pixels.append((r, g, b))
        colorful_image.putdata(pixels)
        
        buffer2 = io.BytesIO()
        colorful_image.save(buffer2, format='JPEG')
        colorful_image_data = buffer2.getvalue()
        
        is_valid_colorful = collector.validate_image_content(colorful_image_data)
        print(f'🌈 Colorful test image: {"✅ Correctly accepted" if is_valid_colorful else "⚠️ Incorrectly rejected"}')
        
        print(f'🎯 Validation working: {"✅ YES" if (not is_valid_gray and is_valid_colorful) else "❌ NO"}')
        
    except Exception as e:
        print(f"❌ Error testing validation: {e}")


def filter_main():
    """Filter existing images to remove placeholders."""
    print("🔍 Filtering existing images to remove placeholders...")
    
    images_dir = "database/images"
    if not os.path.exists(images_dir):
        print(f"❌ Directory {images_dir} does not exist")
        return
    
    try:
        from src.utils.api import FastCollector
        collector = FastCollector()
        
        removed_count = 0
        total_count = 0
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for image_path in Path(images_dir).rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                total_count += 1
                
                # Read and validate image
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                if not collector.validate_image_content(image_data):
                    print(f"🗑️  Removing placeholder: {image_path.name}")
                    os.remove(image_path)
                    removed_count += 1
        
        print(f"\n📊 Filtering complete:")
        print(f"   📸 Total images processed: {total_count}")
        print(f"   🗑️  Placeholders removed: {removed_count}")
        print(f"   ✅ Valid images remaining: {total_count - removed_count}")
        print(f"   📈 Quality improvement: {100 * (total_count - removed_count)/total_count:.1f}% valid")
        
    except Exception as e:
        print(f"❌ Error filtering images: {e}")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='GeoGuessr Data Collection Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Check setup and configuration')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect Street View images')
    collect_parser.add_argument('--max-images', type=int, default=100,
                              help='Maximum number of panoramas to collect (default: 100)')
    collect_parser.add_argument('--batch-size', type=int, default=25,
                              help='Number of coordinates per batch (default: 25)')
    collect_parser.add_argument('--max-workers', type=int, default=10,
                              help='Maximum concurrent downloads (default: 10)')
    collect_parser.add_argument('--requests-per-second', type=int, default=50,
                              help='Rate limit for API requests (default: 50)')
    collect_parser.add_argument('--no-pre-filter', action='store_true',
                              help='Disable pre-filtering with metadata API (faster but lower success rate)')
    collect_parser.add_argument('--north', type=float, help='Northern latitude bound')
    collect_parser.add_argument('--south', type=float, help='Southern latitude bound')
    collect_parser.add_argument('--east', type=float, help='Eastern longitude bound')
    collect_parser.add_argument('--west', type=float, help='Western longitude bound')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess collected data')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter out placeholder images')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Test image validation on existing images')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test image validation algorithm')
    test_parser.add_argument('path', nargs='?', help='Path to image file or directory to test')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        print("🔧 Checking GeoGuessr data collection setup...")
        
        # Check API key
        if check_api_key():
            print("✅ API key configured")
        else:
            return 1
        
        # Check directories
        if os.path.exists('database'):
            print("✅ Database directory exists")
        else:
            print("📁 Creating database directory...")
            os.makedirs('database', exist_ok=True)
            os.makedirs('database/images', exist_ok=True)
        
        print("✅ Setup complete! You can now run data collection.")
        return 0
    
    elif args.command == 'collect':
        print("🌍 Starting Street View data collection...")
        
        if not check_api_key():
            return 1
        
        # Build bounds if specified
        bounds = None
        if any([args.north, args.south, args.east, args.west]):
            if not all([args.north, args.south, args.east, args.west]):
                print("❌ If specifying bounds, all four values (north, south, east, west) are required")
                return 1
            
            bounds = {
                'north': args.north,
                'south': args.south,
                'east': args.east,
                'west': args.west
            }
            print(f"📍 Using custom bounds: N={bounds['north']}, S={bounds['south']}, E={bounds['east']}, W={bounds['west']}")
        else:
            print("📍 Using default bounds (USA)")
        
        max_images = args.max_images
        print(f"🎯 Collecting up to {max_images} panoramas ({max_images * 6} total images)")
        print(f"⚙️  Settings: batch_size={getattr(args, 'batch_size', 25)}, max_workers={getattr(args, 'max_workers', 10)}, rate_limit={getattr(args, 'requests_per_second', 50)}/s")
        print(f"🔍 Pre-filtering: {'disabled (faster but lower quality)' if args.no_pre_filter else 'enabled (higher quality)'}")
        print("⏹️  Press Ctrl+C to stop collection at any time")
        
        try:
            collect_main(
                max_images=max_images, 
                bounds=bounds,
                batch_size=getattr(args, 'batch_size', None),
                max_workers=getattr(args, 'max_workers', None),
                requests_per_second=getattr(args, 'requests_per_second', None),
                pre_filter=not args.no_pre_filter
            )
            show_stats()
        except KeyboardInterrupt:
            print("\n⏹️  Collection stopped by user")
            show_stats()
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            return 1
    
    elif args.command == 'preprocess':
        print("🔄 Starting data preprocessing...")
        try:
            preprocess_main()
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return 1
    
    elif args.command == 'stats':
        show_stats()
    
    elif args.command == 'filter':
        print("🔍 Filtering placeholder images...")
        try:
            filter_main()
        except Exception as e:
            print(f"❌ Filtering failed: {e}")
            return 1
    
    elif args.command == 'validate':
        validate_main()
    
    elif args.command == 'test':
        if args.path:
            if os.path.isfile(args.path):
                test_single_image(args.path)
            elif os.path.isdir(args.path):
                test_images_in_directory(args.path)
            else:
                print(f"❌ Path {args.path} does not exist")
                return 1
        else:
            test_image_validation()
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
