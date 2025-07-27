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
                requests_per_second: int = None):
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
            coordinate_buffer = max_images * 2
            
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
            
            stats = await collector.collect_panoramas_fast(coordinates, max_images)
            
            print(f"\n✅ Collection complete!")
            print(f"   Collected: {stats['collected']} panoramas")
            print(f"   Failed: {stats['failed']} attempts")
            print(f"   Total time: {stats['total_time']:.1f} seconds")
        
        # Run the async collection
        asyncio.run(run_collection())
        
    except Exception as e:
        print(f"❌ Collection error: {e}")
        raise


def preprocess_main():
    """Run the preprocessing workflow."""
    from src.data.preprocess import main as preprocess_main_func
    preprocess_main_func()


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
    collect_parser.add_argument('--north', type=float, help='Northern latitude bound')
    collect_parser.add_argument('--south', type=float, help='Southern latitude bound')
    collect_parser.add_argument('--east', type=float, help='Eastern longitude bound')
    collect_parser.add_argument('--west', type=float, help='Western longitude bound')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess collected data')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter out indoor images')
    
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
        print("⏹️  Press Ctrl+C to stop collection at any time")
        
        try:
            collect_main(
                max_images=max_images, 
                bounds=bounds,
                batch_size=getattr(args, 'batch_size', None),
                max_workers=getattr(args, 'max_workers', None),
                requests_per_second=getattr(args, 'requests_per_second', None)
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
        print("🔍 Filtering indoor images...")
        try:
            from src.utils.filter import main as filter_main
            filter_main()
        except Exception as e:
            print(f"❌ Filtering failed: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
