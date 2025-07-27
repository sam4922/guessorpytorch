#!/usr/bin/env python3
"""
Command line interface for GeoGuessr data collection.
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.collect import main as collect_main
from src.data.preprocess import main as preprocess_main
from src.utils.db import Database


def check_api_key():
    """Check if API key is configured."""
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


def show_stats():
    """Show dataset statistics."""
    try:
        db = Database()
        stats = db.get_statistics()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total panoramas: {stats.get('total_panoramas', 0)}")
        print(f"   Total images: {stats.get('total_images', 0)}")
        print(f"   Total size: {stats.get('total_size_mb', 0):.1f} MB")
        print(f"   Images per panorama: {stats.get('images_per_panorama', 6)}")
        
    except Exception as e:
        print(f"Error getting statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description='GeoGuessr Data Collection Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect Street View images')
    collect_parser.add_argument('--max-images', type=int, default=100,
                              help='Maximum number of panoramas to collect (each has 6 heading images, default: 100)')
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
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Check setup and configuration')
    
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
        
        # Update config if max-images specified
        max_images = args.max_images
        print(f"🎯 Collecting up to {max_images} panoramas ({max_images * 6} total images)")
        
        print("⏹️  Press Ctrl+C to stop collection at any time")
        
        try:
            collect_main(max_images=max_images, bounds=bounds)
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
    exit(main())
