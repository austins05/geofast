#!/usr/bin/env python3
"""
GeoFast Command Line Interface.
Provides utilities for file conversion, system info, and cache management.
"""

import argparse
import sys
import os


def cmd_convert(args):
    """Convert between geospatial file formats."""
    from .formats import convert, detect_format

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return 1

    input_format = detect_format(input_path)
    output_format = detect_format(output_path)

    print(f"Converting: {input_path}")
    print(f"  From: {input_format}")
    print(f"  To:   {output_format}")

    try:
        result = convert(input_path, output_path)
        print(f"Output: {result}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_info(args):
    """Show system information and available backends."""
    from .utils import print_system_info
    from .cuda_kernels import gpu_available, get_gpu_info
    from .primitives import NUMBA_AVAILABLE
    from .cache import print_cache_stats

    print_system_info()

    if gpu_available():
        gpu_info = get_gpu_info()
        print(f"\nGPU: {gpu_info['device_name']}")
        if gpu_info['memory_total']:
            print(f"GPU Memory: {gpu_info['memory_total'] / 1e9:.1f} GB")
    else:
        print("\nGPU: Not available")

    print(f"Numba JIT: {'Available' if NUMBA_AVAILABLE else 'Not available'}")

    if args.cache:
        print()
        print_cache_stats()


def cmd_cache(args):
    """Manage the cache."""
    from .cache import print_cache_stats, clear_all_caches, get_cache_config

    if args.clear:
        print("Clearing all caches...")
        clear_all_caches()
        print("Done.")
    elif args.stats:
        print_cache_stats()
    elif args.path:
        config = get_cache_config()
        print(config.cache_dir)
    else:
        print_cache_stats()


def cmd_formats(args):
    """List supported file formats."""
    print("Supported File Formats")
    print("=" * 40)
    print()
    print("Read & Write:")
    print("  - GeoJSON (.geojson, .json)")
    print("  - KML (.kml)")
    print("  - GPX (.gpx)")
    print("  - CSV with lat/lon (.csv)")
    print()
    print("Read Only:")
    print("  - MPZ/MapPlus (.mpz, .sdb)")
    print("  - KMZ (.kmz) - coming soon")
    print("  - Shapefile (.shp) - requires fiona")
    print()
    print("Use 'geofast convert <input> <output>' to convert between formats.")


def cmd_spray(args):
    """Generate optimal spray patterns for field polygons."""
    from .formats import generate_spray_patterns
    from .spray_optimizer import SprayConfig

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Build config from arguments
    config = SprayConfig(
        swath_width_ft=args.swath,
        spray_speed_mph=args.speed,
        ferry_speed_mph=args.ferry_speed,
        turn_time_sec=args.turn_time,
        hop_distance_ft=args.hop_distance,
        hop_enabled=not args.no_hop,
        headland_ft=args.headland,
        angle_step_deg=args.angle_step,
    )

    print(f"Generating spray patterns: {input_path}")
    print(f"  Swath width: {config.swath_width_ft} ft")
    print(f"  Spray speed: {config.spray_speed_mph} mph")
    print(f"  Hop enabled: {config.hop_enabled}")
    if config.hop_enabled:
        print(f"  Hop distance: {config.hop_distance_ft} ft")

    try:
        result = generate_spray_patterns(
            input_path,
            output_path,
            config=config,
            obstacles_path=args.obstacles,
            return_metadata=args.metadata,
            angle=args.angle,
        )
        print(f"Output: {result}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='geofast',
        description='GeoFast - High-performance geospatial processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  geofast convert input.kml output.geojson
  geofast convert tracks.mpz tracks.geojson
  geofast info
  geofast info --cache
  geofast cache --stats
  geofast cache --clear
  geofast formats
  geofast spray field.kml lines.geojson
  geofast spray field.kml lines.geojson --swath 60 --no-hop
  geofast spray field.kml lines.geojson --metadata
        """
    )

    parser.add_argument('--version', action='version', version='%(prog)s 0.3.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between file formats')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('output', help='Output file path')
    convert_parser.set_defaults(func=cmd_convert)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--cache', action='store_true', help='Include cache statistics')
    info_parser.set_defaults(func=cmd_info)

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage cache')
    cache_parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    cache_parser.add_argument('--clear', action='store_true', help='Clear all caches')
    cache_parser.add_argument('--path', action='store_true', help='Show cache directory path')
    cache_parser.set_defaults(func=cmd_cache)

    # Formats command
    formats_parser = subparsers.add_parser('formats', help='List supported file formats')
    formats_parser.set_defaults(func=cmd_formats)

    # Spray command
    spray_parser = subparsers.add_parser('spray', help='Generate spray patterns for fields')
    spray_parser.add_argument('input', help='Input file with field polygons')
    spray_parser.add_argument('output', help='Output file for spray lines')
    spray_parser.add_argument('--swath', type=float, default=50.0,
                              help='Swath width in feet (default: 50)')
    spray_parser.add_argument('--speed', type=float, default=68.0,
                              help='Spray speed in mph (default: 68)')
    spray_parser.add_argument('--ferry-speed', type=float, default=55.0,
                              help='Ferry/hop speed in mph (default: 55)')
    spray_parser.add_argument('--turn-time', type=float, default=9.0,
                              help='Turn time in seconds (default: 9)')
    spray_parser.add_argument('--hop-distance', type=float, default=1300.0,
                              help='Max hop distance in feet (default: 1300)')
    spray_parser.add_argument('--no-hop', action='store_true',
                              help='Disable hopping (for ground operations)')
    spray_parser.add_argument('--headland', type=float, default=0.0,
                              help='Headland buffer in feet (default: 0)')
    spray_parser.add_argument('--angle-step', type=float, default=5.0,
                              help='Angle search step in degrees (default: 5)')
    spray_parser.add_argument('--angle', type=float, default=None,
                              help='Force specific angle (skip optimization)')
    spray_parser.add_argument('--obstacles', type=str, default=None,
                              help='File with obstacle polygons (no-spray zones)')
    spray_parser.add_argument('--metadata', action='store_true',
                              help='Include efficiency metadata in output')
    spray_parser.set_defaults(func=cmd_spray)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
