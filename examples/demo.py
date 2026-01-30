#!/usr/bin/env python3
"""
GeoFast Examples - Demonstrates framework usage patterns
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geofast import process, Backend, GeoFastConfig, set_config
from geofast.utils import print_system_info, detect_backends
from geofast.geo_ops import (
    convert_geojson_to_kml,
    haversine_distances,
    simplify_geometries,
    points_in_polygon,
)


# =============================================================================
# Example 1: Basic decorator usage
# =============================================================================

@process(backend=Backend.CPU_PARALLEL, batch=True)
def process_files(filepaths):
    """Process multiple files in parallel"""
    results = []
    for fp in filepaths:
        # Your processing logic here
        results.append(f"Processed: {fp}")
    return results


@process(backend=Backend.GPU, fallback=Backend.CPU)
def compute_heavy_math(data):
    """
    GPU-accelerated computation with CPU fallback.
    If GPU isn't available, automatically falls back to CPU.
    """
    import numpy as np
    # This will use CuPy on GPU or NumPy on CPU
    return np.sin(data) ** 2 + np.cos(data) ** 2


@process(backend=Backend.AUTO, item_count_arg="geometries")
def auto_simplify(geometries, tolerance=0.001):
    """
    Automatically selects backend based on geometry count.
    - < 100: single CPU
    - 100-1000: parallel CPU
    - > 1000: GPU (if available)
    """
    from shapely import simplify
    return list(simplify(geometries, tolerance))


@process(backend=Backend.HYBRID)
def hybrid_processing(data):
    """
    Splits work between CPU and GPU simultaneously.
    Good for very large datasets where you want max throughput.
    """
    import numpy as np
    return np.sqrt(data)


# =============================================================================
# Example 2: Override backend at runtime
# =============================================================================

@process(backend=Backend.AUTO)
def flexible_function(data):
    """Function with runtime-overridable backend"""
    import numpy as np
    return np.mean(data)


def demo_runtime_override():
    """Show how to override backend at call time"""
    import numpy as np
    
    data = np.random.rand(10000)
    
    # Use default (AUTO)
    result1 = flexible_function(data)
    
    # Force GPU
    result2 = flexible_function.with_backend(Backend.GPU)(data)
    
    # Force CPU
    result3 = flexible_function.with_backend(Backend.CPU)(data)
    
    print(f"Results: {result1:.4f}, {result2:.4f}, {result3:.4f}")


# =============================================================================
# Example 3: Custom Numba kernel for specialized operations
# =============================================================================

@process(backend=Backend.NUMBA)
def fast_point_classify(x_coords, y_coords, bounds):
    """
    JIT-compiled point classification.
    Much faster than pure Python for tight loops.
    
    bounds: (min_x, min_y, max_x, max_y)
    """
    import numpy as np
    
    n = len(x_coords)
    results = np.zeros(n, dtype=np.int32)
    min_x, min_y, max_x, max_y = bounds
    
    for i in range(n):
        x, y = x_coords[i], y_coords[i]
        if min_x <= x <= max_x and min_y <= y <= max_y:
            results[i] = 1  # Inside
        else:
            results[i] = 0  # Outside
    
    return results


# =============================================================================
# Example 4: Real-world workflow - Processing spray field data
# =============================================================================

def spray_field_workflow():
    """
    Example workflow for processing spray field data.
    Shows how different backends work together.
    """
    import numpy as np
    
    # Simulate field data
    n_fields = 5000
    n_spray_points = 100000
    
    print(f"Processing {n_fields} fields with {n_spray_points} spray points...")
    
    # 1. Generate test data
    field_lats = np.random.uniform(39.0, 42.0, n_fields)
    field_lons = np.random.uniform(-90.0, -87.0, n_fields)
    spray_lats = np.random.uniform(39.0, 42.0, n_spray_points)
    spray_lons = np.random.uniform(-90.0, -87.0, n_spray_points)
    
    # 2. Calculate distances (auto-selects GPU for 100k points)
    print("\nCalculating distances...")
    distances = haversine_distances(
        spray_lats, spray_lons,
        np.full(n_spray_points, 40.0),  # Reference point
        np.full(n_spray_points, -88.5)
    )
    print(f"  Mean distance from reference: {np.mean(distances):.2f} km")
    
    # 3. Configure for verbose output
    set_config(verbose=True)
    
    # 4. Show what backend was used
    backends = detect_backends()
    if backends.get('cupy'):
        print("\n✓ GPU acceleration was used for distance calculation")
    else:
        print("\n→ CPU was used (GPU not available)")


# =============================================================================
# Example 5: File conversion batch processing
# =============================================================================

def batch_conversion_example():
    """
    Convert multiple GeoJSON files to KML in parallel.
    """
    import tempfile
    import json
    
    # Create some test GeoJSON files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_files = []
        for i in range(10):
            filepath = os.path.join(tmpdir, f"field_{i}.geojson")
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                    },
                    "properties": {"id": i}
                }]
            }
            with open(filepath, 'w') as f:
                json.dump(geojson, f)
            test_files.append(filepath)
        
        print(f"Converting {len(test_files)} files...")
        
        # This will use all CPU cores
        # output_files = convert_geojson_to_kml(test_files)
        # print(f"Created: {output_files}")
        
        print("(Skipped actual conversion - fiona not installed in demo)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GeoFast Framework Demo")
    print("=" * 60)
    
    # Show system capabilities
    print_system_info()
    print()
    
    # Run examples
    print("\n--- Example: Spray Field Workflow ---")
    spray_field_workflow()
    
    print("\n--- Example: Runtime Backend Override ---")
    try:
        demo_runtime_override()
    except Exception as e:
        print(f"Skipped (missing dependency): {e}")
    
    print("\n--- Example: Batch File Conversion ---")
    batch_conversion_example()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
