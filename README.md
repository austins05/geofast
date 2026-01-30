# GeoFast

High-performance geospatial processing framework with GPU acceleration, Numba JIT compilation, file format conversion, and intelligent caching.

## Features

- **Multi-backend execution**: CPU, CPU Parallel, GPU (CUDA), Numba JIT, Hybrid
- **Automatic backend selection**: Based on data size and available hardware
- **File format conversion**: GeoJSON, KML, GPX, CSV, MPZ/MapPlus
- **Numba JIT primitives**: Point-in-polygon, haversine, Douglas-Peucker, hex grids
- **Spatial indexing**: R-tree, hex grid index, point index
- **GPU kernels**: CUDA acceleration via Numba CUDA and CuPy
- **Intelligent caching**: Memory and disk caching for expensive computations
- **Command-line interface**: Convert files, view system info, manage cache

## Installation

```bash
# Basic installation (CPU only)
pip install geofast

# With Numba JIT support (recommended)
pip install geofast[full]

# With GPU support (requires NVIDIA GPU + CUDA)
pip install geofast[gpu]

# Everything
pip install geofast[all]
```

## Quick Start

### File Format Conversion

```python
from geofast import convert, read_kml, MPZReader

# Convert between formats (auto-detects from extension)
convert('tracks.kml', 'tracks.geojson')
convert('field.geojson', 'field.kml')
convert('points.csv', 'points.gpx')

# Read KML directly
data = read_kml('file.kml')  # Returns GeoJSON-like dict

# Read MapPlus/MPZ files
with MPZReader('data.mpz') as reader:
    tracks = reader.get_tracks()
    polygons = reader.get_polygons()
    geojson = reader.to_geojson()
```

### High-Performance Primitives

```python
from geofast import (
    haversine_scalar, haversine_batch,
    point_in_polygon, points_in_polygon_batch,
    douglas_peucker,
    lat_lon_to_hex, polygon_to_hex_cells,
)
import numpy as np

# Fast haversine distance (Numba JIT)
dist = haversine_scalar(lat1, lon1, lat2, lon2, radius=3959)  # miles

# Batch operations (parallel)
lats1 = np.array([40.0, 41.0, 42.0])
lons1 = np.array([-74.0, -73.0, -72.0])
distances = haversine_batch(lats1, lons1, lats2, lons2)

# Point-in-polygon (Numba JIT)
poly_x = np.array([0, 1, 1, 0, 0], dtype=np.float64)
poly_y = np.array([0, 0, 1, 1, 0], dtype=np.float64)
inside = point_in_polygon(0.5, 0.5, poly_x, poly_y)  # True

# Line simplification
coords = np.array([[0, 0], [1, 0.1], [2, 0], [3, 0.1], [4, 0]])
simplified = douglas_peucker(coords, epsilon=0.2)

# Hex grid operations
q, r = lat_lon_to_hex(40.7128, -74.0060, hex_size_lat=0.001, hex_size_lon=0.001)
```

### Spatial Indexing

```python
from geofast import SpatialIndex, HexGridIndex

# R-tree index for fast polygon queries
index = SpatialIndex()
index.add_polygon('field1', polygon_coords)
index.add_polygon('field2', polygon_coords)
index.build()

# O(log n) point query
hits = index.query_point(lon, lat)

# Hex grid index for O(1) lookups
hex_index = HexGridIndex(hex_size_lat=0.001, hex_size_lon=0.001)
hex_index.add_polygons(polygons_dict)
hex_index.build()
field_ids = hex_index.query_point(lat, lon)
```

### GPU Acceleration

```python
from geofast import haversine_auto, hex_keys_auto, gpu_available

# Automatically uses GPU for large arrays, CPU for small
distances = haversine_auto(lats1, lons1, lats2, lons2)
q_arr, r_arr = hex_keys_auto(lats, lons, hex_size_lat, hex_size_lon)

# Check GPU availability
if gpu_available():
    print("GPU acceleration available!")
```

### Caching

```python
from geofast import (
    get_polygon_cell_cache,
    cached,
    set_cache_config,
    print_cache_stats,
)

# Enable/configure caching
set_cache_config(enabled=True, cache_dir='~/.geofast/cache')

# Cache expensive computations
cache = get_polygon_cell_cache()
cells = cache.get_or_compute(polygon, hex_lat, hex_lon, compute_func)

# Decorator for automatic caching
@cached(cache_type='disk', ttl=3600)
def expensive_computation(data):
    ...

# View cache statistics
print_cache_stats()
```

### Decorator-based Backend Selection

```python
from geofast import process, Backend

# Auto-select best backend based on data size
@process(backend=Backend.AUTO, item_count_arg="geometries")
def simplify_fields(geometries, tolerance=0.001):
    from shapely import simplify
    return list(simplify(geometries, tolerance))

# Force GPU with CPU fallback
@process(backend=Backend.GPU, fallback=Backend.CPU)
def compute_distances(lats1, lons1, lats2, lons2):
    ...

# Parallel file processing
@process(backend=Backend.CPU_PARALLEL, batch=True)
def convert_files(filepaths):
    ...
```

## Command Line Interface

```bash
# Convert files
geofast convert input.kml output.geojson
geofast convert tracks.mpz tracks.geojson

# Show system info
geofast info
geofast info --cache

# Manage cache
geofast cache --stats
geofast cache --clear

# List supported formats
geofast formats
```

## Supported File Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| GeoJSON | ✅ | ✅ | `.geojson`, `.json` |
| KML | ✅ | ✅ | `.kml` |
| GPX | ✅ | ✅ | `.gpx` (tracks, waypoints, routes) |
| CSV | ✅ | ✅ | Requires lat/lon columns |
| MPZ/MapPlus | ✅ | ❌ | `.mpz`, `.sdb` (SQLite-based) |
| Shapefile | ✅ | ✅ | Requires `fiona` |

## Backends

| Backend | Best For | Requirements |
|---------|----------|--------------|
| `CPU` | Small datasets, simple operations | numpy |
| `CPU_PARALLEL` | File I/O, embarrassingly parallel tasks | numpy |
| `GPU` | Large arrays (10k+), matrix operations | cupy |
| `NUMBA` | Tight loops, custom algorithms | numba |
| `NUMBA_CUDA` | Custom GPU kernels | numba + CUDA |
| `HYBRID` | Very large datasets, max throughput | all |
| `AUTO` | Let GeoFast decide | any |

## Configuration

```python
from geofast import set_config

set_config(
    max_workers=8,           # CPU cores to use
    chunk_size=1000,         # Items per parallel chunk
    gpu_batch_size=10000,    # Items per GPU batch
    auto_gpu_min_items=1000, # Threshold for AUTO->GPU
    verbose=True             # Log backend selection
)
```

## Performance

Benchmarks on a system with 16 CPU cores and NVIDIA RTX 4050:

| Operation | Original | With GeoFast | Speedup |
|-----------|----------|--------------|---------|
| Polygon cell computation (7352 polygons) | ~20s | 1.3s (cached) | **15x** |
| Track processing (262 tracks) | ~60s | 10.7s | **5.6x** |
| Point-in-polygon (batch) | ~30s | 4.7s | **6.4x** |
| Full analysis pipeline | ~300s | 41s | **7.3x** |

## Project Structure

```
geofast/
├── __init__.py      # Package exports
├── core.py          # Backend enum, @process decorator
├── executor.py      # Backend dispatch logic
├── primitives.py    # Numba JIT functions
├── spatial_index.py # R-tree, hex grid indexing
├── cuda_kernels.py  # GPU kernels (CUDA/CuPy)
├── parallel.py      # Shared memory, parallel utilities
├── geo_ops.py       # High-level geo operations
├── formats.py       # File format converters
├── cache.py         # Caching layer
├── cli.py           # Command-line interface
└── utils.py         # Detection, helpers
```

## Requirements

- Python 3.9+
- numpy

### Optional

- numba (for JIT compilation)
- shapely 2.0+ (for spatial operations)
- cupy (for GPU acceleration)
- geopandas, fiona (for shapefile support)

## License

MIT License - see LICENSE file for details.
