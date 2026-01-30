"""
GeoFast - High-performance geospatial processing framework
Supports CPU parallel, GPU (CUDA), Numba JIT, and hybrid execution
"""

from .core import process, Backend, GeoFastConfig, get_config, set_config
from .core import cpu, parallel, gpu, numba_jit, hybrid, auto  # Shorthand decorators
from .executor import GeoFastExecutor
from .utils import detect_backends, get_optimal_backend, print_system_info

# Numba JIT primitives
from .primitives import (
    point_in_polygon, points_in_polygon_batch,
    haversine_scalar, haversine_batch, haversine_matrix,
    douglas_peucker, douglas_peucker_mask,
    lat_lon_to_hex, lat_lon_to_hex_batch, hex_to_lat_lon,
    hex_round, hex_neighbors, line_to_hex_cells, polygon_to_hex_cells,
    bbox_intersects, point_in_bbox, points_in_bbox_batch,
    NUMBA_AVAILABLE
)

# Spatial indexing
from .spatial_index import (
    SpatialIndex, HexGridIndex, PointIndex
)

# GPU kernels
from .cuda_kernels import (
    hex_keys_gpu, haversine_gpu, haversine_matrix_gpu,
    hex_keys_auto, haversine_auto,
    gpu_available, get_gpu_info,
    NUMBA_CUDA_AVAILABLE, CUPY_AVAILABLE
)

# Parallel processing with shared memory
from .parallel import (
    SharedNumpyArray, SharedArrayContext, get_shared_array,
    parallel_map, parallel_starmap, parallel_chunks, parallel_apply_shared,
    batch_iterator, process_in_batches,
    parallel_map_threads, parallel_io,
    reduce_lists, reduce_sets, reduce_dicts, reduce_arrays
)

# Vectorized geo operations
from .geo_ops import (
    simplify_geometries, buffer_geometries, points_in_polygon,
    haversine_distances, validate_geometries, make_valid, spatial_join,
    get_centroids, get_areas, get_lengths, get_bounds,
    get_convex_hulls, get_envelopes,
    intersection, union, difference, unary_union_all,
    distance_between, intersects, contains_geom, within,
    clip_by_rect, is_valid_batch, is_empty_batch,
    get_coordinates, set_precision, segmentize, reverse,
    create_points, create_linestrings, create_polygons
)

# File format converters
from .formats import (
    detect_format, convert, convert_batch,
    read_geojson, write_geojson, geojson_to_features, features_to_geojson,
    filter_features,
    read_kml, write_kml,
    read_gpx, write_gpx,
    read_csv_points, write_csv_points,
    read_mpz, MPZReader
)

# Caching
from .cache import (
    get_cache_config, set_cache_config, cached,
    get_memory_cache, get_disk_cache,
    get_polygon_cell_cache, get_spatial_index_cache,
    clear_all_caches, get_cache_stats, print_cache_stats,
    compute_hash, make_cache_key,
    LRUCache, DiskCache, PolygonCellCache, SpatialIndexCache
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "process",
    "Backend",
    "GeoFastConfig",
    "get_config",
    "set_config",
    # Shorthand decorators
    "cpu",
    "parallel",
    "gpu",
    "numba_jit",
    "hybrid",
    "auto",
    # Executor
    "GeoFastExecutor",
    # Utils
    "detect_backends",
    "get_optimal_backend",
    "print_system_info",
    # Primitives
    "point_in_polygon", "points_in_polygon_batch",
    "haversine_scalar", "haversine_batch", "haversine_matrix",
    "douglas_peucker", "douglas_peucker_mask",
    "lat_lon_to_hex", "lat_lon_to_hex_batch", "hex_to_lat_lon",
    "hex_round", "hex_neighbors", "line_to_hex_cells", "polygon_to_hex_cells",
    "bbox_intersects", "point_in_bbox", "points_in_bbox_batch",
    "NUMBA_AVAILABLE",
    # Spatial indexing
    "SpatialIndex", "HexGridIndex", "PointIndex",
    # GPU
    "hex_keys_gpu", "haversine_gpu", "haversine_matrix_gpu",
    "hex_keys_auto", "haversine_auto",
    "gpu_available", "get_gpu_info",
    "NUMBA_CUDA_AVAILABLE", "CUPY_AVAILABLE",
    # Parallel
    "SharedNumpyArray", "SharedArrayContext", "get_shared_array",
    "parallel_map", "parallel_starmap", "parallel_chunks", "parallel_apply_shared",
    "batch_iterator", "process_in_batches",
    "parallel_map_threads", "parallel_io",
    "reduce_lists", "reduce_sets", "reduce_dicts", "reduce_arrays",
    # Geo operations
    "simplify_geometries", "buffer_geometries", "points_in_polygon",
    "haversine_distances", "validate_geometries", "make_valid", "spatial_join",
    "get_centroids", "get_areas", "get_lengths", "get_bounds",
    "get_convex_hulls", "get_envelopes",
    "intersection", "union", "difference", "unary_union_all",
    "distance_between", "intersects", "contains_geom", "within",
    "clip_by_rect", "is_valid_batch", "is_empty_batch",
    "get_coordinates", "set_precision", "segmentize", "reverse",
    "create_points", "create_linestrings", "create_polygons",
    # File formats
    "detect_format", "convert", "convert_batch",
    "read_geojson", "write_geojson", "geojson_to_features", "features_to_geojson",
    "filter_features",
    "read_kml", "write_kml",
    "read_gpx", "write_gpx",
    "read_csv_points", "write_csv_points",
    "read_mpz", "MPZReader",
    # Caching
    "get_cache_config", "set_cache_config", "cached",
    "get_memory_cache", "get_disk_cache",
    "get_polygon_cell_cache", "get_spatial_index_cache",
    "clear_all_caches", "get_cache_stats", "print_cache_stats",
    "compute_hash", "make_cache_key",
    "LRUCache", "DiskCache", "PolygonCellCache", "SpatialIndexCache",
]
