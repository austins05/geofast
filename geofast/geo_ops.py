"""
Geospatial operations optimized for different backends.
Drop-in functions for common tasks.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import json

from .core import process, Backend, get_config


# =============================================================================
# File Conversions (CPU-bound, parallelized across files)
# =============================================================================

def _convert_single_geojson_to_kml(args):
    """Worker function for GeoJSON to KML conversion"""
    src_path, dst_path = args
    import fiona
    from fiona.crs import from_epsg
    
    with fiona.open(src_path, 'r') as src:
        # KML schema
        schema = src.schema.copy()
        
        with fiona.open(dst_path, 'w', driver='KML', schema=schema, crs=from_epsg(4326)) as dst:
            for feature in src:
                dst.write(feature)
    
    return dst_path


def _convert_single_kml_to_geojson(args):
    """Worker function for KML to GeoJSON conversion"""
    src_path, dst_path = args
    import fiona
    
    with fiona.open(src_path, 'r') as src:
        schema = src.schema.copy()
        crs = src.crs
        
        with fiona.open(dst_path, 'w', driver='GeoJSON', schema=schema, crs=crs) as dst:
            for feature in src:
                dst.write(feature)
    
    return dst_path


@process(backend=Backend.CPU_PARALLEL, batch=True)
def convert_geojson_to_kml(
    src_files: List[str], 
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Convert multiple GeoJSON files to KML in parallel.
    
    Args:
        src_files: List of GeoJSON file paths
        output_dir: Output directory (defaults to same as source)
    
    Returns:
        List of output KML file paths
    """
    tasks = []
    for src in src_files:
        src_path = Path(src)
        if output_dir:
            dst_path = Path(output_dir) / src_path.with_suffix('.kml').name
        else:
            dst_path = src_path.with_suffix('.kml')
        tasks.append((str(src_path), str(dst_path)))
    
    config = get_config()
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(executor.map(_convert_single_geojson_to_kml, tasks))
    
    return results


@process(backend=Backend.CPU_PARALLEL, batch=True)
def convert_kml_to_geojson(
    src_files: List[str], 
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Convert multiple KML files to GeoJSON in parallel.
    """
    tasks = []
    for src in src_files:
        src_path = Path(src)
        if output_dir:
            dst_path = Path(output_dir) / src_path.with_suffix('.geojson').name
        else:
            dst_path = src_path.with_suffix('.geojson')
        tasks.append((str(src_path), str(dst_path)))
    
    config = get_config()
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(executor.map(_convert_single_kml_to_geojson, tasks))
    
    return results


# =============================================================================
# Shapely Operations (Vectorized where possible)
# =============================================================================

@process(backend=Backend.AUTO, item_count_arg="geometries")
def simplify_geometries(
    geometries: List[Any], 
    tolerance: float = 0.001
) -> List[Any]:
    """
    Simplify geometries using Douglas-Peucker algorithm.
    Auto-selects backend based on count.
    """
    import shapely
    from shapely import simplify
    
    # Shapely 2.0+ vectorized operation
    return list(simplify(geometries, tolerance))


@process(backend=Backend.AUTO, item_count_arg="geometries")
def buffer_geometries(
    geometries: List[Any], 
    distance: float
) -> List[Any]:
    """
    Buffer geometries by distance.
    """
    import shapely
    from shapely import buffer
    
    return list(buffer(geometries, distance))


@process(backend=Backend.AUTO, item_count_arg="points")
def points_in_polygon(
    points: Any,  # GeoDataFrame or list of Points
    polygon: Any,  # Single polygon
) -> List[bool]:
    """
    Check which points are inside a polygon.
    Uses GPU if available and dataset is large enough.
    """
    from .utils import detect_backends
    
    backends = detect_backends()
    n_points = len(points) if hasattr(points, '__len__') else 0
    
    # GPU path for large datasets
    if n_points > 5000 and backends.get('cuspatial'):
        try:
            import cuspatial
            import cudf
            import geopandas as gpd
            
            if isinstance(points, gpd.GeoDataFrame):
                # Convert to cuSpatial
                points_cu = cuspatial.from_geopandas(points)
                poly_cu = cuspatial.from_geopandas(
                    gpd.GeoDataFrame(geometry=[polygon])
                )
                result = cuspatial.point_in_polygon(points_cu, poly_cu)
                return result.to_pandas().tolist()
        except Exception:
            pass  # Fall through to CPU
    
    # CPU path
    from shapely import contains
    from shapely.geometry import Point
    
    if hasattr(points, 'geometry'):
        # GeoDataFrame
        return list(contains(polygon, points.geometry))
    else:
        return list(contains(polygon, points))


# =============================================================================
# Distance Calculations (GPU-accelerated)
# =============================================================================

@process(backend=Backend.AUTO, item_count_arg="lats1")
def haversine_distances(
    lats1, lons1, 
    lats2, lons2,
    radius: float = 6371.0  # Earth radius in km
):
    """
    Calculate haversine distances between coordinate pairs.
    Automatically uses GPU for large arrays.
    """
    import numpy as np
    from .utils import detect_backends
    
    backends = detect_backends()
    n = len(lats1)
    
    # GPU path
    if n > 10000 and backends.get('cupy'):
        try:
            import cupy as cp
            
            lats1 = cp.asarray(lats1)
            lons1 = cp.asarray(lons1)
            lats2 = cp.asarray(lats2)
            lons2 = cp.asarray(lons2)
            
            # Convert to radians
            lat1_rad = cp.radians(lats1)
            lon1_rad = cp.radians(lons1)
            lat2_rad = cp.radians(lats2)
            lon2_rad = cp.radians(lons2)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            c = 2 * cp.arcsin(cp.sqrt(a))
            
            return (radius * c).get()  # Back to numpy
            
        except Exception:
            pass  # Fall through to CPU
    
    # CPU path (vectorized numpy)
    lats1 = np.asarray(lats1)
    lons1 = np.asarray(lons1)
    lats2 = np.asarray(lats2)
    lons2 = np.asarray(lons2)
    
    lat1_rad = np.radians(lats1)
    lon1_rad = np.radians(lons1)
    lat2_rad = np.radians(lats2)
    lon2_rad = np.radians(lons2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return radius * c


# =============================================================================
# Batch Geometry Operations
# =============================================================================

@process(backend=Backend.CPU_PARALLEL, batch=True)
def validate_geometries(geometries: List[Any]) -> List[Dict[str, Any]]:
    """
    Validate geometries and return validity info.
    """
    from shapely import is_valid, is_valid_reason
    
    results = []
    for geom in geometries:
        results.append({
            'valid': is_valid(geom),
            'reason': is_valid_reason(geom) if not is_valid(geom) else None
        })
    return results


@process(backend=Backend.CPU_PARALLEL, batch=True)
def make_valid(geometries: List[Any]) -> List[Any]:
    """
    Make geometries valid using buffer(0) trick or make_valid.
    """
    from shapely import make_valid as shapely_make_valid
    
    return [shapely_make_valid(g) for g in geometries]


# =============================================================================
# Spatial Joins (GPU when possible)
# =============================================================================

@process(backend=Backend.AUTO)
def spatial_join(
    left_gdf,  # GeoDataFrame
    right_gdf,  # GeoDataFrame  
    how: str = 'inner',
    predicate: str = 'intersects'
):
    """
    Spatial join between two GeoDataFrames.
    Uses GPU acceleration for large datasets.
    """
    import geopandas as gpd
    from .utils import detect_backends
    
    backends = detect_backends()
    n_left = len(left_gdf)
    n_right = len(right_gdf)
    
    # GPU path for large joins
    if n_left * n_right > 1_000_000 and backends.get('cuspatial'):
        try:
            import cuspatial
            import cudf
            
            left_cu = cuspatial.from_geopandas(left_gdf)
            right_cu = cuspatial.from_geopandas(right_gdf)
            
            result = cuspatial.sjoin(left_cu, right_cu, how=how, predicate=predicate)
            return result.to_geopandas()
            
        except Exception:
            pass  # Fall through to CPU
    
    # CPU path
    return gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)


# =============================================================================
# Vectorized Shapely 2.0+ Operations
# =============================================================================

import numpy as np

@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_centroids(geometries) -> np.ndarray:
    """
    Get centroids of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Array of Point geometries
    """
    from shapely import centroid
    return centroid(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_areas(geometries) -> np.ndarray:
    """
    Get areas of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Array of area values
    """
    from shapely import area
    return area(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_lengths(geometries) -> np.ndarray:
    """
    Get lengths of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Array of length values
    """
    from shapely import length
    return length(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_bounds(geometries) -> np.ndarray:
    """
    Get bounding boxes of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Nx4 array of (minx, miny, maxx, maxy)
    """
    from shapely import bounds
    return bounds(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_convex_hulls(geometries) -> np.ndarray:
    """
    Get convex hulls of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Array of convex hull geometries
    """
    from shapely import convex_hull
    return convex_hull(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_envelopes(geometries) -> np.ndarray:
    """
    Get bounding box polygons of geometries (vectorized).

    Args:
        geometries: Array of Shapely geometries

    Returns:
        Array of envelope (bounding box) polygons
    """
    from shapely import envelope
    return envelope(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def intersection(geometries1, geometries2) -> np.ndarray:
    """
    Compute intersection of geometry pairs (vectorized).

    Args:
        geometries1: First array of geometries
        geometries2: Second array of geometries

    Returns:
        Array of intersection geometries
    """
    from shapely import intersection as shp_intersection
    return shp_intersection(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def union(geometries1, geometries2) -> np.ndarray:
    """
    Compute union of geometry pairs (vectorized).

    Args:
        geometries1: First array of geometries
        geometries2: Second array of geometries

    Returns:
        Array of union geometries
    """
    from shapely import union as shp_union
    return shp_union(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def difference(geometries1, geometries2) -> np.ndarray:
    """
    Compute difference of geometry pairs (vectorized).

    Args:
        geometries1: First array of geometries
        geometries2: Second array of geometries

    Returns:
        Array of difference geometries
    """
    from shapely import difference as shp_difference
    return shp_difference(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def unary_union_all(geometries):
    """
    Compute union of all geometries.

    Args:
        geometries: Array of geometries to union

    Returns:
        Single geometry representing the union
    """
    from shapely import unary_union
    return unary_union(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def distance_between(geometries1, geometries2) -> np.ndarray:
    """
    Compute distances between geometry pairs (vectorized).

    Args:
        geometries1: First array of geometries
        geometries2: Second array of geometries

    Returns:
        Array of distance values
    """
    from shapely import distance
    return distance(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def intersects(geometries1, geometries2) -> np.ndarray:
    """
    Check if geometry pairs intersect (vectorized).

    Args:
        geometries1: First array of geometries
        geometries2: Second array of geometries

    Returns:
        Boolean array
    """
    from shapely import intersects as shp_intersects
    return shp_intersects(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def contains_geom(geometries1, geometries2) -> np.ndarray:
    """
    Check if geometries1 contain geometries2 (vectorized).

    Args:
        geometries1: Container geometries
        geometries2: Contained geometries

    Returns:
        Boolean array
    """
    from shapely import contains
    return contains(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries1")
def within(geometries1, geometries2) -> np.ndarray:
    """
    Check if geometries1 are within geometries2 (vectorized).

    Args:
        geometries1: Inner geometries
        geometries2: Outer geometries

    Returns:
        Boolean array
    """
    from shapely import within as shp_within
    return shp_within(geometries1, geometries2)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def clip_by_rect(geometries, xmin: float, ymin: float,
                  xmax: float, ymax: float) -> np.ndarray:
    """
    Clip geometries by a bounding rectangle (vectorized).
    Fast clipping using rectangle bounds.

    Args:
        geometries: Array of geometries
        xmin, ymin, xmax, ymax: Rectangle bounds

    Returns:
        Array of clipped geometries
    """
    from shapely import clip_by_rect as shp_clip
    return shp_clip(geometries, xmin, ymin, xmax, ymax)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def is_valid_batch(geometries) -> np.ndarray:
    """
    Check validity of geometries (vectorized).

    Args:
        geometries: Array of geometries

    Returns:
        Boolean array
    """
    from shapely import is_valid
    return is_valid(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def is_empty_batch(geometries) -> np.ndarray:
    """
    Check if geometries are empty (vectorized).

    Args:
        geometries: Array of geometries

    Returns:
        Boolean array
    """
    from shapely import is_empty
    return is_empty(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def get_coordinates(geometries) -> np.ndarray:
    """
    Extract coordinates from geometries (vectorized).

    Args:
        geometries: Array of geometries

    Returns:
        Nx2 array of coordinates
    """
    from shapely import get_coordinates as shp_get_coords
    return shp_get_coords(geometries)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def set_precision(geometries, grid_size: float) -> np.ndarray:
    """
    Set precision of geometries by snapping to grid (vectorized).

    Args:
        geometries: Array of geometries
        grid_size: Grid size for snapping

    Returns:
        Array of snapped geometries
    """
    from shapely import set_precision as shp_set_precision
    return shp_set_precision(geometries, grid_size)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def segmentize(geometries, max_segment_length: float) -> np.ndarray:
    """
    Add vertices to geometries so no segment exceeds max length (vectorized).
    Useful before coordinate system transformation.

    Args:
        geometries: Array of geometries
        max_segment_length: Maximum segment length

    Returns:
        Array of segmentized geometries
    """
    from shapely import segmentize as shp_segmentize
    return shp_segmentize(geometries, max_segment_length)


@process(backend=Backend.AUTO, item_count_arg="geometries")
def reverse(geometries) -> np.ndarray:
    """
    Reverse the order of coordinates in geometries (vectorized).

    Args:
        geometries: Array of geometries

    Returns:
        Array of reversed geometries
    """
    from shapely import reverse as shp_reverse
    return shp_reverse(geometries)


# =============================================================================
# Batch Point Creation (Optimized)
# =============================================================================

def create_points(x_coords, y_coords) -> np.ndarray:
    """
    Create Point geometries from coordinate arrays (vectorized).
    Much faster than creating points one at a time.

    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates

    Returns:
        Array of Point geometries
    """
    from shapely import points
    coords = np.column_stack([x_coords, y_coords])
    return points(coords)


def create_linestrings(coords_list) -> np.ndarray:
    """
    Create LineString geometries from coordinate arrays.

    Args:
        coords_list: List of Nx2 coordinate arrays

    Returns:
        Array of LineString geometries
    """
    from shapely import linestrings
    return linestrings(coords_list)


def create_polygons(shell_coords, holes_coords=None) -> np.ndarray:
    """
    Create Polygon geometries from coordinate arrays.

    Args:
        shell_coords: List of shell coordinate arrays
        holes_coords: Optional list of lists of hole coordinate arrays

    Returns:
        Array of Polygon geometries
    """
    from shapely import polygons
    return polygons(shell_coords, holes_coords)
