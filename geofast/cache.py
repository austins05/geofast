"""
Caching layer for expensive geospatial computations.
Supports memory caching, disk caching, and content-based invalidation.
"""

import os
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Set, Tuple, Union
from functools import wraps
import numpy as np

from .core import get_config


# =============================================================================
# Cache Configuration
# =============================================================================

DEFAULT_CACHE_DIR = os.path.expanduser('~/.geofast/cache')


class CacheConfig:
    """Configuration for caching behavior."""

    def __init__(self,
                 enabled: bool = True,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 max_memory_mb: int = 500,
                 max_disk_mb: int = 2000,
                 ttl_seconds: int = 86400 * 7):  # 1 week default
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        self.ttl_seconds = ttl_seconds

        # Create cache directory
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)


_cache_config = CacheConfig()


def get_cache_config() -> CacheConfig:
    """Get current cache configuration."""
    return _cache_config


def set_cache_config(**kwargs):
    """Update cache configuration."""
    global _cache_config
    for key, value in kwargs.items():
        if hasattr(_cache_config, key):
            setattr(_cache_config, key, value)

    if _cache_config.enabled:
        os.makedirs(_cache_config.cache_dir, exist_ok=True)


# =============================================================================
# Memory Cache (LRU)
# =============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with size limits.

    Example:
        >>> cache = LRUCache(max_size_mb=100)
        >>> cache.set('key1', large_array)
        >>> data = cache.get('key1')
    """

    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, size)
        self._current_size = 0
        self._access_order = []  # LRU tracking

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(v) for v in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v)
                      for k, v in value.items())
        elif isinstance(value, set):
            return len(value) * 64  # Rough estimate
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, str):
            return len(value)
        else:
            try:
                return len(pickle.dumps(value))
            except:
                return 1000  # Default estimate

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            value, timestamp, size = self._cache[key]
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return value
        return None

    def set(self, key: str, value: Any):
        """Set value in cache, evicting if necessary."""
        size = self._estimate_size(value)

        # Evict until we have room
        while self._current_size + size > self.max_size_bytes and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                _, _, old_size = self._cache[oldest_key]
                del self._cache[oldest_key]
                self._current_size -= old_size

        # Store
        if key in self._cache:
            _, _, old_size = self._cache[key]
            self._current_size -= old_size
            self._access_order.remove(key)

        self._cache[key] = (value, time.time(), size)
        self._current_size += size
        self._access_order.append(key)

    def delete(self, key: str):
        """Delete key from cache."""
        if key in self._cache:
            _, _, size = self._cache[key]
            del self._cache[key]
            self._current_size -= size
            if key in self._access_order:
                self._access_order.remove(key)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self._current_size = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self._cache),
            'size_mb': self._current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self._current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
        }


# Global memory cache instance
_memory_cache = LRUCache(max_size_mb=500)


def get_memory_cache() -> LRUCache:
    """Get the global memory cache."""
    return _memory_cache


# =============================================================================
# Disk Cache
# =============================================================================

class DiskCache:
    """
    Persistent disk cache for large data.

    Example:
        >>> cache = DiskCache('/path/to/cache')
        >>> cache.set('polygon_cells', cells_data, ttl=3600)
        >>> data = cache.get('polygon_cells')
    """

    def __init__(self, cache_dir: str = None, max_size_mb: int = 2000):
        self.cache_dir = cache_dir or get_cache_config().cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = os.path.join(self.cache_dir, 'index.json')
        self._index = self._load_index()

        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f)

    def _get_path(self, key: str) -> str:
        """Get file path for a cache key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f'{safe_key}.pkl')

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        if key not in self._index:
            return None

        entry = self._index[key]

        # Check TTL
        if 'expires' in entry and time.time() > entry['expires']:
            self.delete(key)
            return None

        path = self._get_path(key)
        if not os.path.exists(path):
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in disk cache."""
        path = self._get_path(key)

        # Serialize and save
        data = pickle.dumps(value)
        size = len(data)

        # Evict if necessary
        self._evict_if_needed(size)

        with open(path, 'wb') as f:
            f.write(data)

        self._index[key] = {
            'path': path,
            'size': size,
            'created': time.time(),
            'expires': time.time() + ttl if ttl else None
        }
        self._save_index()

    def delete(self, key: str):
        """Delete key from disk cache."""
        if key in self._index:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
            del self._index[key]
            self._save_index()

    def _evict_if_needed(self, needed_bytes: int):
        """Evict entries to make room."""
        current_size = sum(e.get('size', 0) for e in self._index.values())

        while current_size + needed_bytes > self.max_size_bytes and self._index:
            # Find oldest entry
            oldest_key = min(self._index.keys(),
                           key=lambda k: self._index[k].get('created', 0))
            old_size = self._index[oldest_key].get('size', 0)
            self.delete(oldest_key)
            current_size -= old_size

    def clear(self):
        """Clear all disk cache entries."""
        for key in list(self._index.keys()):
            self.delete(key)
        self._index.clear()
        self._save_index()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(e.get('size', 0) for e in self._index.values())
        return {
            'entries': len(self._index),
            'size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'cache_dir': self.cache_dir
        }


# Global disk cache instance
_disk_cache = None


def get_disk_cache() -> DiskCache:
    """Get the global disk cache."""
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = DiskCache()
    return _disk_cache


# =============================================================================
# Content-Based Cache Keys
# =============================================================================

def compute_hash(data: Any) -> str:
    """
    Compute a hash for any data type.
    Used for content-based cache invalidation.
    """
    if isinstance(data, np.ndarray):
        return hashlib.md5(data.tobytes()).hexdigest()
    elif isinstance(data, (list, tuple)):
        h = hashlib.md5()
        for item in data:
            h.update(compute_hash(item).encode())
        return h.hexdigest()
    elif isinstance(data, dict):
        h = hashlib.md5()
        for k, v in sorted(data.items()):
            h.update(str(k).encode())
            h.update(compute_hash(v).encode())
        return h.hexdigest()
    elif isinstance(data, set):
        h = hashlib.md5()
        for item in sorted(str(x) for x in data):
            h.update(item.encode())
        return h.hexdigest()
    else:
        return hashlib.md5(str(data).encode()).hexdigest()


def make_cache_key(*args, **kwargs) -> str:
    """
    Create a cache key from function arguments.

    Example:
        >>> key = make_cache_key('polygon_cells', polygon_coords, hex_size=0.001)
    """
    parts = [compute_hash(arg) for arg in args]
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={compute_hash(v)}")
    return '_'.join(parts)


# =============================================================================
# Caching Decorators
# =============================================================================

def cached(cache_type: str = 'memory', ttl: int = None, key_func: Callable = None):
    """
    Decorator to cache function results.

    Args:
        cache_type: 'memory', 'disk', or 'both'
        ttl: Time-to-live in seconds (disk only)
        key_func: Optional function to generate cache key from args

    Example:
        >>> @cached(cache_type='disk', ttl=3600)
        ... def compute_polygon_cells(polygon, hex_size):
        ...     # Expensive computation
        ...     return cells
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_cache_config()
            if not config.enabled:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}_{make_cache_key(*args, **kwargs)}"

            # Try memory cache first
            if cache_type in ('memory', 'both'):
                result = get_memory_cache().get(cache_key)
                if result is not None:
                    return result

            # Try disk cache
            if cache_type in ('disk', 'both'):
                result = get_disk_cache().get(cache_key)
                if result is not None:
                    # Also store in memory for faster subsequent access
                    if cache_type == 'both':
                        get_memory_cache().set(cache_key, result)
                    return result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            if cache_type in ('memory', 'both'):
                get_memory_cache().set(cache_key, result)

            if cache_type in ('disk', 'both'):
                get_disk_cache().set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# =============================================================================
# Polygon Cell Caching (Specific Use Case)
# =============================================================================

class PolygonCellCache:
    """
    Specialized cache for polygon-to-hex-cell mappings.
    Uses content-based keys so cache invalidates when polygon changes.

    Example:
        >>> cache = PolygonCellCache()
        >>> cells = cache.get_or_compute(polygon_coords, hex_size_lat, hex_size_lon)
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(get_cache_config().cache_dir, 'polygon_cells')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _make_key(self, polygon: Any, hex_size_lat: float, hex_size_lon: float) -> str:
        """Create cache key from polygon and hex sizes."""
        poly_hash = compute_hash(polygon)
        return f"{poly_hash}_{hex_size_lat:.8f}_{hex_size_lon:.8f}"

    def get(self, polygon: Any, hex_size_lat: float, hex_size_lon: float) -> Optional[Set[Tuple[int, int]]]:
        """Get cached cells for a polygon."""
        key = self._make_key(polygon, hex_size_lat, hex_size_lon)

        # Check memory first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    cells = pickle.load(f)
                    self._memory_cache[key] = cells
                    return cells
            except:
                pass

        return None

    def set(self, polygon: Any, hex_size_lat: float, hex_size_lon: float,
            cells: Set[Tuple[int, int]]):
        """Cache cells for a polygon."""
        key = self._make_key(polygon, hex_size_lat, hex_size_lon)

        # Store in memory
        self._memory_cache[key] = cells

        # Store on disk
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(cells, f)

    def get_or_compute(self, polygon: Any, hex_size_lat: float, hex_size_lon: float,
                       compute_func: Callable) -> Set[Tuple[int, int]]:
        """Get from cache or compute and cache."""
        cells = self.get(polygon, hex_size_lat, hex_size_lon)
        if cells is not None:
            return cells

        cells = compute_func(polygon)
        self.set(polygon, hex_size_lat, hex_size_lon, cells)
        return cells

    def clear(self):
        """Clear all cached polygon cells."""
        self._memory_cache.clear()
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, f))

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        disk_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in disk_files)

        return {
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(disk_files),
            'disk_size_mb': disk_size / (1024 * 1024)
        }


# Global polygon cell cache
_polygon_cell_cache = None


def get_polygon_cell_cache() -> PolygonCellCache:
    """Get the global polygon cell cache."""
    global _polygon_cell_cache
    if _polygon_cell_cache is None:
        _polygon_cell_cache = PolygonCellCache()
    return _polygon_cell_cache


# =============================================================================
# Spatial Index Caching
# =============================================================================

class SpatialIndexCache:
    """
    Cache for spatial indices.
    Stores pre-built indices to avoid reconstruction.

    Example:
        >>> cache = SpatialIndexCache()
        >>> index = cache.get_or_build('fields', polygons_dict, SpatialIndex)
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(get_cache_config().cache_dir, 'spatial_indices')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _make_key(self, name: str, data: Any) -> str:
        """Create cache key from name and data hash."""
        data_hash = compute_hash(data)
        return f"{name}_{data_hash}"

    def get(self, name: str, data: Any) -> Optional[Any]:
        """Get cached index."""
        key = self._make_key(name, data)

        if key in self._memory_cache:
            return self._memory_cache[key]

        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    index = pickle.load(f)
                    self._memory_cache[key] = index
                    return index
            except:
                pass

        return None

    def set(self, name: str, data: Any, index: Any):
        """Cache an index."""
        key = self._make_key(name, data)

        self._memory_cache[key] = index

        path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(index, f)
        except:
            pass  # Some indices may not be picklable

    def get_or_build(self, name: str, polygons: Dict, index_class,
                     **build_kwargs) -> Any:
        """Get from cache or build and cache."""
        index = self.get(name, polygons)
        if index is not None:
            return index

        # Build index
        index = index_class(**build_kwargs)
        index.add_polygons(polygons)
        index.build()

        self.set(name, polygons, index)
        return index

    def clear(self):
        """Clear all cached indices."""
        self._memory_cache.clear()
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, f))


# Global spatial index cache
_spatial_index_cache = None


def get_spatial_index_cache() -> SpatialIndexCache:
    """Get the global spatial index cache."""
    global _spatial_index_cache
    if _spatial_index_cache is None:
        _spatial_index_cache = SpatialIndexCache()
    return _spatial_index_cache


# =============================================================================
# Cache Management Functions
# =============================================================================

def clear_all_caches():
    """Clear all caches (memory and disk)."""
    get_memory_cache().clear()
    get_disk_cache().clear()
    get_polygon_cell_cache().clear()
    get_spatial_index_cache().clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        'memory': get_memory_cache().stats(),
        'disk': get_disk_cache().stats(),
        'polygon_cells': get_polygon_cell_cache().stats(),
    }


def print_cache_stats():
    """Print cache statistics."""
    stats = get_cache_stats()

    print("=== GeoFast Cache Statistics ===")

    print("\nMemory Cache:")
    print(f"  Entries: {stats['memory']['entries']}")
    print(f"  Size: {stats['memory']['size_mb']:.1f} MB / {stats['memory']['max_size_mb']:.1f} MB")
    print(f"  Utilization: {stats['memory']['utilization']*100:.1f}%")

    print("\nDisk Cache:")
    print(f"  Entries: {stats['disk']['entries']}")
    print(f"  Size: {stats['disk']['size_mb']:.1f} MB / {stats['disk']['max_size_mb']:.1f} MB")
    print(f"  Location: {stats['disk']['cache_dir']}")

    print("\nPolygon Cell Cache:")
    print(f"  Memory entries: {stats['polygon_cells']['memory_entries']}")
    print(f"  Disk entries: {stats['polygon_cells']['disk_entries']}")
    print(f"  Disk size: {stats['polygon_cells']['disk_size_mb']:.1f} MB")
