"""
Utility functions for GeoFast
"""

from typing import Dict, List, Any, Iterator
import os


def detect_backends() -> Dict[str, bool]:
    """
    Detect which backends are available on this system.
    Returns dict of backend name -> availability.
    """
    backends = {
        'numpy': False,
        'cupy': False,
        'cuspatial': False,
        'numba': False,
        'numba_cuda': False,
        'geopandas': False,
        'shapely': False,
    }
    
    # NumPy (should always be available)
    try:
        import numpy
        backends['numpy'] = True
    except ImportError:
        pass
    
    # CuPy (GPU arrays)
    try:
        import cupy
        # Try a simple operation to verify GPU works
        cupy.array([1, 2, 3])
        backends['cupy'] = True
    except (ImportError, Exception):
        pass
    
    # cuSpatial (GPU geospatial)
    try:
        import cuspatial
        backends['cuspatial'] = True
    except ImportError:
        pass
    
    # Numba (JIT)
    try:
        import numba
        backends['numba'] = True
    except ImportError:
        pass
    
    # Numba CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            backends['numba_cuda'] = True
    except (ImportError, Exception):
        pass
    
    # GeoPandas
    try:
        import geopandas
        backends['geopandas'] = True
    except ImportError:
        pass
    
    # Shapely
    try:
        import shapely
        backends['shapely'] = True
    except ImportError:
        pass
    
    return backends


def get_optimal_backend(
    item_count: int, 
    available: Dict[str, bool],
    operation_type: str = 'general'
) -> str:
    """
    Suggest optimal backend based on item count and operation type.
    
    Args:
        item_count: Number of items to process
        available: Dict from detect_backends()
        operation_type: 'io', 'compute', 'spatial', 'general'
    
    Returns:
        Backend name string
    """
    # I/O operations -> parallel CPU is usually best
    if operation_type == 'io':
        return 'CPU_PARALLEL' if item_count > 10 else 'CPU'
    
    # Spatial operations
    if operation_type == 'spatial':
        if item_count > 5000 and available.get('cuspatial'):
            return 'GPU'
        if item_count > 100:
            return 'CPU_PARALLEL'
        return 'CPU'
    
    # Compute-heavy operations
    if operation_type == 'compute':
        if item_count > 10000 and available.get('cupy'):
            return 'GPU'
        if item_count > 1000 and available.get('numba'):
            return 'NUMBA'
        if item_count > 100:
            return 'CPU_PARALLEL'
        return 'CPU'
    
    # General
    if item_count > 10000 and available.get('cupy'):
        return 'GPU'
    if item_count > 100:
        return 'CPU_PARALLEL'
    return 'CPU'


def chunk_data(data: Any, chunk_size: int) -> List[Any]:
    """
    Split data into chunks for parallel processing.
    Works with lists, numpy arrays, and dataframes.
    """
    n = len(data)
    
    if n <= chunk_size:
        return [data]
    
    # For lists
    if isinstance(data, list):
        return [data[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    # For numpy arrays
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return [data[i:i + chunk_size] for i in range(0, n, chunk_size)]
    except ImportError:
        pass
    
    # For pandas DataFrames
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return [data.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
    except ImportError:
        pass
    
    # For GeoDataFrames
    try:
        import geopandas as gpd
        if isinstance(data, gpd.GeoDataFrame):
            return [data.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
    except ImportError:
        pass
    
    # Fallback: convert to list
    return [list(data)[i:i + chunk_size] for i in range(0, n, chunk_size)]


def log(message: str):
    """Log if verbose mode is enabled"""
    from .core import get_config
    if get_config().verbose:
        print(f"[GeoFast] {message}")


def estimate_memory_usage(data: Any) -> int:
    """
    Estimate memory usage of data in bytes.
    Useful for GPU memory management.
    """
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.nbytes
    except ImportError:
        pass
    
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
    except ImportError:
        pass
    
    # Rough estimate for lists
    if isinstance(data, list):
        if len(data) > 0:
            # Assume each element is ~100 bytes on average
            return len(data) * 100
        return 0
    
    return 0


def print_system_info():
    """Print detected system capabilities"""
    backends = detect_backends()
    
    print("=" * 50)
    print("GeoFast System Info")
    print("=" * 50)
    print(f"CPU Cores: {os.cpu_count()}")
    print()
    print("Available Backends:")
    for name, available in backends.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    # GPU info if available
    if backends.get('cupy'):
        try:
            import cupy as cp
            device = cp.cuda.Device()
            print()
            print("GPU Info:")
            print(f"  Name: {device.name}")
            print(f"  Memory: {device.mem_info[1] / 1024**3:.1f} GB")
        except Exception:
            pass
    
    print("=" * 50)
