"""
CUDA GPU kernels for high-performance geospatial operations.
Requires: numba with CUDA support, or CuPy.
"""

import math
import numpy as np

# Check for CUDA availability
NUMBA_CUDA_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    from numba import cuda
    if cuda.is_available():
        NUMBA_CUDA_AVAILABLE = True
except ImportError:
    pass

try:
    import cupy as cp
    # Test that GPU actually works
    cp.array([1])
    CUPY_AVAILABLE = True
except (ImportError, Exception):
    pass


# =============================================================================
# Numba CUDA Kernels
# =============================================================================

if NUMBA_CUDA_AVAILABLE:
    from numba import cuda

    @cuda.jit
    def _hex_keys_kernel(lats, lons, out_q, out_r, hex_size_lat, hex_size_lon):
        """
        CUDA kernel for batch hex key computation.
        Each thread processes one coordinate pair.
        """
        i = cuda.grid(1)
        if i >= lats.shape[0]:
            return

        sqrt3 = 1.7320508075688772

        x = lons[i] / hex_size_lon
        y = lats[i] / hex_size_lat

        q = (2.0/3.0) * x
        r = (-1.0/3.0) * x + (sqrt3/3.0) * y

        # Hex round
        s = -q - r
        rq = round(q)
        rr = round(r)
        rs = round(s)

        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)

        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs

        out_q[i] = int(rq)
        out_r[i] = int(rr)

    @cuda.jit
    def _haversine_kernel(lat1, lon1, lat2, lon2, out, radius):
        """
        CUDA kernel for batch haversine distance calculation.
        """
        i = cuda.grid(1)
        if i >= lat1.shape[0]:
            return

        # Convert to radians
        lat1_rad = lat1[i] * 0.017453292519943295  # pi/180
        lon1_rad = lon1[i] * 0.017453292519943295
        lat2_rad = lat2[i] * 0.017453292519943295
        lon2_rad = lon2[i] * 0.017453292519943295

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon/2)**2)

        out[i] = 2 * radius * math.asin(math.sqrt(a))

    @cuda.jit
    def _haversine_matrix_kernel(lats1, lons1, lats2, lons2, out, radius):
        """
        CUDA kernel for haversine distance matrix.
        2D grid: each thread computes one distance.
        """
        i, j = cuda.grid(2)
        n1 = lats1.shape[0]
        n2 = lats2.shape[0]

        if i >= n1 or j >= n2:
            return

        lat1_rad = lats1[i] * 0.017453292519943295
        lon1_rad = lons1[i] * 0.017453292519943295
        lat2_rad = lats2[j] * 0.017453292519943295
        lon2_rad = lons2[j] * 0.017453292519943295

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon/2)**2)

        out[i, j] = 2 * radius * math.asin(math.sqrt(a))

    @cuda.jit
    def _point_in_bbox_kernel(points_x, points_y, min_x, min_y, max_x, max_y, out):
        """
        CUDA kernel for batch point-in-bbox test.
        """
        i = cuda.grid(1)
        if i >= points_x.shape[0]:
            return

        x = points_x[i]
        y = points_y[i]
        out[i] = (min_x <= x <= max_x) and (min_y <= y <= max_y)


# =============================================================================
# High-level GPU functions
# =============================================================================

def hex_keys_gpu(lats, lons, hex_size_lat, hex_size_lon):
    """
    Compute hex keys for arrays of coordinates on GPU.

    Args:
        lats, lons: NumPy arrays of coordinates
        hex_size_lat, hex_size_lon: Hex size in degrees

    Returns:
        (q_array, r_array) as NumPy arrays
    """
    if not NUMBA_CUDA_AVAILABLE:
        # Fallback to CuPy if available
        if CUPY_AVAILABLE:
            return _hex_keys_cupy(lats, lons, hex_size_lat, hex_size_lon)
        else:
            raise RuntimeError("No GPU backend available")

    n = len(lats)

    # Allocate device arrays
    d_lats = cuda.to_device(np.asarray(lats, dtype=np.float64))
    d_lons = cuda.to_device(np.asarray(lons, dtype=np.float64))
    d_q = cuda.device_array(n, dtype=np.int32)
    d_r = cuda.device_array(n, dtype=np.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    _hex_keys_kernel[blocks, threads_per_block](
        d_lats, d_lons, d_q, d_r, hex_size_lat, hex_size_lon
    )

    return d_q.copy_to_host(), d_r.copy_to_host()


def haversine_gpu(lat1, lon1, lat2, lon2, radius=3959.0):
    """
    Compute haversine distances on GPU.

    Args:
        lat1, lon1, lat2, lon2: NumPy arrays of coordinates
        radius: Earth radius (default 3959 miles)

    Returns:
        NumPy array of distances
    """
    if CUPY_AVAILABLE:
        return _haversine_cupy(lat1, lon1, lat2, lon2, radius)

    if not NUMBA_CUDA_AVAILABLE:
        raise RuntimeError("No GPU backend available")

    n = len(lat1)

    d_lat1 = cuda.to_device(np.asarray(lat1, dtype=np.float64))
    d_lon1 = cuda.to_device(np.asarray(lon1, dtype=np.float64))
    d_lat2 = cuda.to_device(np.asarray(lat2, dtype=np.float64))
    d_lon2 = cuda.to_device(np.asarray(lon2, dtype=np.float64))
    d_out = cuda.device_array(n, dtype=np.float64)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    _haversine_kernel[blocks, threads_per_block](
        d_lat1, d_lon1, d_lat2, d_lon2, d_out, radius
    )

    return d_out.copy_to_host()


def haversine_matrix_gpu(lats1, lons1, lats2, lons2, radius=3959.0):
    """
    Compute distance matrix between two sets of points on GPU.

    Args:
        lats1, lons1: First set of coordinates
        lats2, lons2: Second set of coordinates
        radius: Earth radius

    Returns:
        2D NumPy array (n1 x n2) of distances
    """
    if CUPY_AVAILABLE:
        return _haversine_matrix_cupy(lats1, lons1, lats2, lons2, radius)

    if not NUMBA_CUDA_AVAILABLE:
        raise RuntimeError("No GPU backend available")

    n1 = len(lats1)
    n2 = len(lats2)

    d_lats1 = cuda.to_device(np.asarray(lats1, dtype=np.float64))
    d_lons1 = cuda.to_device(np.asarray(lons1, dtype=np.float64))
    d_lats2 = cuda.to_device(np.asarray(lats2, dtype=np.float64))
    d_lons2 = cuda.to_device(np.asarray(lons2, dtype=np.float64))
    d_out = cuda.device_array((n1, n2), dtype=np.float64)

    threads_per_block = (16, 16)
    blocks_x = (n1 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (n2 + threads_per_block[1] - 1) // threads_per_block[1]

    _haversine_matrix_kernel[(blocks_x, blocks_y), threads_per_block](
        d_lats1, d_lons1, d_lats2, d_lons2, d_out, radius
    )

    return d_out.copy_to_host()


# =============================================================================
# CuPy implementations (alternative GPU backend)
# =============================================================================

def _hex_keys_cupy(lats, lons, hex_size_lat, hex_size_lon):
    """Hex key computation using CuPy."""
    lats = cp.asarray(lats, dtype=cp.float64)
    lons = cp.asarray(lons, dtype=cp.float64)

    sqrt3 = 1.7320508075688772

    x = lons / hex_size_lon
    y = lats / hex_size_lat

    q = (2.0/3.0) * x
    r = (-1.0/3.0) * x + (sqrt3/3.0) * y

    # Hex round (vectorized)
    s = -q - r
    rq = cp.round(q).astype(cp.int32)
    rr = cp.round(r).astype(cp.int32)
    rs = cp.round(s).astype(cp.int32)

    q_diff = cp.abs(rq - q)
    r_diff = cp.abs(rr - r)
    s_diff = cp.abs(rs - s)

    mask_q = (q_diff > r_diff) & (q_diff > s_diff)
    mask_r = ~mask_q & (r_diff > s_diff)

    rq = cp.where(mask_q, -rr - rs, rq)
    rr = cp.where(mask_r, -rq - rs, rr)

    return cp.asnumpy(rq), cp.asnumpy(rr)


def _haversine_cupy(lat1, lon1, lat2, lon2, radius=3959.0):
    """Haversine distance using CuPy."""
    lat1 = cp.asarray(lat1, dtype=cp.float64)
    lon1 = cp.asarray(lon1, dtype=cp.float64)
    lat2 = cp.asarray(lat2, dtype=cp.float64)
    lon2 = cp.asarray(lon2, dtype=cp.float64)

    lat1_rad = cp.radians(lat1)
    lon1_rad = cp.radians(lon1)
    lat2_rad = cp.radians(lat2)
    lon2_rad = cp.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (cp.sin(dlat/2)**2 +
         cp.cos(lat1_rad) * cp.cos(lat2_rad) *
         cp.sin(dlon/2)**2)

    result = 2 * radius * cp.arcsin(cp.sqrt(a))
    return cp.asnumpy(result)


def _haversine_matrix_cupy(lats1, lons1, lats2, lons2, radius=3959.0):
    """Haversine distance matrix using CuPy broadcasting."""
    lats1 = cp.asarray(lats1, dtype=cp.float64)[:, None]
    lons1 = cp.asarray(lons1, dtype=cp.float64)[:, None]
    lats2 = cp.asarray(lats2, dtype=cp.float64)[None, :]
    lons2 = cp.asarray(lons2, dtype=cp.float64)[None, :]

    lat1_rad = cp.radians(lats1)
    lon1_rad = cp.radians(lons1)
    lat2_rad = cp.radians(lats2)
    lon2_rad = cp.radians(lons2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (cp.sin(dlat/2)**2 +
         cp.cos(lat1_rad) * cp.cos(lat2_rad) *
         cp.sin(dlon/2)**2)

    result = 2 * radius * cp.arcsin(cp.sqrt(a))
    return cp.asnumpy(result)


# =============================================================================
# Utility functions
# =============================================================================

def gpu_available():
    """Check if any GPU backend is available."""
    return NUMBA_CUDA_AVAILABLE or CUPY_AVAILABLE


def get_gpu_info():
    """Get information about available GPU."""
    info = {
        'numba_cuda': NUMBA_CUDA_AVAILABLE,
        'cupy': CUPY_AVAILABLE,
        'device_name': None,
        'memory_total': None,
    }

    if CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(0)
            info['device_name'] = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            info['memory_total'] = device.mem_info[1]
        except Exception:
            pass

    if NUMBA_CUDA_AVAILABLE and info['device_name'] is None:
        try:
            info['device_name'] = cuda.get_current_device().name
        except Exception:
            pass

    return info


# =============================================================================
# Auto-selecting wrapper functions
# =============================================================================

def hex_keys_auto(lats, lons, hex_size_lat, hex_size_lon, min_gpu_size=10000):
    """
    Compute hex keys, automatically selecting GPU if beneficial.

    Args:
        lats, lons: Coordinate arrays
        hex_size_lat, hex_size_lon: Hex sizes
        min_gpu_size: Minimum array size to use GPU (default 10000)

    Returns:
        (q_array, r_array)
    """
    n = len(lats)

    if n >= min_gpu_size and gpu_available():
        try:
            return hex_keys_gpu(lats, lons, hex_size_lat, hex_size_lon)
        except Exception:
            pass

    # Fallback to CPU (use primitives)
    from .primitives import lat_lon_to_hex_batch
    return lat_lon_to_hex_batch(
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        hex_size_lat, hex_size_lon
    )


def haversine_auto(lat1, lon1, lat2, lon2, radius=3959.0, min_gpu_size=10000):
    """
    Compute haversine distances, automatically selecting GPU if beneficial.
    """
    n = len(lat1)

    if n >= min_gpu_size and gpu_available():
        try:
            return haversine_gpu(lat1, lon1, lat2, lon2, radius)
        except Exception:
            pass

    # Fallback to CPU
    from .primitives import haversine_batch
    return haversine_batch(
        np.asarray(lat1, dtype=np.float64),
        np.asarray(lon1, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64),
        np.asarray(lon2, dtype=np.float64),
        radius
    )
