"""
Numba JIT-compiled primitives for high-performance geospatial operations.
These are the building blocks used by higher-level functions.
"""

import math
import numpy as np

try:
    from numba import njit, prange, float64, int32, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# =============================================================================
# Point-in-Polygon
# =============================================================================

@njit(cache=True)
def point_in_polygon(x, y, poly_x, poly_y):
    """
    Ray casting algorithm for point-in-polygon test.

    Args:
        x, y: Point coordinates
        poly_x, poly_y: Polygon vertices as numpy arrays

    Returns:
        bool: True if point is inside polygon
    """
    n = len(poly_x)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


@njit(parallel=True, cache=True)
def points_in_polygon_batch(points_x, points_y, poly_x, poly_y):
    """
    Batch point-in-polygon test using parallel execution.

    Args:
        points_x, points_y: Arrays of point coordinates
        poly_x, poly_y: Polygon vertices as numpy arrays

    Returns:
        numpy array of booleans
    """
    n = len(points_x)
    result = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        result[i] = point_in_polygon(points_x[i], points_y[i], poly_x, poly_y)

    return result


# =============================================================================
# Haversine Distance
# =============================================================================

@njit(cache=True)
def haversine_scalar(lat1, lon1, lat2, lon2, radius=3959.0):
    """
    Calculate haversine distance between two points.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
        radius: Earth radius (default 3959 miles, use 6371 for km)

    Returns:
        float: Distance in same units as radius
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)

    return 2 * radius * math.asin(math.sqrt(a))


@njit(parallel=True, cache=True)
def haversine_batch(lat1, lon1, lat2, lon2, radius=3959.0):
    """
    Batch haversine distance calculation.

    Args:
        lat1, lon1, lat2, lon2: Arrays of coordinates (degrees)
        radius: Earth radius (default 3959 miles)

    Returns:
        numpy array of distances
    """
    n = len(lat1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        result[i] = haversine_scalar(lat1[i], lon1[i], lat2[i], lon2[i], radius)

    return result


@njit(parallel=True, cache=True)
def haversine_matrix(lats1, lons1, lats2, lons2, radius=3959.0):
    """
    Compute distance matrix between two sets of points.

    Args:
        lats1, lons1: First set of points
        lats2, lons2: Second set of points
        radius: Earth radius

    Returns:
        2D numpy array of distances (n1 x n2)
    """
    n1 = len(lats1)
    n2 = len(lats2)
    result = np.empty((n1, n2), dtype=np.float64)

    for i in prange(n1):
        for j in range(n2):
            result[i, j] = haversine_scalar(lats1[i], lons1[i], lats2[j], lons2[j], radius)

    return result


# =============================================================================
# Douglas-Peucker Line Simplification
# =============================================================================

@njit(cache=True)
def perpendicular_distance(px, py, x1, y1, x2, y2):
    """
    Calculate perpendicular distance from point to line segment.
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


@njit(cache=True)
def douglas_peucker_mask(coords, epsilon):
    """
    Douglas-Peucker line simplification - returns mask of points to keep.

    Args:
        coords: Nx2 numpy array of coordinates
        epsilon: Simplification threshold

    Returns:
        Boolean mask array
    """
    n = len(coords)
    if n <= 2:
        return np.ones(n, dtype=np.bool_)

    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n-1] = True

    # Stack for iterative processing
    stack_start = np.empty(n, dtype=np.int32)
    stack_end = np.empty(n, dtype=np.int32)
    stack_ptr = 0

    stack_start[stack_ptr] = 0
    stack_end[stack_ptr] = n - 1
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        start = stack_start[stack_ptr]
        end = stack_end[stack_ptr]

        if end - start <= 1:
            continue

        max_dist = 0.0
        max_idx = start

        x1, y1 = coords[start, 0], coords[start, 1]
        x2, y2 = coords[end, 0], coords[end, 1]

        for i in range(start + 1, end):
            dist = perpendicular_distance(coords[i, 0], coords[i, 1], x1, y1, x2, y2)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            keep[max_idx] = True
            stack_start[stack_ptr] = start
            stack_end[stack_ptr] = max_idx
            stack_ptr += 1
            stack_start[stack_ptr] = max_idx
            stack_end[stack_ptr] = end
            stack_ptr += 1

    return keep


def douglas_peucker(coords, epsilon):
    """
    Douglas-Peucker line simplification.

    Args:
        coords: List of (x, y) tuples or Nx2 numpy array
        epsilon: Simplification threshold

    Returns:
        Simplified coordinates as numpy array
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords, dtype=np.float64)

    if len(coords) <= 2:
        return coords

    mask = douglas_peucker_mask(coords, epsilon)
    return coords[mask]


# =============================================================================
# Hex Grid Operations
# =============================================================================

@njit(cache=True)
def hex_round(q, r):
    """Round fractional axial coordinates to nearest hex."""
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

    return int(rq), int(rr)


@njit(cache=True)
def lat_lon_to_hex(lat, lon, hex_size_lat, hex_size_lon):
    """
    Convert lat/lon to hex axial coordinates.

    Args:
        lat, lon: Coordinates in degrees
        hex_size_lat, hex_size_lon: Hex size in degrees

    Returns:
        (q, r) tuple of hex coordinates
    """
    sqrt3 = 1.7320508075688772

    x = lon / hex_size_lon
    y = lat / hex_size_lat

    q = (2.0/3.0) * x
    r = (-1.0/3.0) * x + (sqrt3/3.0) * y

    return hex_round(q, r)


@njit(parallel=True, cache=True)
def lat_lon_to_hex_batch(lats, lons, hex_size_lat, hex_size_lon):
    """
    Batch convert lat/lon to hex coordinates.

    Args:
        lats, lons: Arrays of coordinates
        hex_size_lat, hex_size_lon: Hex size in degrees

    Returns:
        (q_array, r_array) tuple of hex coordinate arrays
    """
    n = len(lats)
    q_arr = np.empty(n, dtype=np.int32)
    r_arr = np.empty(n, dtype=np.int32)

    for i in prange(n):
        q, r = lat_lon_to_hex(lats[i], lons[i], hex_size_lat, hex_size_lon)
        q_arr[i] = q
        r_arr[i] = r

    return q_arr, r_arr


@njit(cache=True)
def hex_to_lat_lon(q, r, hex_size_lat, hex_size_lon):
    """
    Convert hex axial coordinates to lat/lon of hex center.

    Args:
        q, r: Hex axial coordinates
        hex_size_lat, hex_size_lon: Hex size in degrees

    Returns:
        (lat, lon) tuple
    """
    sqrt3 = 1.7320508075688772

    x = hex_size_lon * (3.0/2.0) * q
    y = hex_size_lat * (sqrt3/2.0 * q + sqrt3 * r)

    return y, x  # lat, lon


@njit(cache=True)
def hex_neighbors(q, r):
    """
    Get the 6 neighboring hex coordinates.

    Returns:
        List of (q, r) tuples
    """
    return [
        (q+1, r), (q-1, r),
        (q, r+1), (q, r-1),
        (q+1, r-1), (q-1, r+1)
    ]


@njit(cache=True)
def line_to_hex_cells(lat1, lon1, lat2, lon2, hex_size_lat, hex_size_lon):
    """
    Get all hex cells a line segment passes through.

    Returns:
        Arrays of (q, r) coordinates
    """
    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)

    steps = max(int(dlat / hex_size_lat) + 1, int(dlon / hex_size_lon) + 1, 1) * 2

    # Pre-allocate for max possible cells
    q_arr = np.empty(steps + 1, dtype=np.int32)
    r_arr = np.empty(steps + 1, dtype=np.int32)

    count = 0
    prev_q, prev_r = -999999, -999999

    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0.0
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)

        q, r = lat_lon_to_hex(lat, lon, hex_size_lat, hex_size_lon)

        # Deduplicate consecutive cells
        if q != prev_q or r != prev_r:
            q_arr[count] = q
            r_arr[count] = r
            count += 1
            prev_q, prev_r = q, r

    return q_arr[:count], r_arr[:count]


@njit(cache=True)
def polygon_to_hex_cells(poly_x, poly_y, hex_size_lat, hex_size_lon):
    """
    Get all hex cells inside a polygon.

    Args:
        poly_x, poly_y: Polygon vertices (lon, lat order for x, y)
        hex_size_lat, hex_size_lon: Hex size in degrees

    Returns:
        Arrays of (q, r) coordinates
    """
    min_lat, max_lat = poly_y.min(), poly_y.max()
    min_lon, max_lon = poly_x.min(), poly_x.max()

    step_lat = hex_size_lat * 0.5
    step_lon = hex_size_lon * 0.5

    n_lat = int((max_lat - min_lat) / step_lat) + 1
    n_lon = int((max_lon - min_lon) / step_lon) + 1

    # Pre-allocate for max possible cells
    max_cells = n_lat * n_lon
    q_arr = np.empty(max_cells, dtype=np.int32)
    r_arr = np.empty(max_cells, dtype=np.int32)

    count = 0
    sqrt3 = 1.7320508075688772

    for i in range(n_lat):
        lat = min_lat + i * step_lat
        for j in range(n_lon):
            lon = min_lon + j * step_lon

            if point_in_polygon(lon, lat, poly_x, poly_y):
                # Inline hex calculation for speed
                x = lon / hex_size_lon
                y = lat / hex_size_lat
                qf = (2.0/3.0) * x
                rf = (-1.0/3.0) * x + (sqrt3/3.0) * y

                q, r = hex_round(qf, rf)
                q_arr[count] = q
                r_arr[count] = r
                count += 1

    return q_arr[:count], r_arr[:count]


# =============================================================================
# Bounding Box Operations
# =============================================================================

@njit(cache=True)
def bbox_intersects(min1_x, min1_y, max1_x, max1_y,
                    min2_x, min2_y, max2_x, max2_y):
    """Check if two bounding boxes intersect."""
    return (min1_x <= max2_x and max1_x >= min2_x and
            min1_y <= max2_y and max1_y >= min2_y)


@njit(cache=True)
def point_in_bbox(x, y, min_x, min_y, max_x, max_y):
    """Check if point is inside bounding box."""
    return min_x <= x <= max_x and min_y <= y <= max_y


@njit(parallel=True, cache=True)
def points_in_bbox_batch(points_x, points_y, min_x, min_y, max_x, max_y):
    """Batch check points in bounding box."""
    n = len(points_x)
    result = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        result[i] = point_in_bbox(points_x[i], points_y[i], min_x, min_y, max_x, max_y)

    return result
