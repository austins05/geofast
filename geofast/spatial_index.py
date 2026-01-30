"""
Spatial indexing for fast geometric queries.
Uses R-tree (via Shapely STRtree) for O(log n) lookups.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from collections import defaultdict

try:
    from shapely import STRtree, box, Point, Polygon
    from shapely.geometry import shape
    from shapely import contains, intersects, within
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from .primitives import (
    point_in_polygon, lat_lon_to_hex, hex_round,
    polygon_to_hex_cells, bbox_intersects
)


class SpatialIndex:
    """
    R-tree based spatial index for fast polygon queries.

    Example:
        >>> index = SpatialIndex()
        >>> index.add_polygon('field1', polygon_coords)
        >>> index.add_polygon('field2', polygon_coords)
        >>> index.build()
        >>> hits = index.query_point(lat, lon)
    """

    def __init__(self):
        self.polygons = {}  # id -> polygon coords
        self.geometries = []  # Shapely geometries
        self.ids = []  # Corresponding IDs
        self._tree = None
        self._built = False

        # Bounding boxes for fast pre-filter
        self.bboxes = {}  # id -> (min_x, min_y, max_x, max_y)

        # Numpy arrays for Numba
        self.poly_x_arrays = {}
        self.poly_y_arrays = {}

    def add_polygon(self, id: Any, coords: List[Tuple[float, float]]):
        """
        Add a polygon to the index.

        Args:
            id: Unique identifier for this polygon
            coords: List of (lon, lat) or (x, y) tuples
        """
        self.polygons[id] = coords

        # Pre-compute numpy arrays for Numba
        poly_x = np.array([p[0] for p in coords], dtype=np.float64)
        poly_y = np.array([p[1] for p in coords], dtype=np.float64)
        self.poly_x_arrays[id] = poly_x
        self.poly_y_arrays[id] = poly_y

        # Compute bounding box
        self.bboxes[id] = (poly_x.min(), poly_y.min(), poly_x.max(), poly_y.max())

        self._built = False

    def add_polygons(self, polygons: Dict[Any, List[Tuple[float, float]]]):
        """Add multiple polygons at once."""
        for id, coords in polygons.items():
            self.add_polygon(id, coords)

    def build(self):
        """Build the R-tree index. Call after adding all polygons."""
        if not SHAPELY_AVAILABLE:
            self._built = True
            return

        self.geometries = []
        self.ids = []

        for id, coords in self.polygons.items():
            try:
                geom = Polygon(coords)
                if geom.is_valid:
                    self.geometries.append(geom)
                    self.ids.append(id)
            except Exception:
                pass

        if self.geometries:
            self._tree = STRtree(self.geometries)

        self._built = True

    def query_point(self, x: float, y: float) -> List[Any]:
        """
        Find all polygons containing a point.

        Args:
            x, y: Point coordinates (lon, lat)

        Returns:
            List of polygon IDs containing the point
        """
        if not self._built:
            self.build()

        results = []

        if SHAPELY_AVAILABLE and self._tree is not None:
            # Use R-tree for O(log n) candidate lookup
            point = Point(x, y)
            candidates = self._tree.query(point)

            for idx in candidates:
                geom = self.geometries[idx]
                if contains(geom, point):
                    results.append(self.ids[idx])
        else:
            # Fallback: linear scan with bbox pre-filter
            for id, (min_x, min_y, max_x, max_y) in self.bboxes.items():
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    poly_x = self.poly_x_arrays[id]
                    poly_y = self.poly_y_arrays[id]
                    if point_in_polygon(x, y, poly_x, poly_y):
                        results.append(id)

        return results

    def query_point_first(self, x: float, y: float) -> Optional[Any]:
        """
        Find the first polygon containing a point.
        Faster than query_point when you only need one result.
        """
        if not self._built:
            self.build()

        if SHAPELY_AVAILABLE and self._tree is not None:
            point = Point(x, y)
            candidates = self._tree.query(point)

            for idx in candidates:
                geom = self.geometries[idx]
                if contains(geom, point):
                    return self.ids[idx]
        else:
            for id, (min_x, min_y, max_x, max_y) in self.bboxes.items():
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    poly_x = self.poly_x_arrays[id]
                    poly_y = self.poly_y_arrays[id]
                    if point_in_polygon(x, y, poly_x, poly_y):
                        return id

        return None

    def query_points_batch(self, x_arr: np.ndarray, y_arr: np.ndarray) -> List[List[Any]]:
        """
        Batch query for multiple points.

        Args:
            x_arr, y_arr: Arrays of coordinates

        Returns:
            List of lists of polygon IDs for each point
        """
        if not self._built:
            self.build()

        results = []

        if SHAPELY_AVAILABLE and self._tree is not None:
            from shapely import points as make_points

            pts = make_points(np.column_stack([x_arr, y_arr]))

            for i, pt in enumerate(pts):
                candidates = self._tree.query(pt)
                hits = []
                for idx in candidates:
                    if contains(self.geometries[idx], pt):
                        hits.append(self.ids[idx])
                results.append(hits)
        else:
            for i in range(len(x_arr)):
                results.append(self.query_point(x_arr[i], y_arr[i]))

        return results

    def query_bbox(self, min_x: float, min_y: float,
                   max_x: float, max_y: float) -> List[Any]:
        """
        Find all polygons intersecting a bounding box.
        """
        if not self._built:
            self.build()

        results = []

        if SHAPELY_AVAILABLE and self._tree is not None:
            query_box = box(min_x, min_y, max_x, max_y)
            candidates = self._tree.query(query_box)

            for idx in candidates:
                if intersects(self.geometries[idx], query_box):
                    results.append(self.ids[idx])
        else:
            for id, bbox in self.bboxes.items():
                if bbox_intersects(min_x, min_y, max_x, max_y, *bbox):
                    results.append(id)

        return results


class HexGridIndex:
    """
    Spatial index using hex grid cells.
    Maps hex cells to polygon IDs for O(1) point-to-polygon lookup.

    Example:
        >>> index = HexGridIndex(hex_size_lat=0.001, hex_size_lon=0.001)
        >>> index.add_polygon('field1', polygon_coords)
        >>> index.build()
        >>> field_id = index.query_point(lat, lon)
    """

    def __init__(self, hex_size_lat: float, hex_size_lon: float):
        """
        Args:
            hex_size_lat: Hex size in degrees latitude
            hex_size_lon: Hex size in degrees longitude
        """
        self.hex_size_lat = hex_size_lat
        self.hex_size_lon = hex_size_lon

        self.polygons = {}  # id -> coords
        self.cell_to_ids = defaultdict(set)  # (q, r) -> set of polygon IDs
        self._built = False

    def add_polygon(self, id: Any, coords: List[Tuple[float, float]]):
        """Add a polygon to the index."""
        self.polygons[id] = coords
        self._built = False

    def add_polygons(self, polygons: Dict[Any, List[Tuple[float, float]]]):
        """Add multiple polygons."""
        for id, coords in polygons.items():
            self.add_polygon(id, coords)

    def build(self):
        """Build the hex grid index."""
        self.cell_to_ids.clear()

        for id, coords in self.polygons.items():
            poly_x = np.array([p[0] for p in coords], dtype=np.float64)
            poly_y = np.array([p[1] for p in coords], dtype=np.float64)

            q_arr, r_arr = polygon_to_hex_cells(
                poly_x, poly_y,
                self.hex_size_lat, self.hex_size_lon
            )

            for i in range(len(q_arr)):
                self.cell_to_ids[(q_arr[i], r_arr[i])].add(id)

        self._built = True

    def query_point(self, lat: float, lon: float) -> Set[Any]:
        """
        Find all polygons whose hex cells contain this point.
        O(1) lookup time.

        Args:
            lat, lon: Point coordinates

        Returns:
            Set of polygon IDs
        """
        if not self._built:
            self.build()

        q, r = lat_lon_to_hex(lat, lon, self.hex_size_lat, self.hex_size_lon)
        return self.cell_to_ids.get((q, r), set())

    def query_point_first(self, lat: float, lon: float) -> Optional[Any]:
        """Get first polygon ID at this point, or None."""
        hits = self.query_point(lat, lon)
        return next(iter(hits)) if hits else None

    def query_cell(self, q: int, r: int) -> Set[Any]:
        """Get polygon IDs for a specific hex cell."""
        if not self._built:
            self.build()
        return self.cell_to_ids.get((q, r), set())

    def get_polygon_cells(self, id: Any) -> Set[Tuple[int, int]]:
        """Get all hex cells for a polygon."""
        if id not in self.polygons:
            return set()

        coords = self.polygons[id]
        poly_x = np.array([p[0] for p in coords], dtype=np.float64)
        poly_y = np.array([p[1] for p in coords], dtype=np.float64)

        q_arr, r_arr = polygon_to_hex_cells(
            poly_x, poly_y,
            self.hex_size_lat, self.hex_size_lon
        )

        return {(q_arr[i], r_arr[i]) for i in range(len(q_arr))}


class PointIndex:
    """
    Spatial index for point data using R-tree.
    Fast nearest-neighbor and range queries.
    """

    def __init__(self):
        self.points = []  # List of (x, y)
        self.ids = []  # Corresponding IDs
        self.data = {}  # id -> associated data
        self._tree = None
        self._built = False

    def add_point(self, id: Any, x: float, y: float, data: Any = None):
        """Add a point to the index."""
        self.points.append((x, y))
        self.ids.append(id)
        if data is not None:
            self.data[id] = data
        self._built = False

    def add_points(self, points: List[Tuple[Any, float, float, Any]]):
        """Add multiple points: [(id, x, y, data), ...]"""
        for item in points:
            if len(item) == 4:
                id, x, y, data = item
                self.add_point(id, x, y, data)
            else:
                id, x, y = item
                self.add_point(id, x, y)

    def build(self):
        """Build the R-tree index."""
        if not SHAPELY_AVAILABLE or not self.points:
            self._built = True
            return

        from shapely import points as make_points
        geoms = make_points(self.points)
        self._tree = STRtree(geoms)
        self._built = True

    def query_nearest(self, x: float, y: float, k: int = 1) -> List[Tuple[Any, float]]:
        """
        Find k nearest points.

        Returns:
            List of (id, distance) tuples
        """
        if not self._built:
            self.build()

        if not SHAPELY_AVAILABLE or self._tree is None:
            # Fallback: brute force
            distances = []
            for i, (px, py) in enumerate(self.points):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                distances.append((self.ids[i], dist))
            distances.sort(key=lambda x: x[1])
            return distances[:k]

        query_point = Point(x, y)
        indices = self._tree.nearest(query_point, k)

        results = []
        for idx in (indices if hasattr(indices, '__iter__') else [indices]):
            px, py = self.points[idx]
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            results.append((self.ids[idx], dist))

        return sorted(results, key=lambda x: x[1])

    def query_radius(self, x: float, y: float, radius: float) -> List[Tuple[Any, float]]:
        """
        Find all points within radius.

        Returns:
            List of (id, distance) tuples
        """
        if not self._built:
            self.build()

        results = []

        if SHAPELY_AVAILABLE and self._tree is not None:
            from shapely import buffer
            query_circle = Point(x, y).buffer(radius)
            candidates = self._tree.query(query_circle)

            for idx in candidates:
                px, py = self.points[idx]
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist <= radius:
                    results.append((self.ids[idx], dist))
        else:
            for i, (px, py) in enumerate(self.points):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist <= radius:
                    results.append((self.ids[i], dist))

        return sorted(results, key=lambda x: x[1])
