#!/usr/bin/env python3
"""
Spray Pattern Optimizer for GeoFast.

Generates optimal spray/application lines for agricultural fields,
maximizing acres per hour while handling obstacles and various constraints.

Supports both aerial (with hops) and ground operations (no hops).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os

import numpy as np

try:
    from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPolygon
    from shapely.ops import linemerge, unary_union
    from shapely import affinity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SprayConfig:
    """Configuration for spray pattern optimization.

    Attributes:
        swath_width_ft: Width of spray coverage per pass (default 50ft)
        spray_speed_mph: Speed while spraying (default 68 mph)
        ferry_speed_mph: Speed when not spraying/hopping (default 55 mph)
        turn_time_sec: Time penalty for each turn (default 9 seconds)
        hop_distance_ft: Max distance to hop without spraying (default 1300ft)
        hop_enabled: Whether hopping is allowed (False for ground ops)
        headland_ft: Buffer distance from field edge (default 0)
        angle_step_deg: Angle increment for optimization search (default 5)
        angle_change_penalty_min: Time penalty for changing spray angle between fields (default 2 min)
        group_distance_ft: Max distance between fields to consider grouping (default 1300ft)
        max_workers: Max parallel workers for multiprocessing (default: CPU count)
    """
    swath_width_ft: float = 50.0
    spray_speed_mph: float = 68.0
    ferry_speed_mph: float = 55.0
    turn_time_sec: float = 9.0
    hop_distance_ft: float = 1300.0
    hop_enabled: bool = True
    headland_ft: float = 0.0
    angle_step_deg: float = 5.0
    angle_change_penalty_min: float = 2.0
    group_distance_ft: float = 1300.0
    max_workers: int = None  # None = use CPU count


# ============================================================================
# Unit Conversion Helpers
# ============================================================================

# Approximate feet per degree at mid-latitudes (US)
# More accurate would use pyproj, but this is fast and reasonable
FEET_PER_DEGREE_LAT = 364000  # ~69 miles
FEET_PER_DEGREE_LON = 288000  # ~55 miles at 40°N latitude


def ft_to_deg_lat(feet: float) -> float:
    """Convert feet to degrees latitude."""
    return feet / FEET_PER_DEGREE_LAT


def ft_to_deg_lon(feet: float, lat: float = 40.0) -> float:
    """Convert feet to degrees longitude (varies with latitude)."""
    # Adjust for latitude
    cos_lat = math.cos(math.radians(lat))
    return feet / (FEET_PER_DEGREE_LAT * cos_lat)


def ft_to_deg_at_angle(feet: float, angle_deg: float, lat: float = 40.0) -> float:
    """
    Convert feet to degrees for a direction rotated by angle from north.

    At 0°: spacing is in N-S direction (latitude)
    At 90°: spacing is in E-W direction (longitude)

    This correctly handles the lat/lon distortion at different rotation angles.
    """
    angle_rad = math.radians(angle_deg)
    cos_lat = math.cos(math.radians(lat))

    ft_per_deg_lat = FEET_PER_DEGREE_LAT
    ft_per_deg_lon = FEET_PER_DEGREE_LAT * cos_lat

    # For a displacement dy in the rotated Y direction:
    # Original space: (dy*sin(angle), dy*cos(angle))
    # Physical distance: sqrt((dy*sin*ft_lon)^2 + (dy*cos*ft_lat)^2)
    # To get 'feet' physical distance, solve for dy:
    effective_ft_per_deg = math.sqrt(
        (math.sin(angle_rad) * ft_per_deg_lon)**2 +
        (math.cos(angle_rad) * ft_per_deg_lat)**2
    )

    return feet / effective_ft_per_deg


def deg_to_ft_lat(deg: float) -> float:
    """Convert degrees latitude to feet."""
    return deg * FEET_PER_DEGREE_LAT


def deg_to_ft_lon(deg: float, lat: float = 40.0) -> float:
    """Convert degrees longitude to feet."""
    cos_lat = math.cos(math.radians(lat))
    return deg * FEET_PER_DEGREE_LAT * cos_lat


def calculate_line_length_ft(line: 'LineString', centroid_lat: float = 40.0) -> float:
    """Calculate line length in feet (approximate)."""
    coords = list(line.coords)
    total_ft = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = deg_to_ft_lat(lat2 - lat1)
        dlon = deg_to_ft_lon(lon2 - lon1, centroid_lat)
        total_ft += math.sqrt(dlat**2 + dlon**2)
    return total_ft


def calculate_polygon_area_acres(polygon: 'Polygon', centroid_lat: float = 40.0) -> float:
    """Calculate polygon area in acres (approximate)."""
    # Get area in square degrees
    area_sq_deg = polygon.area

    # Convert to square feet
    ft_per_deg_lat = FEET_PER_DEGREE_LAT
    ft_per_deg_lon = FEET_PER_DEGREE_LAT * math.cos(math.radians(centroid_lat))
    area_sq_ft = area_sq_deg * ft_per_deg_lat * ft_per_deg_lon

    # Convert to acres (43560 sq ft per acre)
    return area_sq_ft / 43560


# ============================================================================
# Core Line Generation
# ============================================================================

def _ensure_polygon(geom) -> 'Polygon':
    """Convert various input types to a Shapely Polygon."""
    if not SHAPELY_AVAILABLE:
        raise ImportError("shapely is required for spray pattern optimization. "
                          "Install with: pip install shapely")

    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, dict):
        # GeoJSON-like dict
        if geom.get('type') == 'Polygon':
            coords = geom.get('coordinates', [[]])
            return Polygon(coords[0])
        elif geom.get('type') == 'Feature':
            return _ensure_polygon(geom.get('geometry', {}))
    elif isinstance(geom, (list, tuple)):
        # List of coordinates
        return Polygon(geom)

    raise ValueError(f"Cannot convert {type(geom)} to Polygon")


def _ensure_obstacles(obstacles) -> List['Polygon']:
    """Convert various input types to a list of Shapely Polygons."""
    if obstacles is None:
        return []

    result = []
    if isinstance(obstacles, (Polygon, MultiPolygon)):
        if isinstance(obstacles, MultiPolygon):
            result.extend(list(obstacles.geoms))
        else:
            result.append(obstacles)
    elif isinstance(obstacles, list):
        for obs in obstacles:
            result.append(_ensure_polygon(obs))
    else:
        result.append(_ensure_polygon(obstacles))

    return result


def generate_parallel_lines(
    polygon: Union['Polygon', dict, list],
    angle_deg: float,
    swath_width_ft: float,
    headland_ft: float = 0.0
) -> List[List['LineString']]:
    """
    Generate parallel lines across a polygon at a given angle.

    Algorithm:
    1. If headland > 0, buffer polygon inward
    2. Rotate polygon by -angle to align with x-axis
    3. Get bounding box of rotated polygon
    4. Generate horizontal lines spaced by swath_width
    5. Clip lines to rotated polygon boundary
    6. Rotate lines back by +angle

    Args:
        polygon: Field boundary (Shapely Polygon, GeoJSON dict, or coordinate list)
        angle_deg: Angle of lines in degrees (0 = east-west, 90 = north-south)
        swath_width_ft: Spacing between lines in feet
        headland_ft: Buffer from field edge in feet (default 0)

    Returns:
        List of "tracks" where each track is a list of LineString segments.
        Segments within the same track are on the same pass (gaps = hops).
        Different tracks require turns.
    """
    poly = _ensure_polygon(polygon)

    if not poly.is_valid:
        poly = poly.buffer(0)  # Fix invalid geometries

    # Get centroid for coordinate conversions
    centroid = poly.centroid
    centroid_lat = centroid.y

    # Apply headland buffer if specified
    if headland_ft > 0:
        buffer_deg = ft_to_deg_lat(headland_ft)
        poly = poly.buffer(-buffer_deg)
        if poly.is_empty:
            return []

    # Convert swath width to degrees, accounting for lat/lon distortion at this angle
    # This is critical: at 90° the spacing is in longitude direction which has
    # different ft-per-degree than latitude
    swath_deg = ft_to_deg_at_angle(swath_width_ft, angle_deg, centroid_lat)

    # Rotate polygon to align lines with x-axis
    rotated_poly = affinity.rotate(poly, -angle_deg, origin=centroid)

    # Get bounding box of rotated polygon
    minx, miny, maxx, maxy = rotated_poly.bounds

    # Generate horizontal lines - each "track" may have multiple segments
    tracks = []  # List of lists - each inner list is segments on same track
    y = miny + swath_deg / 2  # Start half-swath from edge

    while y < maxy:
        # Create horizontal line across full width
        line = LineString([(minx - swath_deg, y), (maxx + swath_deg, y)])

        # Clip to rotated polygon
        clipped = line.intersection(rotated_poly)

        if not clipped.is_empty:
            track_segments = []

            # Handle multi-line results (polygon with holes or concave)
            if isinstance(clipped, MultiLineString):
                for segment in clipped.geoms:
                    if segment.length > 0:
                        # Rotate back
                        rotated_line = affinity.rotate(segment, angle_deg, origin=centroid)
                        track_segments.append(rotated_line)
            elif isinstance(clipped, LineString) and clipped.length > 0:
                # Rotate back
                rotated_line = affinity.rotate(clipped, angle_deg, origin=centroid)
                track_segments.append(rotated_line)

            if track_segments:
                # Sort segments by x-coordinate (left to right after rotation back)
                track_segments.sort(key=lambda seg: min(seg.coords[0][0], seg.coords[-1][0]))
                tracks.append(track_segments)

        y += swath_deg

    return tracks


def flatten_tracks(tracks: List[List['LineString']]) -> List['LineString']:
    """Flatten tracks to a simple list of lines (for backward compatibility)."""
    lines = []
    for track in tracks:
        lines.extend(track)
    return lines


# ============================================================================
# Obstacle Handling
# ============================================================================

def handle_obstacles(
    lines: List['LineString'],
    obstacles: List['Polygon'],
    config: SprayConfig,
    centroid_lat: float = 40.0
) -> List[Dict[str, Any]]:
    """
    Process lines that cross obstacles (no-spray zones).

    For each line crossing an obstacle:
    1. Split line at obstacle boundary
    2. Create "spray_on" segments outside obstacle
    3. Create "spray_off" segment across obstacle

    Args:
        lines: List of spray lines
        obstacles: List of obstacle polygons
        config: Spray configuration
        centroid_lat: Latitude for distance calculations

    Returns:
        List of segment dicts with 'geometry' and 'spray_on' keys
    """
    if not obstacles:
        # No obstacles - all lines are spray-on
        return [{'geometry': line, 'spray_on': True} for line in lines]

    # Merge obstacles into single geometry for faster intersection
    obstacle_union = unary_union(obstacles)

    segments = []

    for line in lines:
        if not line.intersects(obstacle_union):
            # Line doesn't cross any obstacle
            segments.append({'geometry': line, 'spray_on': True})
            continue

        # Line crosses obstacle - split it
        # Get the part outside obstacles (spray on)
        outside = line.difference(obstacle_union)
        # Get the part inside obstacles (spray off / hop)
        inside = line.intersection(obstacle_union)

        # Process outside segments (spray on)
        if not outside.is_empty:
            if isinstance(outside, MultiLineString):
                for seg in outside.geoms:
                    if seg.length > 0:
                        segments.append({'geometry': seg, 'spray_on': True})
            elif isinstance(outside, LineString) and outside.length > 0:
                segments.append({'geometry': outside, 'spray_on': True})

        # Process inside segments (spray off / hop)
        if not inside.is_empty:
            if isinstance(inside, MultiLineString):
                for seg in inside.geoms:
                    if seg.length > 0:
                        seg_length_ft = calculate_line_length_ft(seg, centroid_lat)
                        # Check if within hop distance
                        if config.hop_enabled and seg_length_ft <= config.hop_distance_ft:
                            segments.append({'geometry': seg, 'spray_on': False, 'hop': True})
                        else:
                            segments.append({'geometry': seg, 'spray_on': False, 'hop': False})
            elif isinstance(inside, LineString) and inside.length > 0:
                seg_length_ft = calculate_line_length_ft(inside, centroid_lat)
                if config.hop_enabled and seg_length_ft <= config.hop_distance_ft:
                    segments.append({'geometry': inside, 'spray_on': False, 'hop': True})
                else:
                    segments.append({'geometry': inside, 'spray_on': False, 'hop': False})

    return segments


# ============================================================================
# Efficiency Calculation
# ============================================================================

def _get_track_endpoints(track: List['LineString']) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get the start and end points of a track (which may have multiple segments)."""
    # First segment's first point is track start
    start = track[0].coords[0]
    # Last segment's last point is track end
    end = track[-1].coords[-1]
    return start, end


def _distance_ft(coord1: Tuple[float, float], coord2: Tuple[float, float], centroid_lat: float) -> float:
    """Calculate distance in feet between two coordinates."""
    dlat = deg_to_ft_lat(coord2[1] - coord1[1])
    dlon = deg_to_ft_lon(coord2[0] - coord1[0], centroid_lat)
    return math.sqrt(dlat**2 + dlon**2)


def calculate_efficiency_from_tracks(
    tracks: List[List['LineString']],
    polygon: 'Polygon',
    config: SprayConfig,
) -> Dict[str, float]:
    """
    Calculate efficiency metrics for a spray pattern using track structure.

    A "track" is a single pass across the field. It may have multiple segments
    if the polygon is concave.

    Turn vs Hop logic:
    - Transitions between DIFFERENT tracks: ALWAYS a TURN (you must turn around
      at the end of each line to fly the next line in opposite direction)
    - Gaps between segments on the SAME track: HOP if distance ≤ hop_distance_ft
      (flying straight over a gap, spray off, without turning)

    Args:
        tracks: List of tracks, where each track is a list of line segments
        polygon: Field boundary polygon
        config: Spray configuration

    Returns:
        Dict with efficiency metrics
    """
    centroid_lat = polygon.centroid.y

    # Calculate field area
    acres = calculate_polygon_area_acres(polygon, centroid_lat)

    # Calculate distances and counts
    spray_distance_ft = 0.0
    hop_distance_ft = 0.0
    ferry_distance_ft = 0.0  # Distance flown during turns (not hopping)
    num_segments = 0
    num_hops = 0  # Gaps within same track where we hop over

    for track_idx, track in enumerate(tracks):
        # Process all segments in this track
        for i, segment in enumerate(track):
            spray_distance_ft += calculate_line_length_ft(segment, centroid_lat)
            num_segments += 1

            # If there's a next segment on this track, it's a potential hop
            if i < len(track) - 1:
                next_segment = track[i + 1]
                end_coord = segment.coords[-1]
                start_coord = next_segment.coords[0]

                gap_ft = _distance_ft(end_coord, start_coord, centroid_lat)

                # Gaps within same track can be hops (fly straight, spray off)
                if config.hop_enabled and gap_ft <= config.hop_distance_ft:
                    hop_distance_ft += gap_ft
                    num_hops += 1
                # If gap too large, we can't hop - would need to handle differently
                # For now, still count as hop distance but this is a limitation

        # Transition to next track - ALWAYS a turn (must turn around)
        if track_idx < len(tracks) - 1:
            next_track = tracks[track_idx + 1]

            # Get end of current track and start of next track
            _, current_end = _get_track_endpoints(track)
            next_start, next_end = _get_track_endpoints(next_track)

            # Pilot enters next track from whichever end is closer
            dist_to_start = _distance_ft(current_end, next_start, centroid_lat)
            dist_to_end = _distance_ft(current_end, next_end, centroid_lat)
            min_dist = min(dist_to_start, dist_to_end)

            # This distance is flown during the turn (at ferry speed)
            ferry_distance_ft += min_dist

    # Number of turns = number of track transitions (always turn at end of each line)
    num_turns = max(0, len(tracks) - 1)

    # Calculate time components
    spray_speed_ft_min = config.spray_speed_mph * 5280 / 60
    spray_time_min = spray_distance_ft / spray_speed_ft_min if spray_speed_ft_min > 0 else 0

    ferry_speed_ft_min = config.ferry_speed_mph * 5280 / 60
    hop_time_min = hop_distance_ft / ferry_speed_ft_min if ferry_speed_ft_min > 0 else 0
    ferry_time_min = ferry_distance_ft / ferry_speed_ft_min if ferry_speed_ft_min > 0 else 0

    # Turn time penalty (the actual turnaround maneuver)
    turn_time_min = num_turns * config.turn_time_sec / 60

    # Total time includes spray + hop + ferry + turn penalty
    total_time_min = spray_time_min + hop_time_min + ferry_time_min + turn_time_min

    acres_per_hour = acres / (total_time_min / 60) if total_time_min > 0 else 0

    return {
        'acres': acres,
        'spray_distance_ft': spray_distance_ft,
        'hop_distance_ft': hop_distance_ft,
        'ferry_distance_ft': ferry_distance_ft,
        'num_tracks': len(tracks),
        'num_segments': num_segments,
        'num_turns': num_turns,
        'num_hops': num_hops,
        'spray_time_min': spray_time_min,
        'hop_time_min': hop_time_min,
        'ferry_time_min': ferry_time_min,
        'turn_time_min': turn_time_min,
        'total_time_min': total_time_min,
        'acres_per_hour': acres_per_hour,
    }


def calculate_efficiency(
    lines: List['LineString'],
    polygon: 'Polygon',
    config: SprayConfig,
    obstacles: Optional[List['Polygon']] = None,
    segments: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Calculate efficiency metrics for a spray pattern (legacy interface).

    Args:
        lines: List of spray lines
        polygon: Field boundary polygon
        config: Spray configuration
        obstacles: Optional list of obstacle polygons
        segments: Pre-computed segments with spray_on info

    Returns:
        Dict with efficiency metrics
    """
    centroid_lat = polygon.centroid.y

    # Calculate field area
    acres = calculate_polygon_area_acres(polygon, centroid_lat)

    # Calculate distances
    spray_distance_ft = 0.0
    hop_distance_ft = 0.0

    if segments:
        for seg in segments:
            length = calculate_line_length_ft(seg['geometry'], centroid_lat)
            if seg.get('spray_on', True):
                spray_distance_ft += length
            else:
                hop_distance_ft += length
    else:
        for line in lines:
            spray_distance_ft += calculate_line_length_ft(line, centroid_lat)

    num_turns = max(0, len(lines) - 1)

    spray_speed_ft_min = config.spray_speed_mph * 5280 / 60
    spray_time_min = spray_distance_ft / spray_speed_ft_min if spray_speed_ft_min > 0 else 0

    ferry_speed_ft_min = config.ferry_speed_mph * 5280 / 60
    hop_time_min = hop_distance_ft / ferry_speed_ft_min if ferry_speed_ft_min > 0 else 0

    turn_time_min = num_turns * config.turn_time_sec / 60

    total_time_min = spray_time_min + hop_time_min + turn_time_min

    acres_per_hour = acres / (total_time_min / 60) if total_time_min > 0 else 0

    return {
        'acres': acres,
        'spray_distance_ft': spray_distance_ft,
        'hop_distance_ft': hop_distance_ft,
        'num_turns': num_turns,
        'num_lines': len(lines),
        'spray_time_min': spray_time_min,
        'hop_time_min': hop_time_min,
        'turn_time_min': turn_time_min,
        'total_time_min': total_time_min,
        'acres_per_hour': acres_per_hour,
    }


# ============================================================================
# Angle Optimization
# ============================================================================

def _get_field_orientation_angle(poly: 'Polygon') -> float:
    """
    Get the angle of the field's long axis using minimum rotated rectangle.

    Returns angle in degrees (0-90 range).
    """
    try:
        min_rect = poly.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)

        # Calculate edge lengths
        def edge_length_sq(p1, p2):
            return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2

        edge1_len = edge_length_sq(coords[0], coords[1])
        edge2_len = edge_length_sq(coords[1], coords[2])

        # Get the longer edge
        if edge1_len > edge2_len:
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
        else:
            dx = coords[2][0] - coords[1][0]
            dy = coords[2][1] - coords[1][1]

        # Calculate angle (0 = E-W, 90 = N-S)
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize to 0-90 range
        angle = abs(angle)
        if angle > 90:
            angle = 180 - angle

        return angle
    except Exception:
        return 45.0  # Default to diagonal if calculation fails


def optimize_angle(
    polygon: Union['Polygon', dict, list],
    config: Optional[SprayConfig] = None,
    obstacles: Optional[List] = None
) -> Tuple[float, Dict[str, float], List[List['LineString']]]:
    """
    Find optimal spray angle that maximizes acres per hour.

    Uses a fast two-phase approach:
    1. Estimate optimal angle from field orientation (minimum rotated rectangle)
    2. Search around that angle ±20° to refine

    This reduces evaluations from 19 (full search) to ~9 while maintaining accuracy.

    Args:
        polygon: Field boundary
        config: Spray configuration (uses defaults if None)
        obstacles: Optional list of obstacle polygons

    Returns:
        Tuple of (optimal_angle_deg, efficiency_metrics, best_tracks)
    """
    if config is None:
        config = SprayConfig()

    poly = _ensure_polygon(polygon)
    obs_list = _ensure_obstacles(obstacles)

    # Two-phase search for speed: coarse (every 15°) then refine around best
    # Phase 1: Coarse search - 13 angles covering full 0-180° range
    # (0° and 180° are same orientation, so we check 0-175°)
    coarse_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0]
    coarse_results = {}

    for angle in coarse_angles:
        tracks = generate_parallel_lines(poly, angle, config.swath_width_ft, config.headland_ft)
        if tracks:
            metrics = calculate_efficiency_from_tracks(tracks, poly, config)
            coarse_results[angle] = (metrics['acres_per_hour'], tracks, metrics)

    if not coarse_results:
        return 0.0, {}, []

    # Find top 2 coarse angles (to handle multiple local peaks)
    sorted_coarse = sorted(coarse_results.keys(), key=lambda a: coarse_results[a][0], reverse=True)
    best_coarse_angle = sorted_coarse[0]
    second_coarse_angle = sorted_coarse[1] if len(sorted_coarse) > 1 else best_coarse_angle

    # Phase 2: Refine around top 2 coarse angles (±10° in steps of 5)
    angles_to_try = set()
    for base_angle in [best_coarse_angle, second_coarse_angle]:
        for delta in [-10, -5, 5, 10]:
            refined = (base_angle + delta) % 180  # Wrap around 0-180 range
            if refined not in coarse_angles:
                angles_to_try.add(refined)

    # Initialize best from coarse results
    best_angle = best_coarse_angle
    best_efficiency, best_tracks, best_metrics = coarse_results[best_coarse_angle]

    # Check refined angles (only the new ones, not already in coarse)
    for angle in sorted(angles_to_try):
        tracks = generate_parallel_lines(poly, angle, config.swath_width_ft, config.headland_ft)
        if tracks:
            metrics = calculate_efficiency_from_tracks(tracks, poly, config)
            if metrics['acres_per_hour'] > best_efficiency:
                best_efficiency = metrics['acres_per_hour']
                best_angle = angle
                best_metrics = metrics
                best_tracks = tracks

    return best_angle, best_metrics, best_tracks


# ============================================================================
# Field Grouping and Multi-Field Optimization
# ============================================================================

def _polygon_distance_ft(poly1: 'Polygon', poly2: 'Polygon') -> float:
    """Calculate minimum distance in feet between two polygons."""
    # Get centroids for latitude reference
    c1 = poly1.centroid
    c2 = poly2.centroid
    avg_lat = (c1.y + c2.y) / 2

    # Get minimum distance in degrees
    dist_deg = poly1.distance(poly2)

    # Convert to feet (approximate using latitude)
    # Use average of lat/lon conversion since we don't know the direction
    ft_per_deg = FEET_PER_DEGREE_LAT * math.cos(math.radians(avg_lat))
    return dist_deg * ft_per_deg


def group_nearby_polygons(
    polygons: List['Polygon'],
    group_distance_ft: float = 1300.0
) -> List[List[int]]:
    """
    Group polygons that are within group_distance_ft of each other.

    Uses a grid-based spatial index for O(n) average case instead of O(n²).
    If polygon A is within distance of B, and B is within distance of C,
    then A, B, C are all in the same group.

    Args:
        polygons: List of Shapely Polygon objects
        group_distance_ft: Maximum distance to consider polygons as neighbors

    Returns:
        List of groups, where each group is a list of polygon indices
    """
    n = len(polygons)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Get centroids and average latitude for distance calculations
    centroids = [poly.centroid for poly in polygons]
    avg_lat = sum(c.y for c in centroids) / n

    # Convert group_distance to degrees for grid cell size
    # Use a slightly larger cell to ensure we don't miss neighbors
    cell_size_deg = (group_distance_ft / FEET_PER_DEGREE_LAT) * 1.5

    # Build spatial grid index
    grid = {}  # (grid_x, grid_y) -> list of polygon indices

    for i, centroid in enumerate(centroids):
        gx = int(centroid.x / cell_size_deg)
        gy = int(centroid.y / cell_size_deg)
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    # Build adjacency using grid - only check same and neighboring cells
    adjacent = [set() for _ in range(n)]

    for (gx, gy), indices in grid.items():
        # Get all polygons in this cell and 8 neighboring cells
        neighbor_indices = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    neighbor_indices.extend(grid[key])

        # Check distances only between polygons in nearby cells
        for i in indices:
            for j in neighbor_indices:
                if i >= j:  # Avoid duplicate checks
                    continue
                dist = _polygon_distance_ft(polygons[i], polygons[j])
                if dist <= group_distance_ft:
                    adjacent[i].add(j)
                    adjacent[j].add(i)

    # Find connected components using BFS
    visited = [False] * n
    groups = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS to find all connected polygons
        group = []
        queue = [start]
        visited[start] = True

        while queue:
            current = queue.pop(0)
            group.append(current)

            for neighbor in adjacent[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        groups.append(group)

    return groups


def optimize_field_group(
    polygons: List['Polygon'],
    config: Optional[SprayConfig] = None,
    obstacles: Optional[List] = None
) -> Tuple[List[float], List[Dict], float]:
    """
    Optimize spray angles for a group of nearby fields.

    Compares two strategies:
    1. Individual optimization: Each field gets its own optimal angle,
       but pay a penalty (angle_change_penalty_min) for each angle change
    2. Common angle: All fields use the same angle, no penalties

    Returns whichever strategy has lower total time.

    Args:
        polygons: List of Shapely Polygon objects in the group
        config: Spray configuration
        obstacles: Optional list of obstacle polygons

    Returns:
        Tuple of:
        - angles: List of optimal angles for each polygon
        - metrics: List of efficiency metrics for each polygon
        - strategy: 'individual' or 'common'
    """
    if config is None:
        config = SprayConfig()

    n = len(polygons)
    if n == 0:
        return [], [], 'none'

    # For very large groups (>20 fields), the angle change penalty becomes
    # negligible compared to total spray time. Skip common angle search.
    if n > 20:
        obs_list = _ensure_obstacles(obstacles)
        angles = []
        metrics_list = []
        for poly in polygons:
            angle, metrics, _ = optimize_angle(poly, config, obs_list)
            if not metrics or 'total_time_min' not in metrics:
                metrics = {'total_time_min': 0, 'acres_per_hour': 0, 'acres': 0,
                           'num_tracks': 0, 'num_turns': 0, 'num_hops': 0}
            angles.append(angle)
            metrics_list.append(metrics)
        return angles, metrics_list, 'individual'

    obs_list = _ensure_obstacles(obstacles)

    # Strategy 1: Individual optimization
    individual_angles = []
    individual_metrics = []
    individual_total_time = 0.0

    for poly in polygons:
        angle, metrics, _ = optimize_angle(poly, config, obs_list)
        individual_angles.append(angle)
        # Handle case where no tracks could be generated (tiny/invalid polygon)
        if not metrics or 'total_time_min' not in metrics:
            metrics = {'total_time_min': 0, 'acres_per_hour': 0, 'acres': 0,
                       'num_tracks': 0, 'num_turns': 0, 'num_hops': 0}
        individual_metrics.append(metrics)
        individual_total_time += metrics['total_time_min']

    # Count angle changes and add penalties
    angle_changes = 0
    for i in range(1, n):
        if abs(individual_angles[i] - individual_angles[i-1]) > config.angle_step_deg:
            angle_changes += 1

    individual_total_time += angle_changes * config.angle_change_penalty_min

    # Strategy 2: Find best common angle
    # Use coarser step (15°) for common angle search to speed up
    best_common_angle = 0.0
    best_common_time = float('inf')
    best_common_metrics = []

    common_angle_step = 15.0  # Coarser step for common angle search
    angle = 0.0
    while angle < 180.0:
        total_time = 0.0
        metrics_at_angle = []

        for poly in polygons:
            tracks = generate_parallel_lines(poly, angle, config.swath_width_ft, config.headland_ft)
            if tracks:
                metrics = calculate_efficiency_from_tracks(tracks, poly, config)
                total_time += metrics['total_time_min']
                metrics_at_angle.append(metrics)
            else:
                # No tracks at this angle, use a large time penalty
                total_time += 999
                metrics_at_angle.append({'total_time_min': 999, 'acres_per_hour': 0})

        if total_time < best_common_time:
            best_common_time = total_time
            best_common_angle = angle
            best_common_metrics = metrics_at_angle

        angle += common_angle_step

    # Compare strategies
    if best_common_time <= individual_total_time:
        # Common angle is better (or equal)
        return (
            [best_common_angle] * n,
            best_common_metrics,
            'common'
        )
    else:
        # Individual angles are better
        return (
            individual_angles,
            individual_metrics,
            'individual'
        )


def _optimize_single_polygon_worker(args):
    """Worker function for parallel polygon optimization."""
    poly_wkt, config_dict = args
    from shapely import wkt

    poly = wkt.loads(poly_wkt)
    config = SprayConfig(**config_dict)

    angle, metrics, _ = optimize_angle(poly, config)
    if not metrics or 'total_time_min' not in metrics:
        metrics = {'total_time_min': 0, 'acres_per_hour': 0, 'acres': 0,
                   'num_tracks': 0, 'num_turns': 0, 'num_hops': 0}

    return angle, metrics


def optimize_multi_field(
    polygons: List[Union['Polygon', dict, list]],
    config: Optional[SprayConfig] = None,
    obstacles: Optional[List] = None
) -> List[Dict[str, Any]]:
    """
    Optimize spray patterns for multiple fields, grouping nearby fields.

    Fields within group_distance_ft of each other are considered together.
    For each group, determines whether to use individual optimal angles
    (with angle change penalties) or a common angle for the whole group.

    Uses multiprocessing for parallel optimization across CPU cores.

    Args:
        polygons: List of field boundaries
        config: Spray configuration
        obstacles: Optional list of obstacle polygons

    Returns:
        List of result dicts, one per field, each containing:
        - 'angle': Selected spray angle
        - 'metrics': Efficiency metrics
        - 'group_id': Which group this field belongs to
        - 'strategy': 'individual' or 'common' for the group
    """
    if config is None:
        config = SprayConfig()

    # Convert all polygons
    polys = [_ensure_polygon(p) for p in polygons]

    # Group nearby polygons
    groups = group_nearby_polygons(polys, config.group_distance_ft)

    # Determine number of workers
    max_workers = config.max_workers or os.cpu_count() or 4

    # For parallel processing, we'll optimize individual polygons in parallel
    # then apply grouping logic after

    # First, get all polygons that need individual optimization (groups > 20 or need to compare)
    results = [None] * len(polys)

    # Convert config to dict for pickling
    config_dict = {
        'swath_width_ft': config.swath_width_ft,
        'spray_speed_mph': config.spray_speed_mph,
        'ferry_speed_mph': config.ferry_speed_mph,
        'turn_time_sec': config.turn_time_sec,
        'hop_distance_ft': config.hop_distance_ft,
        'hop_enabled': config.hop_enabled,
        'headland_ft': config.headland_ft,
        'angle_step_deg': config.angle_step_deg,
    }

    # Prepare work items: (poly_wkt, config_dict, original_index)
    work_items = []
    for i, poly in enumerate(polys):
        work_items.append((poly.wkt, config_dict))

    # Run parallel optimization
    poly_results = [None] * len(polys)

    if len(polys) > 10 and max_workers > 1:
        # Use multiprocessing for large datasets
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_optimize_single_polygon_worker, item): i
                       for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    angle, metrics = future.result()
                    poly_results[idx] = (angle, metrics)
                except Exception as e:
                    # Fallback for failed polygons
                    poly_results[idx] = (0.0, {'total_time_min': 0, 'acres_per_hour': 0})
    else:
        # Sequential for small datasets
        for i, item in enumerate(work_items):
            angle, metrics = _optimize_single_polygon_worker(item)
            poly_results[i] = (angle, metrics)

    # Now apply grouping logic to determine common vs individual angles
    for group_id, group_indices in enumerate(groups):
        n = len(group_indices)

        # Get individual results for this group
        group_angles = [poly_results[i][0] for i in group_indices]
        group_metrics = [poly_results[i][1] for i in group_indices]

        if n <= 20:
            # For small groups, check if common angle is better
            individual_total_time = sum(m['total_time_min'] for m in group_metrics)

            # Count angle changes
            angle_changes = sum(1 for i in range(1, n)
                               if abs(group_angles[i] - group_angles[i-1]) > config.angle_step_deg)
            individual_total_time += angle_changes * config.angle_change_penalty_min

            # Find best common angle (check the most common individual angles)
            from collections import Counter
            angle_counts = Counter(int(a / 15) * 15 for a in group_angles)  # Bucket by 15°
            common_candidates = [a for a, _ in angle_counts.most_common(3)]
            common_candidates.extend([0, 45, 90, 135])  # Always try cardinal and diagonal angles

            best_common_time = float('inf')
            best_common_angle = 0
            best_common_metrics = group_metrics

            for test_angle in set(common_candidates):
                total_time = 0
                test_metrics = []
                for i in group_indices:
                    # Generate at common angle
                    tracks = generate_parallel_lines(polys[i], test_angle,
                                                     config.swath_width_ft, config.headland_ft)
                    if tracks:
                        m = calculate_efficiency_from_tracks(tracks, polys[i], config)
                        total_time += m['total_time_min']
                        test_metrics.append(m)
                    else:
                        total_time += 999
                        test_metrics.append({'total_time_min': 999, 'acres_per_hour': 0})

                if total_time < best_common_time:
                    best_common_time = total_time
                    best_common_angle = test_angle
                    best_common_metrics = test_metrics

            # Choose better strategy
            if best_common_time <= individual_total_time:
                strategy = 'common'
                final_angles = [best_common_angle] * n
                final_metrics = best_common_metrics
            else:
                strategy = 'individual'
                final_angles = group_angles
                final_metrics = group_metrics
        else:
            # Large groups: use individual angles
            strategy = 'individual'
            final_angles = group_angles
            final_metrics = group_metrics

        # Store results
        for idx, poly_idx in enumerate(group_indices):
            results[poly_idx] = {
                'angle': final_angles[idx],
                'metrics': final_metrics[idx],
                'group_id': group_id,
                'strategy': strategy,
                'group_size': n
            }

    return results


# ============================================================================
# Main Entry Points
# ============================================================================

def optimize_spray_pattern(
    polygon: Union['Polygon', dict, list],
    config: Optional[SprayConfig] = None,
    obstacles: Optional[List] = None,
    angle: Optional[float] = None,
    return_metadata: bool = False,
    return_segments: bool = False
) -> Union[List['LineString'], Dict[str, Any]]:
    """
    Generate optimal spray pattern for a field.

    This is the main entry point for spray pattern optimization.

    Args:
        polygon: Field boundary (Shapely Polygon, GeoJSON dict, or coords)
        config: SprayConfig with operation parameters (uses defaults if None)
        obstacles: Optional list of no-spray zones (polygons)
        angle: Force specific angle (if None, optimizes automatically)
        return_metadata: Include efficiency statistics in output
        return_segments: Include on/off spray segment information

    Returns:
        If return_metadata=False and return_segments=False:
            List of LineString spray lines

        If return_metadata=True:
            Dict with:
            - 'lines': List of LineStrings
            - 'tracks': List of tracks (each track is list of segments)
            - 'angle': Optimal angle used
            - 'acres': Field area
            - 'acres_per_hour': Efficiency score
            - 'num_turns': Number of turns (between tracks)
            - 'num_hops': Number of hops (within tracks)
            - 'total_time_min': Total operation time
            - ... (other metrics)

        If return_segments=True:
            Dict with:
            - 'segments': List of dicts with 'geometry' and 'spray_on'
            - (plus metadata if return_metadata=True)

    Example:
        >>> from geofast import optimize_spray_pattern, SprayConfig
        >>>
        >>> # Basic usage
        >>> lines = optimize_spray_pattern(field_polygon)
        >>>
        >>> # With custom config
        >>> config = SprayConfig(swath_width_ft=60, hop_distance_ft=1500)
        >>> lines = optimize_spray_pattern(field_polygon, config=config)
        >>>
        >>> # Get efficiency stats
        >>> result = optimize_spray_pattern(field_polygon, return_metadata=True)
        >>> print(f"Efficiency: {result['acres_per_hour']:.1f} ac/hr")
        >>>
        >>> # Disable hops for ground operations
        >>> config = SprayConfig(hop_enabled=False)
        >>> lines = optimize_spray_pattern(field_polygon, config=config)
    """
    if config is None:
        config = SprayConfig()

    poly = _ensure_polygon(polygon)
    obs_list = _ensure_obstacles(obstacles)
    centroid_lat = poly.centroid.y

    # Determine angle to use
    if angle is not None:
        optimal_angle = angle
        tracks = generate_parallel_lines(poly, angle, config.swath_width_ft, config.headland_ft)
        metrics = calculate_efficiency_from_tracks(tracks, poly, config)
    else:
        # Find optimal angle
        optimal_angle, metrics, tracks = optimize_angle(poly, config, obs_list)

    # Flatten tracks to lines
    lines = flatten_tracks(tracks)

    # Handle obstacles if present
    segments = None
    if obs_list:
        segments = handle_obstacles(lines, obs_list, config, centroid_lat)

    # Build response based on requested format
    if return_segments:
        if segments is None:
            segments = [{'geometry': line, 'spray_on': True} for line in lines]

        result = {
            'segments': segments,
            'angle': optimal_angle,
        }
        if return_metadata:
            result.update(metrics)
        return result

    if return_metadata:
        return {
            'lines': lines,
            'tracks': tracks,
            'angle': optimal_angle,
            **metrics
        }

    return lines


def generate_spray_pattern_geojson(
    polygon: Union['Polygon', dict, list],
    config: Optional[SprayConfig] = None,
    obstacles: Optional[List] = None,
    angle: Optional[float] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Generate spray pattern and return as GeoJSON FeatureCollection.

    Args:
        polygon: Field boundary
        config: Spray configuration
        obstacles: Optional no-spray zones
        angle: Force specific angle (None for auto-optimization)
        include_metadata: Include efficiency stats in feature properties

    Returns:
        GeoJSON FeatureCollection with spray lines as features
    """
    result = optimize_spray_pattern(
        polygon, config, obstacles, angle,
        return_metadata=True, return_segments=True
    )

    features = []

    for i, seg in enumerate(result['segments']):
        geom = seg['geometry']
        coords = list(geom.coords)

        properties = {
            'index': i,
            'spray_on': seg.get('spray_on', True),
        }
        if 'hop' in seg:
            properties['hop'] = seg['hop']

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coords,
            },
            'properties': properties,
        }
        features.append(feature)

    # Build feature collection
    fc = {
        'type': 'FeatureCollection',
        'features': features,
    }

    if include_metadata:
        fc['properties'] = {
            'angle': result.get('angle', 0),
            'acres': result.get('acres', 0),
            'acres_per_hour': result.get('acres_per_hour', 0),
            'total_time_min': result.get('total_time_min', 0),
            'num_tracks': result.get('num_tracks', 0),
            'num_segments': result.get('num_segments', 0),
            'num_turns': result.get('num_turns', 0),
            'num_hops': result.get('num_hops', 0),
            'spray_distance_ft': result.get('spray_distance_ft', 0),
            'hop_distance_ft': result.get('hop_distance_ft', 0),
            'ferry_distance_ft': result.get('ferry_distance_ft', 0),
            'spray_time_min': result.get('spray_time_min', 0),
            'turn_time_min': result.get('turn_time_min', 0),
        }

    return fc
