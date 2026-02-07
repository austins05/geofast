#!/usr/bin/env python3
"""
Spray Pattern Optimizer for GeoFast.

Generates optimal spray lines for agricultural fields based on the
spray-line-generator algorithm. Supports multi-field optimization
and powerline detection.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import os

import numpy as np

try:
    from shapely.geometry import Polygon, LineString, MultiPolygon, Point, box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from .spray_line_generator import SprayLineGenerator, SprayConfig as GeneratorConfig, SprayResult


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
        powerlines_path: Path to transmission lines GeoJSON for detection
    """
    swath_width_ft: float = 50.0
    spray_speed_mph: float = 68.0
    ferry_speed_mph: float = 55.0
    turn_time_sec: float = 9.0
    hop_distance_ft: float = 1300.0
    hop_enabled: bool = True
    powerlines_path: Optional[str] = None


# ============================================================================
# Constants
# ============================================================================

FT_PER_DEG_LAT = 364567.2
POWERLINE_BUFFER_FT = 100
NS_PREFERENCE_THRESHOLD = 0.10  # E-W must be 10% better to beat N-S


# ============================================================================
# Powerline Detection
# ============================================================================

# Global powerline data (loaded once)
_POWERLINES: List[LineString] = []
_POWERLINE_INDEX: Optional[STRtree] = None


def load_powerlines(filepath: str) -> None:
    """Load transmission lines from GeoJSON and build spatial index."""
    global _POWERLINES, _POWERLINE_INDEX
    
    import json
    
    if not os.path.exists(filepath):
        return
    
    with open(filepath) as f:
        data = json.load(f)
    
    lines = []
    for feat in data.get('features', []):
        geom = feat.get('geometry', {})
        if geom.get('type') == 'LineString':
            coords = geom.get('coordinates', [])
            if len(coords) >= 2:
                lines.append(LineString(coords))
        elif geom.get('type') == 'MultiLineString':
            for line_coords in geom.get('coordinates', []):
                if len(line_coords) >= 2:
                    lines.append(LineString(line_coords))
    
    _POWERLINES = lines
    if lines:
        _POWERLINE_INDEX = STRtree(lines)


def find_intersecting_powerlines(polygon: Polygon, buffer_ft: float = 500) -> List[LineString]:
    """Find powerlines that intersect or are near a polygon."""
    if not _POWERLINES or _POWERLINE_INDEX is None:
        return []
    
    # Convert buffer to degrees
    center_lat = polygon.centroid.y
    ft_per_deg = FT_PER_DEG_LAT * math.cos(math.radians(center_lat))
    buffer_deg = buffer_ft / ft_per_deg
    
    # Query spatial index
    search_area = polygon.buffer(buffer_deg)
    candidates = _POWERLINE_INDEX.query(search_area)
    
    # Filter to actual intersections
    intersecting = []
    for idx in candidates:
        line = _POWERLINES[idx]
        if polygon.buffer(buffer_deg).intersects(line):
            intersecting.append(line)
    
    return intersecting


def check_powerlines(polygon: Polygon) -> Tuple[bool, List[Polygon]]:
    """
    Check if powerlines intersect a field.
    
    Returns:
        Tuple of (hasPowerlines, exclusion_zones)
    """
    powerlines = find_intersecting_powerlines(polygon)
    
    if not powerlines:
        return False, []
    
    # Create exclusion zones (buffers around powerlines)
    center_lat = polygon.centroid.y
    ft_per_deg = FT_PER_DEG_LAT * math.cos(math.radians(center_lat))
    buffer_deg = (POWERLINE_BUFFER_FT / 2) / ft_per_deg
    
    exclusion_zones = []
    for line in powerlines:
        buffered = line.buffer(buffer_deg)
        intersection = polygon.intersection(buffered)
        if not intersection.is_empty:
            if intersection.geom_type == 'Polygon':
                exclusion_zones.append(intersection)
            elif intersection.geom_type == 'MultiPolygon':
                exclusion_zones.extend(list(intersection.geoms))
    
    return True, exclusion_zones


# ============================================================================
# Helpers
# ============================================================================

def _ensure_polygon(geom) -> Polygon:
    """Convert geometry dict or Polygon to Shapely Polygon."""
    if isinstance(geom, Polygon):
        return geom
    
    if isinstance(geom, dict):
        coords = geom.get('coordinates', [])
        if coords:
            outer = coords[0]
            holes = coords[1:] if len(coords) > 1 else None
            return Polygon(outer, holes)
    
    raise ValueError(f"Cannot convert to Polygon: {type(geom)}")


def ft_per_deg_lon(lat: float) -> float:
    """Feet per degree longitude at given latitude."""
    return FT_PER_DEG_LAT * math.cos(math.radians(lat))


def count_effective_lines(lines: List[List[Tuple[float, float]]], bearing: float, 
                          swath_width_ft: float = 50) -> int:
    """
    Count effective spray lines by merging segments on the same track.
    """
    if not lines:
        return 0
    
    # Get center latitude for conversion
    all_lats = [coord[1] for line in lines for coord in line]
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 40.0
    
    ft_per_deg_x = ft_per_deg_lon(center_lat)
    ft_per_deg_y = FT_PER_DEG_LAT
    
    # Convert swath to degrees
    swath_deg = swath_width_ft / ft_per_deg_x
    
    # Group lines by track position
    tracks = {}  # track_id -> list of (start_pos, end_pos)
    
    for line in lines:
        if len(line) < 2:
            continue
        
        # Get line midpoint
        mid_x = (line[0][0] + line[-1][0]) / 2
        mid_y = (line[0][1] + line[-1][1]) / 2
        
        # Calculate track position based on bearing
        if abs(bearing) < 45 or abs(bearing - 180) < 45:
            # N-S lines: track by longitude
            track_pos = mid_x
            line_pos = mid_y
        else:
            # E-W lines: track by latitude
            track_pos = mid_y
            line_pos = mid_x
        
        # Round to grid
        track_id = round(track_pos / swath_deg)
        
        # Get line extent along track
        if abs(bearing) < 45 or abs(bearing - 180) < 45:
            start_pos = min(line[0][1], line[-1][1])
            end_pos = max(line[0][1], line[-1][1])
        else:
            start_pos = min(line[0][0], line[-1][0])
            end_pos = max(line[0][0], line[-1][0])
        
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append((start_pos, end_pos))
    
    # Count merged segments per track
    max_hop_deg = 1300 / ft_per_deg_y  # 1300ft hop distance
    total_segments = 0
    
    for track_id, segments in tracks.items():
        if not segments:
            continue
        
        # Sort by start position
        segments.sort(key=lambda x: x[0])
        
        # Merge segments within hop distance
        merged = 1
        current_end = segments[0][1]
        
        for start, end in segments[1:]:
            if start - current_end <= max_hop_deg:
                # Can hop - extend current segment
                current_end = max(current_end, end)
            else:
                # Gap too large - new segment
                merged += 1
                current_end = end
        
        total_segments += merged
    
    return total_segments


# ============================================================================
# Core Generation
# ============================================================================

def generate_lines_for_polygon(
    polygon: Polygon,
    config: SprayConfig,
    bearing: Optional[float] = None
) -> Tuple[List[List[Tuple[float, float]]], float, float]:
    """
    Generate spray lines for a single polygon.
    
    Returns:
        Tuple of (lines, total_distance_ft, area_acres)
    """
    # Create generator config
    gen_config = GeneratorConfig(swath_width_ft=config.swath_width_ft)
    generator = SprayLineGenerator(gen_config)
    
    # Get coordinates
    coords = [list(polygon.exterior.coords)]
    for hole in polygon.interiors:
        coords.append(list(hole.coords))
    
    # Generate lines
    result = generator.generate(coords, bearing_override=bearing)
    
    return result.lines, result.total_spray_distance_ft, result.field_area_acres


def calculate_optimal_bearing(polygon: Polygon, config: SprayConfig) -> Tuple[float, int, int]:
    """
    Calculate optimal bearing for a polygon.
    
    Returns:
        Tuple of (optimal_bearing, ns_lines, ew_lines)
    """
    gen_config = GeneratorConfig(swath_width_ft=config.swath_width_ft)
    generator = SprayLineGenerator(gen_config)
    
    coords = [list(polygon.exterior.coords)]
    
    # Try N-S (0°)
    result_ns = generator.generate(coords, bearing_override=0)
    ns_lines = count_effective_lines(result_ns.lines, 0, config.swath_width_ft)
    
    # Try E-W (90°)
    result_ew = generator.generate(coords, bearing_override=90)
    ew_lines = count_effective_lines(result_ew.lines, 90, config.swath_width_ft)
    
    # Pick direction (with N-S preference)
    if ns_lines == 0 and ew_lines == 0:
        return 0, 0, 0
    
    ew_advantage = (ns_lines - ew_lines) / max(ns_lines, 1)
    
    if ew_advantage > NS_PREFERENCE_THRESHOLD:
        return 90, ns_lines, ew_lines
    else:
        return 0, ns_lines, ew_lines


# ============================================================================
# Multi-Field Optimization
# ============================================================================

def optimize_multi_field(
    polygons: List[Polygon],
    config: SprayConfig,
    obstacles: Optional[List[Polygon]] = None,
    job_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Optimize spray angles across multiple fields.
    
    Groups nearby fields and determines optimal consistent direction.
    
    Returns:
        List of dicts with 'angle', 'group_id', 'strategy', 'group_size' for each polygon.
    """
    if not polygons:
        return []
    
    n = len(polygons)
    
    # Calculate optimal angle for each polygon individually
    individual_results = []
    for poly in polygons:
        bearing, ns_lines, ew_lines = calculate_optimal_bearing(poly, config)
        individual_results.append({
            'bearing': bearing,
            'ns_lines': ns_lines,
            'ew_lines': ew_lines,
            'polygon': poly
        })
    
    # Calculate global optimal direction
    total_ns = sum(r['ns_lines'] for r in individual_results)
    total_ew = sum(r['ew_lines'] for r in individual_results)
    
    if total_ns == 0 and total_ew == 0:
        global_bearing = 0
    else:
        ew_advantage = (total_ns - total_ew) / max(total_ns, 1)
        global_bearing = 90 if ew_advantage > NS_PREFERENCE_THRESHOLD else 0
    
    # Group nearby polygons
    groups = _group_polygons(polygons, config.hop_distance_ft)
    
    # For each group, determine if consistent direction is beneficial
    results = []
    for i, poly in enumerate(polygons):
        group_id = groups[i]
        group_polys = [j for j in range(n) if groups[j] == group_id]
        group_size = len(group_polys)
        
        if group_size == 1:
            # Isolated polygon - use individual optimal
            angle = individual_results[i]['bearing']
            strategy = 'individual'
        else:
            # Part of group - use global direction
            angle = global_bearing
            strategy = 'group'
        
        results.append({
            'angle': angle,
            'group_id': group_id,
            'strategy': strategy,
            'group_size': group_size
        })
    
    return results


def _group_polygons(polygons: List[Polygon], hop_distance_ft: float) -> List[int]:
    """
    Group nearby polygons using union-find.
    
    Returns list of group IDs (one per polygon).
    """
    n = len(polygons)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Convert hop distance to degrees
    hop_deg = hop_distance_ft / FT_PER_DEG_LAT
    
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            dist = polygons[i].distance(polygons[j])
            if dist <= hop_deg:
                union(i, j)
    
    # Normalize group IDs
    return [find(i) for i in range(n)]


# ============================================================================
# GeoJSON Output
# ============================================================================

def generate_spray_pattern_geojson(
    geometry: Dict[str, Any],
    config: Union[SprayConfig, Dict[str, Any], None] = None,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    angle: Optional[float] = None,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """
    Generate spray pattern as GeoJSON for a single polygon.
    
    Args:
        geometry: GeoJSON geometry dict
        config: SprayConfig or dict of config options
        obstacles: List of obstacle geometries (not yet implemented)
        angle: Force specific angle (None for auto)
        include_metadata: Include efficiency stats in properties
    
    Returns:
        GeoJSON FeatureCollection with spray lines
    """
    # Build config
    if config is None:
        spray_config = SprayConfig()
    elif isinstance(config, dict):
        spray_config = SprayConfig(**{k: v for k, v in config.items() 
                                       if k in SprayConfig.__dataclass_fields__})
    else:
        spray_config = config
    
    # Convert to Shapely polygon
    polygon = _ensure_polygon(geometry)
    
    # Check for powerlines
    has_powerlines, exclusion_zones = check_powerlines(polygon)
    
    # Determine angle
    if angle is None:
        angle, _, _ = calculate_optimal_bearing(polygon, spray_config)
    
    # Generate lines
    lines, total_distance_ft, area_acres = generate_lines_for_polygon(
        polygon, spray_config, bearing=angle
    )
    
    # Build GeoJSON features
    features = []
    for line in lines:
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line
            },
            'properties': {
                'type': 'spray_line'
            }
        })
    
    # Add exclusion zones
    for zone in exclusion_zones:
        if zone.geom_type == 'Polygon':
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [list(zone.exterior.coords)]
                },
                'properties': {
                    'type': 'powerline_exclusion',
                    'hasPowerlines': True
                }
            })
    
    # Calculate stats
    total_miles = total_distance_ft / 5280
    num_lines = count_effective_lines(lines, angle, spray_config.swath_width_ft)
    
    # Estimate time
    spray_time_hr = total_miles / spray_config.spray_speed_mph if spray_config.spray_speed_mph > 0 else 0
    turn_time_hr = (num_lines * spray_config.turn_time_sec) / 3600
    total_time_hr = spray_time_hr + turn_time_hr
    acres_per_hour = area_acres / total_time_hr if total_time_hr > 0 else 0
    
    result = {
        'type': 'FeatureCollection',
        'features': features,
        'properties': {
            'angle': angle,
            'acres': area_acres,
            'num_tracks': num_lines,
            'total_miles': total_miles,
            'hasPowerlines': has_powerlines
        }
    }
    
    if include_metadata:
        result['properties'].update({
            'acres_per_hour': acres_per_hour,
            'total_time_min': total_time_hr * 60,
            'num_turns': max(0, num_lines - 1),
            'spray_time_min': spray_time_hr * 60,
            'turn_time_min': turn_time_hr * 60
        })
    
    return result


# ============================================================================
# Legacy API Compatibility
# ============================================================================

def generate_parallel_lines(
    polygon: Polygon,
    angle_deg: float = 0,
    swath_width_ft: float = 50.0,
    centroid_lat: float = 40.0
) -> List[LineString]:
    """
    Generate parallel spray lines at given angle.
    
    Legacy API - wraps new SprayLineGenerator.
    """
    config = SprayConfig(swath_width_ft=swath_width_ft)
    lines, _, _ = generate_lines_for_polygon(polygon, config, bearing=angle_deg)
    return [LineString(line) for line in lines]


def calculate_efficiency(
    polygon: Polygon,
    angle_deg: float = 0,
    config: Optional[SprayConfig] = None
) -> Dict[str, float]:
    """
    Calculate spray efficiency at given angle.
    
    Legacy API.
    """
    if config is None:
        config = SprayConfig()
    
    lines, total_distance_ft, area_acres = generate_lines_for_polygon(
        polygon, config, bearing=angle_deg
    )
    
    num_lines = count_effective_lines(lines, angle_deg, config.swath_width_ft)
    total_miles = total_distance_ft / 5280
    
    spray_time_hr = total_miles / config.spray_speed_mph if config.spray_speed_mph > 0 else 0
    turn_time_hr = (num_lines * config.turn_time_sec) / 3600
    total_time_hr = spray_time_hr + turn_time_hr
    acres_per_hour = area_acres / total_time_hr if total_time_hr > 0 else 0
    
    return {
        'angle': angle_deg,
        'num_tracks': num_lines,
        'total_miles': total_miles,
        'area_acres': area_acres,
        'acres_per_hour': acres_per_hour,
        'total_time_min': total_time_hr * 60
    }


def optimize_angle(
    polygon: Polygon,
    config: Optional[SprayConfig] = None,
    angle_step: float = 5.0
) -> Dict[str, Any]:
    """
    Find optimal spray angle for a polygon.
    
    Legacy API - now just returns 0 or 90 (N-S or E-W).
    """
    if config is None:
        config = SprayConfig()
    
    angle, ns_lines, ew_lines = calculate_optimal_bearing(polygon, config)
    
    return {
        'optimal_angle': angle,
        'ns_lines': ns_lines,
        'ew_lines': ew_lines
    }


def optimize_spray_pattern(
    polygon: Polygon,
    config: SprayConfig,
    obstacles: Optional[List[Polygon]] = None
) -> Dict[str, Any]:
    """
    Optimize spray pattern for a single polygon.
    
    Returns dict with optimal angle and efficiency stats.
    """
    angle, ns_lines, ew_lines = calculate_optimal_bearing(polygon, config)
    
    lines, total_distance_ft, area_acres = generate_lines_for_polygon(
        polygon, config, bearing=angle
    )
    
    total_miles = total_distance_ft / 5280
    num_lines = count_effective_lines(lines, angle, config.swath_width_ft)
    
    # Calculate efficiency
    spray_time_hr = total_miles / config.spray_speed_mph if config.spray_speed_mph > 0 else 0
    turn_time_hr = (num_lines * config.turn_time_sec) / 3600
    total_time_hr = spray_time_hr + turn_time_hr
    acres_per_hour = area_acres / total_time_hr if total_time_hr > 0 else 0
    
    return {
        'angle': angle,
        'lines': lines,
        'num_tracks': num_lines,
        'total_miles': total_miles,
        'area_acres': area_acres,
        'acres_per_hour': acres_per_hour,
        'total_time_min': total_time_hr * 60
    }
