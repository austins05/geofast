"""
Spray Line Generator for Aerial Crop Dusting

Generates optimal spray lines for agricultural fields based on patterns
learned from real pilot flight data.

Key parameters derived from analysis of 3,400+ real spray jobs:
- Swath width: ~50 ft (configurable)
- Spray direction: Along field's longest axis (82% of real flights)
- Preferred angles: N-S or E-W when field allows
- Efficiency target: ~0.17 miles spray per acre
"""

import json
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
from shapely.ops import unary_union
from shapely import affinity
import numpy as np


@dataclass
class SprayConfig:
    """Configuration for spray line generation"""
    swath_width_ft: float = 50.0  # Width of spray coverage per pass
    buffer_ft: float = 0.0  # Buffer inside field boundary (0 = spray to edge)
    prefer_cardinal: bool = True  # Prefer N-S or E-W when close
    cardinal_tolerance_deg: float = 15.0  # Snap to cardinal if within this
    min_line_length_ft: float = 50.0  # Skip lines shorter than this


@dataclass
class SprayResult:
    """Result of spray line generation"""
    lines: List[List[Tuple[float, float]]]  # List of spray lines (lon, lat coords)
    total_spray_distance_ft: float
    total_spray_distance_miles: float
    num_lines: int
    field_area_acres: float
    efficiency_miles_per_acre: float
    spray_bearing_deg: float
    estimated_swath_width_ft: float


class SprayLineGenerator:
    """Generates optimal spray lines for agricultural fields"""

    # Conversion constants
    FT_PER_DEG_LAT = 364567.2  # feet per degree latitude
    ACRES_PER_SQFT = 1 / 43560

    def __init__(self, config: Optional[SprayConfig] = None):
        self.config = config or SprayConfig()

    def ft_per_deg_lon(self, lat: float) -> float:
        """Feet per degree longitude at given latitude"""
        return self.FT_PER_DEG_LAT * math.cos(math.radians(lat))

    def polygon_from_coords(self, coords: List[List[Tuple[float, float]]]) -> Polygon:
        """Create Shapely polygon from GeoJSON coordinates"""
        outer_ring = coords[0]
        holes = coords[1:] if len(coords) > 1 else None
        return Polygon(outer_ring, holes)

    def get_field_center(self, polygon: Polygon) -> Tuple[float, float]:
        """Get centroid of polygon"""
        centroid = polygon.centroid
        return (centroid.x, centroid.y)

    def calculate_optimal_bearing(self, polygon: Polygon) -> float:
        """
        Calculate optimal spray bearing based on field shape.

        Uses minimum rotated rectangle to find the field's natural orientation.
        Sprays along the long axis (parallel to long dimension).

        Only snaps to cardinal if the field is VERY close (within 10°).
        """
        center_lat = polygon.centroid.y
        lon_scale = self.ft_per_deg_lon(center_lat)
        lat_scale = self.FT_PER_DEG_LAT

        # Get minimum rotated rectangle
        min_rect = polygon.minimum_rotated_rectangle
        rect_coords = list(min_rect.exterior.coords)

        # Calculate rotated rectangle side lengths
        side1_len = math.sqrt(
            ((rect_coords[1][0] - rect_coords[0][0]) * lon_scale) ** 2 +
            ((rect_coords[1][1] - rect_coords[0][1]) * lat_scale) ** 2
        )
        side2_len = math.sqrt(
            ((rect_coords[2][0] - rect_coords[1][0]) * lon_scale) ** 2 +
            ((rect_coords[2][1] - rect_coords[1][1]) * lat_scale) ** 2
        )

        # Find longer side and calculate its bearing
        if side1_len >= side2_len:
            dx = rect_coords[1][0] - rect_coords[0][0]
            dy = rect_coords[1][1] - rect_coords[0][1]
        else:
            dx = rect_coords[2][0] - rect_coords[1][0]
            dy = rect_coords[2][1] - rect_coords[1][1]

        # Bearing of the long SIDE of the bounding rectangle
        side_bearing = math.degrees(math.atan2(dx * lon_scale, dy * lat_scale))
        side_bearing = (side_bearing + 360) % 360
        if side_bearing > 180:
            side_bearing -= 180

        # Spray direction is PARALLEL to the long side (aligned with field edges)
        # Pilots fly along the field's length, not across it
        bearing = side_bearing

        # Snap to cardinal directions when reasonably close (within 25°)
        # Pilots tend to prefer clean N-S or E-W when the field allows
        snap_tolerance = 25
        for cardinal in [0, 90, 180]:
            if abs(bearing - cardinal) < snap_tolerance:
                bearing = cardinal
                break
            if cardinal == 0 and abs(bearing - 180) < snap_tolerance:
                bearing = 0
                break

        return bearing

    def generate_lines(self, polygon: Polygon, bearing_deg: float) -> List[LineString]:
        """
        Generate parallel spray lines across polygon at given bearing.
        """
        center_lon, center_lat = self.get_field_center(polygon)
        lon_scale = self.ft_per_deg_lon(center_lat)
        lat_scale = self.FT_PER_DEG_LAT

        # Convert swath width to degrees
        swath_deg = self.config.swath_width_ft / lon_scale  # approximate
        buffer_deg = self.config.buffer_ft / lon_scale

        # Buffer the polygon inward
        buffered = polygon.buffer(-buffer_deg)
        if buffered.is_empty:
            buffered = polygon

        # Rotate polygon so spray direction (bearing_deg) becomes horizontal
        # This means rotating by -(bearing_deg - 90) = 90 - bearing_deg
        rotated = affinity.rotate(buffered, 90 - bearing_deg, origin='centroid')

        # Get bounds of rotated polygon
        minx, miny, maxx, maxy = rotated.bounds

        # Generate horizontal lines across the rotated polygon
        lines = []
        y = miny + swath_deg / 2  # Start half swath from edge

        while y < maxy:
            # Create horizontal line across full width
            line = LineString([(minx - 0.01, y), (maxx + 0.01, y)])

            # Intersect with polygon
            intersection = line.intersection(rotated)

            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    lines.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    lines.extend(intersection.geoms)

            y += swath_deg

        # Rotate lines back to original orientation (reverse of forward rotation)
        result_lines = []
        for line in lines:
            rotated_back = affinity.rotate(line, bearing_deg - 90, origin=buffered.centroid)
            # Filter short lines
            line_len_ft = self.line_length_ft(list(rotated_back.coords), center_lat)
            if line_len_ft >= self.config.min_line_length_ft:
                result_lines.append(rotated_back)

        return result_lines

    def line_length_ft(self, coords: List[Tuple[float, float]], ref_lat: float) -> float:
        """Calculate line length in feet"""
        lon_scale = self.ft_per_deg_lon(ref_lat)
        lat_scale = self.FT_PER_DEG_LAT

        total = 0
        for i in range(len(coords) - 1):
            dx = (coords[i+1][0] - coords[i][0]) * lon_scale
            dy = (coords[i+1][1] - coords[i][1]) * lat_scale
            total += math.sqrt(dx*dx + dy*dy)
        return total

    def polygon_area_acres(self, polygon: Polygon, ref_lat: float) -> float:
        """Calculate polygon area in acres"""
        lon_scale = self.ft_per_deg_lon(ref_lat)
        lat_scale = self.FT_PER_DEG_LAT

        # Scale polygon to feet
        scaled = affinity.scale(polygon, xfact=lon_scale, yfact=lat_scale, origin=(0, 0))
        area_sqft = scaled.area
        return area_sqft * self.ACRES_PER_SQFT

    def generate(self, geojson_coords: List, bearing_override: Optional[float] = None) -> SprayResult:
        """
        Generate spray lines for a field.

        Args:
            geojson_coords: GeoJSON polygon coordinates [[outer_ring], [hole1], ...]
            bearing_override: Optional manual bearing in degrees (0=N, 90=E)

        Returns:
            SprayResult with spray lines and metrics
        """
        polygon = self.polygon_from_coords(geojson_coords)
        center_lat = polygon.centroid.y

        # Determine spray bearing
        if bearing_override is not None:
            bearing = bearing_override
        else:
            bearing = self.calculate_optimal_bearing(polygon)

        # Generate lines
        lines = self.generate_lines(polygon, bearing)

        # Calculate metrics
        total_distance_ft = sum(
            self.line_length_ft(list(line.coords), center_lat)
            for line in lines
        )
        total_distance_miles = total_distance_ft / 5280
        area_acres = self.polygon_area_acres(polygon, center_lat)

        efficiency = total_distance_miles / area_acres if area_acres > 0 else 0

        # Convert lines to coordinate lists
        line_coords = [list(line.coords) for line in lines]

        return SprayResult(
            lines=line_coords,
            total_spray_distance_ft=total_distance_ft,
            total_spray_distance_miles=total_distance_miles,
            num_lines=len(lines),
            field_area_acres=area_acres,
            efficiency_miles_per_acre=efficiency,
            spray_bearing_deg=bearing,
            estimated_swath_width_ft=self.config.swath_width_ft
        )

    def generate_geojson(self, geojson_coords: List, bearing_override: Optional[float] = None) -> dict:
        """
        Generate spray lines and return as GeoJSON FeatureCollection.
        """
        result = self.generate(geojson_coords, bearing_override)

        features = []

        # Add original field boundary
        features.append({
            "type": "Feature",
            "properties": {
                "type": "field_boundary",
                "area_acres": result.field_area_acres
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": geojson_coords
            }
        })

        # Add spray lines
        for i, line_coords in enumerate(result.lines):
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "spray_line",
                    "line_number": i + 1,
                    "bearing_deg": result.spray_bearing_deg
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords
                }
            })

        return {
            "type": "FeatureCollection",
            "properties": {
                "total_spray_distance_miles": result.total_spray_distance_miles,
                "num_lines": result.num_lines,
                "field_area_acres": result.field_area_acres,
                "efficiency_miles_per_acre": result.efficiency_miles_per_acre,
                "spray_bearing_deg": result.spray_bearing_deg,
                "swath_width_ft": result.estimated_swath_width_ft
            },
            "features": features
        }


def process_field_file(input_path: str, output_path: str, swath_width: float = 50.0):
    """
    Process a GeoJSON file with field boundaries and generate spray lines.
    """
    config = SprayConfig(swath_width_ft=swath_width)
    generator = SprayLineGenerator(config)

    with open(input_path) as f:
        data = json.load(f)

    results = []

    for feat in data['features']:
        if feat['geometry']['type'] != 'Polygon':
            continue

        coords = feat['geometry']['coordinates']
        props = feat['properties']

        try:
            result = generator.generate_geojson(coords)

            # Merge original properties
            result['properties']['jobId'] = props.get('jobId')
            result['properties']['name'] = props.get('name')
            result['properties']['address'] = props.get('address')
            result['properties']['original_area'] = props.get('area')

            results.append(result)

        except Exception as e:
            print(f"Error processing job {props.get('jobId')}: {e}")

    # Combine all results
    all_features = []
    for r in results:
        all_features.extend(r['features'])

    output = {
        "type": "FeatureCollection",
        "features": all_features
    }

    with open(output_path, 'w') as f:
        json.dump(output, f)

    print(f"Generated spray lines for {len(results)} fields")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python spray_line_generator.py <input.geojson> <output.geojson> [swath_width_ft]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    swath_width = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

    process_field_file(input_path, output_path, swath_width)
