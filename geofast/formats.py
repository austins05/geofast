"""
File format converters for geospatial data.
Supports: GeoJSON, KML, GPX, Shapefile, CSV, MPZ (MapPlus), and more.
"""

import os
import json
import struct
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np

from .parallel import parallel_map, parallel_io


# =============================================================================
# Format Detection
# =============================================================================

def detect_format(file_path: str) -> str:
    """
    Detect file format from extension or content.

    Returns:
        Format string: 'geojson', 'kml', 'gpx', 'shp', 'csv', 'mpz', 'unknown'
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    format_map = {
        '.geojson': 'geojson',
        '.json': 'geojson',
        '.kml': 'kml',
        '.kmz': 'kmz',
        '.gpx': 'gpx',
        '.shp': 'shp',
        '.csv': 'csv',
        '.tsv': 'csv',
        '.mpz': 'mpz',
    }

    if ext in format_map:
        return format_map[ext]

    # Try to detect from content
    if path.exists():
        with open(path, 'rb') as f:
            header = f.read(100)
            if b'FeatureCollection' in header or b'"type"' in header:
                return 'geojson'
            if b'<kml' in header.lower():
                return 'kml'
            if b'<gpx' in header.lower():
                return 'gpx'
            if b'SQLite' in header:
                return 'mpz'

    return 'unknown'


# =============================================================================
# GeoJSON Operations
# =============================================================================

def read_geojson(file_path: str) -> Dict[str, Any]:
    """Read GeoJSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def write_geojson(data: Dict[str, Any], file_path: str, indent: int = None):
    """Write GeoJSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def geojson_to_features(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract features from GeoJSON (handles FeatureCollection or single Feature)."""
    if data.get('type') == 'FeatureCollection':
        return data.get('features', [])
    elif data.get('type') == 'Feature':
        return [data]
    else:
        # Assume it's a geometry, wrap it
        return [{'type': 'Feature', 'geometry': data, 'properties': {}}]


def features_to_geojson(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap features in a FeatureCollection."""
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def filter_features(data: Dict[str, Any],
                    geometry_type: str = None,
                    property_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Filter GeoJSON features by geometry type and/or properties.

    Args:
        data: GeoJSON data
        geometry_type: Filter by geometry type ('Point', 'Polygon', etc.)
        property_filter: Dict of property key-value pairs to match

    Returns:
        List of matching features
    """
    features = geojson_to_features(data)
    result = []

    for feature in features:
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})

        if geometry_type and geom.get('type') != geometry_type:
            continue

        if property_filter:
            match = all(props.get(k) == v for k, v in property_filter.items())
            if not match:
                continue

        result.append(feature)

    return result


# =============================================================================
# KML Operations
# =============================================================================

KML_NS = {
    'kml': 'http://www.opengis.net/kml/2.2',
    'gx': 'http://www.google.com/kml/ext/2.2'
}


def read_kml(file_path: str) -> Dict[str, Any]:
    """
    Read KML file and convert to GeoJSON-like structure.

    Returns:
        Dict with 'features' list containing converted placemarks
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle namespace
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    features = []

    # Find all Placemarks
    for placemark in root.iter(f'{ns}Placemark'):
        feature = _kml_placemark_to_feature(placemark, ns)
        if feature:
            features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}


def _kml_placemark_to_feature(placemark, ns: str) -> Optional[Dict[str, Any]]:
    """Convert a KML Placemark to GeoJSON Feature."""
    properties = {}
    geometry = None

    # Extract name
    name_elem = placemark.find(f'{ns}name')
    if name_elem is not None and name_elem.text:
        properties['name'] = name_elem.text

    # Extract description
    desc_elem = placemark.find(f'{ns}description')
    if desc_elem is not None and desc_elem.text:
        properties['description'] = desc_elem.text

    # Extract extended data
    for data in placemark.findall(f'.//{ns}Data'):
        name = data.get('name')
        value_elem = data.find(f'{ns}value')
        if name and value_elem is not None:
            properties[name] = value_elem.text

    # Extract geometry
    point = placemark.find(f'{ns}Point')
    if point is not None:
        coords = point.find(f'{ns}coordinates')
        if coords is not None and coords.text:
            lon, lat, *alt = map(float, coords.text.strip().split(','))
            geometry = {'type': 'Point', 'coordinates': [lon, lat]}

    linestring = placemark.find(f'{ns}LineString')
    if linestring is not None:
        coords = linestring.find(f'{ns}coordinates')
        if coords is not None and coords.text:
            geometry = {'type': 'LineString', 'coordinates': _parse_kml_coords(coords.text)}

    polygon = placemark.find(f'{ns}Polygon')
    if polygon is not None:
        outer = polygon.find(f'.//{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates')
        if outer is not None and outer.text:
            outer_coords = _parse_kml_coords(outer.text)
            rings = [outer_coords]

            # Inner boundaries (holes)
            for inner in polygon.findall(f'.//{ns}innerBoundaryIs/{ns}LinearRing/{ns}coordinates'):
                if inner.text:
                    rings.append(_parse_kml_coords(inner.text))

            geometry = {'type': 'Polygon', 'coordinates': rings}

    if geometry:
        return {
            'type': 'Feature',
            'geometry': geometry,
            'properties': properties
        }

    return None


def _parse_kml_coords(text: str) -> List[List[float]]:
    """Parse KML coordinate string to list of [lon, lat] pairs."""
    coords = []
    for part in text.strip().split():
        if part:
            parts = part.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append([lon, lat])
    return coords


def write_kml(data: Dict[str, Any], file_path: str, name: str = 'GeoFast Export'):
    """
    Write GeoJSON-like data to KML file.

    Args:
        data: GeoJSON FeatureCollection or list of features
        file_path: Output file path
        name: Document name
    """
    features = geojson_to_features(data) if isinstance(data, dict) else data

    kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
    document = ET.SubElement(kml, 'Document')
    ET.SubElement(document, 'name').text = name

    for feature in features:
        placemark = ET.SubElement(document, 'Placemark')

        props = feature.get('properties', {})
        if 'name' in props:
            ET.SubElement(placemark, 'name').text = str(props['name'])
        if 'description' in props:
            ET.SubElement(placemark, 'description').text = str(props['description'])

        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])

        if geom_type == 'Point':
            point = ET.SubElement(placemark, 'Point')
            ET.SubElement(point, 'coordinates').text = f'{coords[0]},{coords[1]},0'

        elif geom_type == 'LineString':
            linestring = ET.SubElement(placemark, 'LineString')
            coord_str = ' '.join(f'{c[0]},{c[1]},0' for c in coords)
            ET.SubElement(linestring, 'coordinates').text = coord_str

        elif geom_type == 'Polygon':
            polygon = ET.SubElement(placemark, 'Polygon')
            outer = ET.SubElement(polygon, 'outerBoundaryIs')
            ring = ET.SubElement(outer, 'LinearRing')
            coord_str = ' '.join(f'{c[0]},{c[1]},0' for c in coords[0])
            ET.SubElement(ring, 'coordinates').text = coord_str

            # Inner rings (holes)
            for hole in coords[1:]:
                inner = ET.SubElement(polygon, 'innerBoundaryIs')
                ring = ET.SubElement(inner, 'LinearRing')
                coord_str = ' '.join(f'{c[0]},{c[1]},0' for c in hole)
                ET.SubElement(ring, 'coordinates').text = coord_str

    tree = ET.ElementTree(kml)
    ET.indent(tree, space='  ')
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


# =============================================================================
# GPX Operations
# =============================================================================

GPX_NS = {'gpx': 'http://www.topografix.com/GPX/1/1'}


def read_gpx(file_path: str) -> Dict[str, Any]:
    """
    Read GPX file and convert to GeoJSON-like structure.

    Returns:
        Dict with tracks, waypoints, and routes as features
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    features = []

    # Waypoints -> Points
    for wpt in root.findall(f'{ns}wpt'):
        lat = float(wpt.get('lat'))
        lon = float(wpt.get('lon'))

        props = {'type': 'waypoint'}
        name_elem = wpt.find(f'{ns}name')
        if name_elem is not None:
            props['name'] = name_elem.text

        ele_elem = wpt.find(f'{ns}ele')
        if ele_elem is not None:
            props['elevation'] = float(ele_elem.text)

        time_elem = wpt.find(f'{ns}time')
        if time_elem is not None:
            props['time'] = time_elem.text

        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': props
        })

    # Tracks -> LineStrings
    for trk in root.findall(f'{ns}trk'):
        props = {'type': 'track'}
        name_elem = trk.find(f'{ns}name')
        if name_elem is not None:
            props['name'] = name_elem.text

        for trkseg in trk.findall(f'{ns}trkseg'):
            coords = []
            times = []
            elevations = []

            for trkpt in trkseg.findall(f'{ns}trkpt'):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                coords.append([lon, lat])

                ele_elem = trkpt.find(f'{ns}ele')
                if ele_elem is not None:
                    elevations.append(float(ele_elem.text))

                time_elem = trkpt.find(f'{ns}time')
                if time_elem is not None:
                    times.append(time_elem.text)

            if coords:
                seg_props = props.copy()
                if times:
                    seg_props['times'] = times
                if elevations:
                    seg_props['elevations'] = elevations

                features.append({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': coords},
                    'properties': seg_props
                })

    # Routes -> LineStrings
    for rte in root.findall(f'{ns}rte'):
        props = {'type': 'route'}
        name_elem = rte.find(f'{ns}name')
        if name_elem is not None:
            props['name'] = name_elem.text

        coords = []
        for rtept in rte.findall(f'{ns}rtept'):
            lat = float(rtept.get('lat'))
            lon = float(rtept.get('lon'))
            coords.append([lon, lat])

        if coords:
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': coords},
                'properties': props
            })

    return {'type': 'FeatureCollection', 'features': features}


def write_gpx(data: Dict[str, Any], file_path: str, name: str = 'GeoFast Export'):
    """
    Write GeoJSON-like data to GPX file.
    Points become waypoints, LineStrings become tracks.
    """
    features = geojson_to_features(data) if isinstance(data, dict) else data

    gpx = ET.Element('gpx', {
        'version': '1.1',
        'creator': 'GeoFast',
        'xmlns': 'http://www.topografix.com/GPX/1/1'
    })

    for feature in features:
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])

        if geom_type == 'Point':
            wpt = ET.SubElement(gpx, 'wpt', lat=str(coords[1]), lon=str(coords[0]))
            if 'name' in props:
                ET.SubElement(wpt, 'name').text = str(props['name'])
            if 'elevation' in props:
                ET.SubElement(wpt, 'ele').text = str(props['elevation'])

        elif geom_type == 'LineString':
            trk = ET.SubElement(gpx, 'trk')
            if 'name' in props:
                ET.SubElement(trk, 'name').text = str(props['name'])

            trkseg = ET.SubElement(trk, 'trkseg')
            times = props.get('times', [])
            elevations = props.get('elevations', [])

            for i, coord in enumerate(coords):
                trkpt = ET.SubElement(trkseg, 'trkpt', lat=str(coord[1]), lon=str(coord[0]))
                if i < len(elevations):
                    ET.SubElement(trkpt, 'ele').text = str(elevations[i])
                if i < len(times):
                    ET.SubElement(trkpt, 'time').text = times[i]

    tree = ET.ElementTree(gpx)
    ET.indent(tree, space='  ')
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


# =============================================================================
# MPZ (MapPlus) Operations
# =============================================================================

class MPZReader:
    """
    Reader for MapPlus .mpz files (SQLite-based format).

    Example:
        >>> reader = MPZReader('data.mpz')
        >>> tracks = reader.get_tracks()
        >>> polygons = reader.get_polygons()
        >>> reader.close()

    Or as context manager:
        >>> with MPZReader('data.mpz') as reader:
        ...     tracks = reader.get_tracks()
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.conn = None
        self._open()

    def _open(self):
        """Open the SQLite database."""
        # MPZ files are SQLite databases, sometimes with .mpz extension
        # or extracted to data.sdb
        if self.file_path.endswith('.mpz'):
            # Try to find data.sdb inside
            import zipfile
            import tempfile

            if zipfile.is_zipfile(self.file_path):
                with zipfile.ZipFile(self.file_path, 'r') as zf:
                    # Extract data.sdb to temp
                    if 'data.sdb' in zf.namelist():
                        self._temp_dir = tempfile.mkdtemp()
                        zf.extract('data.sdb', self._temp_dir)
                        db_path = os.path.join(self._temp_dir, 'data.sdb')
                    else:
                        raise ValueError("No data.sdb found in MPZ file")
            else:
                # Might be a plain SQLite file with .mpz extension
                db_path = self.file_path
        else:
            db_path = self.file_path

        self.conn = sqlite3.connect(db_path)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

        # Clean up temp directory if we created one
        if hasattr(self, '_temp_dir'):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_items(self, item_type: int = None) -> List[Dict[str, Any]]:
        """
        Get items from the database.

        Args:
            item_type: Filter by type (1=folder, 10=track, etc.)

        Returns:
            List of item dictionaries
        """
        cursor = self.conn.cursor()

        if item_type is not None:
            cursor.execute("""
                SELECT id, parent_id, name, type, insert_date
                FROM t_item WHERE type = ?
            """, (item_type,))
        else:
            cursor.execute("""
                SELECT id, parent_id, name, type, insert_date
                FROM t_item
            """)

        items = []
        for row in cursor.fetchall():
            items.append({
                'id': row[0],
                'parent_id': row[1],
                'name': row[2],
                'type': row[3],
                'insert_date': row[4]
            })

        return items

    def get_folders(self) -> Dict[int, str]:
        """Get folder ID to name mapping."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name FROM t_item WHERE type = 1")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_tracks(self, include_coordinates: bool = True) -> List[Dict[str, Any]]:
        """
        Get all tracks from the MPZ file.

        Args:
            include_coordinates: Whether to decode coordinate data

        Returns:
            List of track dictionaries with coordinates
        """
        cursor = self.conn.cursor()
        folders = self.get_folders()

        # Find root folder
        cursor.execute("SELECT id FROM t_item WHERE type = 1 ORDER BY parent_id LIMIT 1")
        root_row = cursor.fetchone()
        root_id = root_row[0] if root_row else 0

        # Get pilot folders (direct children of root)
        cursor.execute("SELECT id, name FROM t_item WHERE parent_id = ? AND type = 1", (root_id,))
        pilots = {row[0]: row[1] for row in cursor.fetchall()}

        tracks = []

        if include_coordinates:
            cursor.execute("""
                SELECT i.id, i.parent_id, i.name, i.insert_date, s.coordinates
                FROM t_item i
                JOIN t_shape s ON i.id = s.item_id
                WHERE i.type = 10
            """)
        else:
            cursor.execute("""
                SELECT id, parent_id, name, insert_date
                FROM t_item WHERE type = 10
            """)

        for row in cursor.fetchall():
            if include_coordinates:
                item_id, parent_id, name, insert_date, blob = row
                coords = self._decode_track_coordinates(blob)
            else:
                item_id, parent_id, name, insert_date = row
                coords = None

            # Find pilot
            pilot_id = parent_id
            while pilot_id not in pilots and pilot_id != 0:
                cursor.execute("SELECT parent_id FROM t_item WHERE id = ?", (pilot_id,))
                result = cursor.fetchone()
                if result:
                    pilot_id = result[0]
                else:
                    break

            pilot_name = pilots.get(pilot_id, "Unknown")

            # Parse datetime
            dt = None
            if coords and coords[0].get('timestamp'):
                dt = datetime.fromtimestamp(coords[0]['timestamp'], tz=timezone.utc)
            elif insert_date:
                try:
                    dt = datetime.fromtimestamp(insert_date, tz=timezone.utc)
                except:
                    pass

            track = {
                'id': item_id,
                'name': name,
                'pilot': pilot_name,
                'datetime': dt,
                'date': dt.strftime('%Y-%m-%d') if dt else None,
            }

            if include_coordinates and coords:
                track['coordinates'] = coords
                track['point_count'] = len(coords)

            tracks.append(track)

        return tracks

    def _decode_track_coordinates(self, blob: bytes) -> List[Dict[str, Any]]:
        """Decode track coordinates from MapPlus binary format."""
        if not blob or len(blob) < 20:
            return []

        num_points = struct.unpack('<I', blob[4:8])[0]

        if num_points == 0 or num_points > 100000:
            return []

        latlon_offset = 12
        latlon_size = num_points * 16
        alt_offset = latlon_offset + latlon_size
        alt_size = num_points * 4
        ts_offset = alt_offset + alt_size

        coords = []
        for i in range(num_points):
            ll_off = latlon_offset + i * 16
            if ll_off + 16 > len(blob):
                break

            lat = struct.unpack('<d', blob[ll_off:ll_off+8])[0]
            lon = struct.unpack('<d', blob[ll_off+8:ll_off+16])[0]

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            point = {'lat': lat, 'lon': lon}

            # Altitude
            a_off = alt_offset + i * 4
            if a_off + 4 <= len(blob):
                point['altitude'] = struct.unpack('<f', blob[a_off:a_off+4])[0]

            # Timestamp
            t_off = ts_offset + i * 8
            if t_off + 8 <= len(blob):
                ts = struct.unpack('<d', blob[t_off:t_off+8])[0]
                if 1700000000 < ts < 1900000000:
                    point['timestamp'] = ts

            coords.append(point)

        return coords

    def get_polygons(self) -> List[Dict[str, Any]]:
        """Get all polygons from the MPZ file."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT i.id, i.name, s.coordinates
            FROM t_item i
            JOIN t_shape s ON i.id = s.item_id
            WHERE i.type = 3
        """)

        polygons = []
        for row in cursor.fetchall():
            item_id, name, blob = row
            coords = self._decode_polygon_coordinates(blob)

            if coords:
                polygons.append({
                    'id': item_id,
                    'name': name,
                    'coordinates': coords
                })

        return polygons

    def _decode_polygon_coordinates(self, blob: bytes) -> List[Tuple[float, float]]:
        """Decode polygon coordinates from MapPlus binary format."""
        if not blob or len(blob) < 12:
            return []

        # Similar structure to tracks but simpler
        num_points = struct.unpack('<I', blob[4:8])[0]

        if num_points == 0 or num_points > 100000:
            return []

        coords = []
        offset = 12

        for i in range(num_points):
            if offset + 16 > len(blob):
                break

            lat = struct.unpack('<d', blob[offset:offset+8])[0]
            lon = struct.unpack('<d', blob[offset+8:offset+16])[0]
            offset += 16

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append((lon, lat))

        return coords

    def to_geojson(self, include_tracks: bool = True,
                   include_polygons: bool = True) -> Dict[str, Any]:
        """
        Export MPZ contents to GeoJSON format.

        Args:
            include_tracks: Include track data
            include_polygons: Include polygon data

        Returns:
            GeoJSON FeatureCollection
        """
        features = []

        if include_tracks:
            tracks = self.get_tracks()
            for track in tracks:
                if track.get('coordinates'):
                    coords = [[p['lon'], p['lat']] for p in track['coordinates']]
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': coords
                        },
                        'properties': {
                            'id': track['id'],
                            'name': track['name'],
                            'pilot': track['pilot'],
                            'date': track['date'],
                            'point_count': track.get('point_count', len(coords))
                        }
                    })

        if include_polygons:
            polygons = self.get_polygons()
            for poly in polygons:
                if poly.get('coordinates'):
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [poly['coordinates']]
                        },
                        'properties': {
                            'id': poly['id'],
                            'name': poly['name']
                        }
                    })

        return features_to_geojson(features)


def read_mpz(file_path: str) -> Dict[str, Any]:
    """Read MPZ file and return GeoJSON."""
    with MPZReader(file_path) as reader:
        return reader.to_geojson()


# =============================================================================
# CSV Operations
# =============================================================================

def read_csv_points(file_path: str,
                    lat_col: str = 'lat',
                    lon_col: str = 'lon',
                    delimiter: str = ',') -> Dict[str, Any]:
    """
    Read CSV file with lat/lon columns as GeoJSON Points.

    Args:
        file_path: Path to CSV file
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        delimiter: CSV delimiter

    Returns:
        GeoJSON FeatureCollection
    """
    import csv

    features = []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            try:
                lat = float(row.get(lat_col, 0))
                lon = float(row.get(lon_col, 0))
            except (ValueError, TypeError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            # All columns become properties
            props = {k: v for k, v in row.items() if k not in (lat_col, lon_col)}

            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': props
            })

    return features_to_geojson(features)


def write_csv_points(data: Dict[str, Any],
                     file_path: str,
                     lat_col: str = 'lat',
                     lon_col: str = 'lon',
                     delimiter: str = ','):
    """Write GeoJSON Points to CSV."""
    import csv

    features = [f for f in geojson_to_features(data)
                if f.get('geometry', {}).get('type') == 'Point']

    if not features:
        return

    # Collect all property keys
    all_keys = set()
    for f in features:
        all_keys.update(f.get('properties', {}).keys())

    fieldnames = [lat_col, lon_col] + sorted(all_keys)

    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        for feature in features:
            coords = feature.get('geometry', {}).get('coordinates', [0, 0])
            row = {lat_col: coords[1], lon_col: coords[0]}
            row.update(feature.get('properties', {}))
            writer.writerow(row)


# =============================================================================
# Universal Converter
# =============================================================================

def convert(input_path: str, output_path: str, **kwargs) -> str:
    """
    Convert between geospatial formats.
    Auto-detects input format from extension/content.

    Args:
        input_path: Input file path
        output_path: Output file path
        **kwargs: Format-specific options

    Returns:
        Output file path

    Example:
        >>> convert('tracks.mpz', 'tracks.geojson')
        >>> convert('field.kml', 'field.geojson')
        >>> convert('points.csv', 'points.kml', lat_col='latitude', lon_col='longitude')
    """
    input_format = detect_format(input_path)
    output_format = detect_format(output_path)

    # Read input
    if input_format == 'geojson':
        data = read_geojson(input_path)
    elif input_format == 'kml':
        data = read_kml(input_path)
    elif input_format == 'gpx':
        data = read_gpx(input_path)
    elif input_format == 'mpz':
        data = read_mpz(input_path)
    elif input_format == 'csv':
        data = read_csv_points(input_path, **kwargs)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # Write output
    if output_format == 'geojson':
        write_geojson(data, output_path, indent=kwargs.get('indent', 2))
    elif output_format == 'kml':
        write_kml(data, output_path, name=kwargs.get('name', 'GeoFast Export'))
    elif output_format == 'gpx':
        write_gpx(data, output_path, name=kwargs.get('name', 'GeoFast Export'))
    elif output_format == 'csv':
        write_csv_points(data, output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return output_path


def convert_batch(input_files: List[str], output_dir: str,
                  output_format: str = 'geojson', **kwargs) -> List[str]:
    """
    Convert multiple files in parallel.

    Args:
        input_files: List of input file paths
        output_dir: Output directory
        output_format: Target format
        **kwargs: Format-specific options

    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    def convert_one(input_path):
        name = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{name}.{output_format}")
        return convert(input_path, output_path, **kwargs)

    return parallel_io(convert_one, input_files)
