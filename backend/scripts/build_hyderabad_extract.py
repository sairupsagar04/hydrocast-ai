#!/usr/bin/env python3
# scripts/build_hyderabad_extract.py
"""
Build the offline Hyderabad OpenStreetMap extract used by
``backend/services/osm_service.py``.

WHY THIS EXISTS
    In production (Render), live Overpass requests time out or fail, so
    HydroCast fell back to a 3-street synthetic layout and the flood
    router reported ``route_source="no_path"``. This script precomputes
    the roads and open waterways for the Hyderabad area ONCE, offline,
    into compact JSON part files (``backend/data/hyderabad_osm.part*.json``
    plus a ``backend/data/hyderabad_osm.manifest.json`` index). At
    runtime ``osm_service`` serves any point inside the extract
    coverage from those files - no network call - and only points
    outside coverage use the live Overpass path.

REFRESH CADENCE
    The extract is a snapshot. Rebuild it monthly, or on demand when
    road coverage matters (new flyovers/ORR exits appear in OSM
    regularly). Rebuilding is: download a fresh source (below), run
    this script, commit the regenerated part files and manifest, push.
    The extract date is stored in the file metadata and surfaced by the
    API as ``data_extract_date`` so staleness is always visible.

DATA SOURCES (either works; the PBF route is what produced the
committed extract)
    1. Geofabrik PBF (recommended): download the Southern Zone extract
       (Geofabrik has no Telangana sub-extract; the whole-India file
       also works) and cut the Hyderabad bbox with pyosmium:

           https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf
           pip install osmium        # pyosmium, only needed for --source pbf
           python build_hyderabad_extract.py --source pbf \
               --input southern-zone-latest.osm.pbf \
               --output-dir ../data

    2. Chunked Overpass queries (no extra dependency, stdlib only):
       tiles the bbox and queries ``highway=*`` ways and
       ``waterway=drain/canal/stream/river/ditch`` ways per tile.

           python build_hyderabad_extract.py --source overpass \
               --output-dir ../data

OUTPUT FORMAT (compact, spatial-indexed, format 2, split into parts)
    The logical extract is one object:

    {
      "metadata": {dataset, extract_date, bbox [S, W, N, E], source,
                   source_attribution, license, coordinate_scale,
                   grid_cell_degrees, hw_values, counts...},
      "names":     [str, ...]                       # interned name table
      "nodes":     [lat_e5, lng_e5, d_lat, d_lng, ...]
                                                    # flat, delta-encoded
                                                    # shared node table
      "edges":     [[n_deltas, name_idx, hw_code, sf?, tun?, loc?], ...]
                                                    # roads, pre-split at
                                                    # junction nodes so
                                                    # endpoints are real
                                                    # intersections;
                                                    # trailing tag fields
                                                    # omitted when empty
      "waterways": [[n_deltas, name_idx, waterway_type], ...]
    }

    On disk it is split into part files of at most ~4.5 MB each (the
    GitHub transport used to commit this repo inlines file contents and
    rejects larger request bodies). Each part is a JSON object holding
    a subset of the top-level keys; list values are consecutive chunks
    of the full arrays, split at item boundaries (``nodes`` at lat/lng
    pair boundaries, since it is a flat delta chain). A tiny manifest
    (``hyderabad_osm.manifest.json``) records the format, extract date,
    bbox, source, attribution, license and the ordered part list; the
    runtime merges parts by concatenation in manifest order, which
    reproduces the logical extract exactly - every stored node/name
    index reference stays valid.

    The spatial grid is NOT stored: the runtime rebuilds it in memory
    at load from edge midpoints, using ``grid_cell_degrees``.

    Coordinates are integers scaled by 1e5 (~1.1 m resolution, the same
    precision the flood router joins endpoints at). Node coordinates
    and node-index sequences are delta-encoded, names are interned and
    highway classes are stored as small integer codes - together these
    keep the whole Hyderabad file under ~20 MB. The grid buckets each
    road edge by its midpoint into 0.02-degree cells (~2.2 km) so the
    runtime lookup only materialises nearby edges.

OpenStreetMap data is (c) OpenStreetMap contributors, available under
the Open Database License (ODbL). The attribution is stored in the
file metadata and must be kept if the data is republished.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Sequence, Tuple

# Hyderabad / ORR coverage: [south, west, north, east].
DEFAULT_BBOX = (17.15, 78.15, 17.65, 78.65)

COORD_SCALE = 100000  # 1e5 -> ~1.1 m
GRID_CELL_DEGREES = 0.02

# Same highway set osm_service ranks (its HIGHWAY_RANK keys) plus "road".
ROAD_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street",
    "service", "track", "road",
}
WATERWAY_TYPES = {"river", "stream", "canal", "drain", "ditch"}

OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
USER_AGENT = "HydroCastAI-extract-builder/1.0 (offline OSM extract)"


# ---------------------------------------------------------------------------
# Source 1: Geofabrik PBF via pyosmium
# ---------------------------------------------------------------------------

def read_ways_from_pbf(
    pbf_path: str, bbox: Tuple[float, float, float, float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (road_ways, waterway_ways) from a PBF file.

    Each way is {"nodes": [(lat, lng), ...], "tags": {...}}. Only ways
    with at least one node inside the bbox are kept.
    """
    import osmium  # pyosmium; imported lazily so --source overpass needs nothing

    south, west, north, east = bbox
    roads: List[Dict[str, Any]] = []
    waterways: List[Dict[str, Any]] = []

    class Handler(osmium.SimpleHandler):
        def way(self, w):  # noqa: N802 - pyosmium callback name
            tags = {t.k: t.v for t in w.tags}
            highway = tags.get("highway")
            waterway = tags.get("waterway")
            if highway not in ROAD_HIGHWAYS and waterway not in WATERWAY_TYPES:
                return
            coords = []
            inside = False
            for node in w.nodes:
                if not node.location.valid():
                    return  # unresolved node reference: skip the way
                lat = node.location.lat
                lng = node.location.lon
                coords.append((lat, lng))
                if south <= lat <= north and west <= lng <= east:
                    inside = True
            if not inside or len(coords) < 2:
                return
            entry = {"nodes": coords, "tags": tags}
            if highway in ROAD_HIGHWAYS:
                roads.append(entry)
            else:
                waterways.append(entry)

    handler = Handler()
    # locations=True makes node locations available on ways in one pass.
    handler.apply_file(pbf_path, locations=True)
    return roads, waterways


# ---------------------------------------------------------------------------
# Source 2: chunked Overpass queries (stdlib only)
# ---------------------------------------------------------------------------

def _overpass_query(query: str, timeout: int = 300) -> Dict[str, Any]:
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            request = urllib.request.Request(
                endpoint,
                data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - try the next endpoint
            last_error = error
            print(f"[!] Overpass endpoint {endpoint} failed ({error}); trying next.")
    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")


def read_ways_from_overpass(
    bbox: Tuple[float, float, float, float], tile_degrees: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tile the bbox and fetch road + waterway ways with geometry."""
    south, west, north, east = bbox
    roads_by_id: Dict[int, Dict[str, Any]] = {}
    waterways_by_id: Dict[int, Dict[str, Any]] = {}

    lat_steps = math.ceil((north - south) / tile_degrees)
    lng_steps = math.ceil((east - west) / tile_degrees)
    total_tiles = lat_steps * lng_steps
    tile_no = 0
    for i in range(lat_steps):
        for j in range(lng_steps):
            tile_no += 1
            t_south = south + i * tile_degrees
            t_north = min(north, t_south + tile_degrees)
            t_west = west + j * tile_degrees
            t_east = min(east, t_west + tile_degrees)
            tile_bbox = f"{t_south},{t_west},{t_north},{t_east}"
            query = (
                "[out:json][timeout:280];"
                "("
                f'way["highway"]({tile_bbox});'
                f'way["waterway"~"^(river|stream|canal|drain|ditch)$"]({tile_bbox});'
                ");out geom;"
            )
            print(f"[..] Tile {tile_no}/{total_tiles} ({tile_bbox})")
            payload = _overpass_query(query)
            for element in payload.get("elements", []):
                if element.get("type") != "way":
                    continue
                geometry = element.get("geometry") or []
                coords = [
                    (float(point["lat"]), float(point["lon"])) for point in geometry
                ]
                if len(coords) < 2:
                    continue
                tags = element.get("tags") or {}
                entry = {"nodes": coords, "tags": tags}
                if tags.get("highway") in ROAD_HIGHWAYS:
                    roads_by_id[int(element["id"])] = entry
                elif tags.get("waterway") in WATERWAY_TYPES:
                    waterways_by_id[int(element["id"])] = entry
            time.sleep(1)  # be polite to a shared public service
    return list(roads_by_id.values()), list(waterways_by_id.values())


# ---------------------------------------------------------------------------
# Shared: split road ways at junctions, index nodes, build the grid
# ---------------------------------------------------------------------------

def _scale(lat: float, lng: float) -> Tuple[int, int]:
    return (int(round(lat * COORD_SCALE)), int(round(lng * COORD_SCALE)))


def build_extract(
    road_ways: List[Dict[str, Any]],
    waterway_ways: List[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
    source_label: str,
) -> Dict[str, Any]:
    # Count how many road ways use each (scaled) node coordinate: nodes
    # used by 2+ ways are junctions, and ways are split there so edge
    # endpoints are real intersections (the flood router joins edges by
    # endpoint coordinates).
    usage: Counter = Counter()
    scaled_ways: List[List[Tuple[int, int]]] = []
    for way in road_ways:
        scaled = [_scale(lat, lng) for lat, lng in way["nodes"]]
        scaled_ways.append(scaled)
        for point in set(scaled):
            usage[point] += 1
    junctions = {point for point, count in usage.items() if count >= 2}

    node_index: Dict[Tuple[int, int], int] = {}
    nodes: List[List[int]] = []

    def index_of(point: Tuple[int, int]) -> int:
        idx = node_index.get(point)
        if idx is None:
            idx = len(nodes)
            node_index[point] = idx
            nodes.append([point[0], point[1]])
        return idx

    edges: List[Dict[str, Any]] = []
    for way, scaled in zip(road_ways, scaled_ways):
        tags = way["tags"]
        # Split points: way endpoints plus every junction in between.
        split_at = {0, len(scaled) - 1}
        for pos, point in enumerate(scaled):
            if point in junctions:
                split_at.add(pos)
        ordered_splits = sorted(split_at)
        for start_pos, end_pos in zip(ordered_splits, ordered_splits[1:]):
            segment = scaled[start_pos : end_pos + 1]
            if len(segment) < 2 or segment[0] == segment[-1] and len(set(segment)) == 1:
                continue
            edges.append(
                {
                    "n": [index_of(point) for point in segment],
                    "name": tags.get("name", ""),
                    "hw": tags.get("highway", "unclassified"),
                    "sf": tags.get("surface", ""),
                    "tun": tags.get("tunnel", ""),
                    "loc": tags.get("location", ""),
                }
            )

    waterways: List[Dict[str, Any]] = []
    for way in waterway_ways:
        scaled = [_scale(lat, lng) for lat, lng in way["nodes"]]
        if len(scaled) < 2:
            continue
        waterways.append(
            {
                "n": [index_of(point) for point in scaled],
                "name": way["tags"].get("name", ""),
                "ww": way["tags"].get("waterway", "stream"),
            }
        )

    # Note: no spatial grid is baked into the file. osm_service builds
    # the grid index in memory at load time from edge midpoints (see
    # GRID_CELL_DEGREES in the metadata) - baking it in cost ~3 MB of
    # JSON for something that takes a fraction of a second to rebuild.

    south, west, north, east = bbox
    metadata = {
        "dataset": "HydroCast Hyderabad offline OpenStreetMap extract",
        "format": 2,
        "extract_date": date.today().isoformat(),
        "bbox": [south, west, north, east],
        "source": source_label,
        "source_attribution": "(c) OpenStreetMap contributors",
        "license": "Open Database License (ODbL) 1.0",
        "coordinate_scale": COORD_SCALE,
        "grid_cell_degrees": GRID_CELL_DEGREES,
        "road_way_count": len(road_ways),
        "waterway_count": len(waterways),
        "edge_count": len(edges),
        "node_count": len(nodes),
    }
    return _encode_extract(metadata, nodes, edges, waterways)


def _delta_encode(sequence: Sequence[int]) -> List[int]:
    encoded = [sequence[0]]
    for previous, current in zip(sequence, sequence[1:]):
        encoded.append(current - previous)
    return encoded


def _encode_extract(
    metadata: Dict[str, Any],
    nodes: List[List[int]],
    edges: List[Dict[str, Any]],
    waterways: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Encode the extract into the compact format-2 on-disk shape."""
    names: List[str] = []
    name_index: Dict[str, int] = {}

    def intern(name: str) -> int:
        idx = name_index.get(name)
        if idx is None:
            idx = len(names)
            name_index[name] = idx
            names.append(name)
        return idx

    hw_values = sorted({edge["hw"] for edge in edges})
    hw_code = {value: idx for idx, value in enumerate(hw_values)}
    metadata["hw_values"] = hw_values

    flat_nodes: List[int] = []
    prev_lat = prev_lng = 0
    for lat, lng in nodes:
        flat_nodes.append(lat - prev_lat)
        flat_nodes.append(lng - prev_lng)
        prev_lat, prev_lng = lat, lng

    encoded_edges: List[List[Any]] = []
    for edge in edges:
        row: List[Any] = [
            _delta_encode(edge["n"]),
            intern(edge["name"]),
            hw_code[edge["hw"]],
        ]
        tail = [edge.get("sf", ""), edge.get("tun", ""), edge.get("loc", "")]
        while tail and tail[-1] == "":
            tail.pop()
        row.extend(tail)
        encoded_edges.append(row)

    encoded_waterways: List[List[Any]] = [
        [_delta_encode(way["n"]), intern(way["name"]), way["ww"]]
        for way in waterways
    ]

    return {
        "metadata": metadata,
        "names": names,
        "nodes": flat_nodes,
        "edges": encoded_edges,
        "waterways": encoded_waterways,
    }


# ---------------------------------------------------------------------------
# Output: split part files + manifest (see OUTPUT FORMAT in the docstring)
# ---------------------------------------------------------------------------

MAX_PART_BYTES = 4_500_000
PART_NAME_TEMPLATE = "hyderabad_osm.part{index}.json"
MANIFEST_NAME = "hyderabad_osm.manifest.json"
SECTION_ORDER = ("metadata", "names", "nodes", "edges", "waterways")


def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def split_extract_parts(
    extract: Dict[str, Any], max_part_bytes: int = MAX_PART_BYTES
) -> List[Dict[str, Any]]:
    """Split an encoded extract into part dicts of <= max_part_bytes each.

    Each part is a JSON object holding a subset of the extract's
    top-level keys; list values are consecutive chunks of the full
    arrays. Concatenating the chunks in part order reproduces the
    original arrays exactly.
    """
    parts: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    current_size = 2  # "{}"

    def flush() -> None:
        nonlocal current, current_size
        if current:
            parts.append(current)
        current = {}
        current_size = 2

    def key_cost(key: str, first_key: bool) -> int:
        # '"key":' plus the comma separating it from the previous key.
        return len(key) + 3 + (0 if first_key else 1)

    for key in SECTION_ORDER:
        value = extract[key]
        if isinstance(value, dict):
            blob_size = len(_dump(value).encode("utf-8"))
            cost = key_cost(key, not current) + blob_size
            if current and current_size + cost > max_part_bytes:
                flush()
                cost = key_cost(key, True) + blob_size
            current[key] = value
            current_size += cost
            continue
        items = list(value)
        item_sizes = [
            len(_dump(item).encode("utf-8")) for item in items
        ]
        index = 0
        while index < len(items):
            if key not in current:
                cost = key_cost(key, not current) + 2  # "[]"
                if current and current_size + cost > max_part_bytes:
                    flush()
                    cost = key_cost(key, True) + 2
                current[key] = []
                current_size += cost
            budget = max_part_bytes - current_size
            taken = 0
            used = 0
            while index + taken < len(items):
                extra = item_sizes[index + taken] + (
                    1 if (current[key] or taken) else 0
                )
                if used + extra > budget and (current[key] or taken):
                    break
                used += extra
                taken += 1
            if taken == 0:
                if current:
                    flush()
                    continue
                # A single item larger than a whole part: take it anyway
                # (cannot happen for this format - items are tiny).
                taken = 1
                used = item_sizes[index]
            if key == "nodes" and index + taken < len(items) and taken % 2 == 1:
                # Keep lat/lng pairs together across part boundaries.
                taken -= 1
                used = sum(item_sizes[index : index + taken]) + max(0, taken - 1)
            current[key].extend(items[index : index + taken])
            current_size += used
            index += taken
            if index < len(items):
                flush()
    flush()
    return parts


def write_extract_parts(
    extract: Dict[str, Any],
    output_dir: str,
    max_part_bytes: int = MAX_PART_BYTES,
) -> List[str]:
    """Write the extract as split part files + manifest into output_dir.

    Returns the ordered part file names. Stale part files from a
    previous build with more parts are removed so the directory always
    matches the manifest.
    """
    os.makedirs(output_dir, exist_ok=True)
    parts = split_extract_parts(extract, max_part_bytes)
    part_names: List[str] = []
    for i, part in enumerate(parts, start=1):
        name = PART_NAME_TEMPLATE.format(index=i)
        payload = _dump(part).encode("utf-8")
        if len(payload) > max_part_bytes:
            raise ValueError(
                f"Part {name} is {len(payload)} bytes > {max_part_bytes}"
            )
        with open(os.path.join(output_dir, name), "wb") as handle:
            handle.write(payload)
        part_names.append(name)
    # Remove stale parts (e.g. part5 from an older, larger build).
    stale_index = len(parts) + 1
    while os.path.exists(
        os.path.join(output_dir, PART_NAME_TEMPLATE.format(index=stale_index))
    ):
        os.remove(
            os.path.join(output_dir, PART_NAME_TEMPLATE.format(index=stale_index))
        )
        stale_index += 1
    metadata = extract["metadata"]
    manifest = {
        "dataset": metadata["dataset"],
        "format": metadata["format"],
        "extract_date": metadata["extract_date"],
        "bbox": metadata["bbox"],
        "source": metadata["source"],
        "source_attribution": metadata["source_attribution"],
        "license": metadata["license"],
        "parts": part_names,
    }
    with open(os.path.join(output_dir, MANIFEST_NAME), "wb") as handle:
        handle.write(_dump(manifest).encode("utf-8"))
    return part_names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["pbf", "overpass"], required=True)
    parser.add_argument("--input", help="PBF file path (for --source pbf)")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        ),
        help="Directory for the part files + manifest (default: ../data)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=list(DEFAULT_BBOX),
    )
    args = parser.parse_args(argv)

    bbox = tuple(args.bbox)
    if args.source == "pbf":
        if not args.input:
            parser.error("--input is required for --source pbf")
        print(f"[..] Reading ways from PBF: {args.input}")
        roads, waterways = read_ways_from_pbf(args.input, bbox)
        source_label = (
            "OpenStreetMap via Geofabrik extract "
            f"({os.path.basename(args.input)}), cut to the Hyderabad bbox"
        )
    else:
        print("[..] Fetching ways from Overpass in tiles")
        roads, waterways = read_ways_from_overpass(bbox)
        source_label = "OpenStreetMap via Overpass API, tiled over the Hyderabad bbox"

    print(f"[..] {len(roads)} road ways, {len(waterways)} waterways; building extract")
    extract = build_extract(roads, waterways, bbox, source_label)

    output_dir = os.path.abspath(args.output_dir)
    part_names = write_extract_parts(extract, output_dir)
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, name)) for name in part_names
    )
    print(
        f"[ok] Wrote {len(part_names)} part files + {MANIFEST_NAME} to "
        f"{output_dir} ({total_bytes / 1e6:.1f} MB total): "
        f"{extract['metadata']['edge_count']} edges, "
        f"{extract['metadata']['node_count']} nodes, "
        f"{extract['metadata']['waterway_count']} waterways"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
