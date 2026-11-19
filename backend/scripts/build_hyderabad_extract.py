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
    into compact JSON tile files (``backend/data/tiles/*.json`` plus a
    ``backend/data/hyderabad_osm.manifest.json`` index). At runtime
    ``osm_service`` serves any point inside the extract coverage from
    those files - no network call - and only points outside coverage
    use the live Overpass path.

WHY TILES (format 3)
    Format 2 stored the whole city as one logical extract split into
    ~4.5 MB part files, and the runtime had to parse and build compact
    arrays for the ENTIRE city on first use (~143 MB peak, ~85 MB
    retained) inside a ~512 MB free-tier instance that also runs the
    DEM/raster stages - the first prediction still OOM-killed the
    process. Format 3 therefore cuts the coverage into geographic
    tiles (default 0.05 degrees, a 10x10 grid over the bbox). Each
    tile file holds ONLY the road edges and waterways whose geometry
    intersects that tile (an edge crossing a boundary appears in each
    tile it touches; the runtime dedupes by global edge id). A query
    loads and parses only the tiles under its search area - a 2 km
    radius query touches at most 4 tiles geometrically - so the
    extract's memory contribution drops to a few MB per query.

REFRESH CADENCE
    The extract is a snapshot. Rebuild it monthly, or on demand when
    road coverage matters (new flyovers/ORR exits appear in OSM
    regularly). Rebuilding is: download a fresh source (below), run
    this script, commit the regenerated tile files and manifest, push.
    The extract date is stored in the manifest and surfaced by the
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

OUTPUT FORMAT (compact, geographic tiles, format 3)
    A tiny manifest (``hyderabad_osm.manifest.json``) records the
    format, extract date, bbox, source, attribution, license, the
    shared vocabularies (``hw_values``), the tile grid
    (``tile_degrees``, ``tile_rows``, ``tile_cols``, ``tile_dir``,
    ``tile_file_template``) and a ``tiles`` index mapping
    ``"<row>_<col>"`` to per-tile edge/waterway counts.

    Each tile file (``tiles/hyd_r{row:02d}_c{col:02d}.json``) is one
    JSON object in the format-2-style compact encoding, scoped to the
    tile:

    {
      "tile":      [row, col],
      "names":     [str, ...]        # interned name table (tile-local)
      "nodes":     [lat_e5, lng_e5, d_lat, d_lng, ...]
                                     # flat, delta-encoded coords of
                                     # the tile's nodes, ordered by
                                     # ascending global node id
      "node_ids":  [id0, d_id, ...]  # delta-encoded GLOBAL node ids,
                                     # parallel to "nodes"; lets the
                                     # runtime keep exact global node
                                     # identity when merging tiles
      "edges":     [[id_delta, n_deltas, name_idx, hw_code,
                     sf?, tun?, loc?], ...]
                                     # roads pre-split at junction
                                     # nodes; sorted by global edge
                                     # id, id stored delta-encoded;
                                     # n_deltas are TILE-LOCAL node
                                     # indices; trailing tag fields
                                     # omitted when empty
      "waterways": [[id_delta, n_deltas, name_idx, waterway_type],
                    ...]               # same id/sequence encoding
    }

    Coordinates are integers scaled by 1e5 (~1.1 m resolution, the
    same precision the flood router joins endpoints at). Node
    coordinates and node-index sequences are delta-encoded, names are
    interned per tile and highway classes are stored as small integer
    codes into the manifest's shared ``hw_values`` table.

    No spatial grid is baked into the tiles: the runtime rebuilds the
    midpoint grid in memory per query from just the loaded tiles,
    using ``grid_cell_degrees`` (0.02 degrees, ~2.2 km) from the
    manifest metadata.

    Individual tile files stay small (largest ~0.7 MB at the default
    0.05-degree grid), comfortably below the per-file size limit of
    the GitHub transport used to commit this repo, which inlines file
    contents and rejects larger request bodies.

OpenStreetMap data is (c) OpenStreetMap contributors, available under
the Open Database License (ODbL). The attribution is stored in the
manifest and must be kept if the data is republished.
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
DEFAULT_TILE_DEGREES = 0.05

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

MANIFEST_NAME = "hyderabad_osm.manifest.json"
TILE_DIR_NAME = "tiles"
TILE_NAME_TEMPLATE = "hyd_r{row:02d}_c{col:02d}.json"
# Legacy format-2 artefacts removed by write_extract_tiles().
LEGACY_SINGLE_NAME = "hyderabad_osm.json"


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
# Shared: split road ways at junctions, index nodes globally
# ---------------------------------------------------------------------------

def _scale(lat: float, lng: float) -> Tuple[int, int]:
    return (int(round(lat * COORD_SCALE)), int(round(lng * COORD_SCALE)))


def build_extract(
    road_ways: List[Dict[str, Any]],
    waterway_ways: List[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
    source_label: str,
    extract_date: str,
) -> Dict[str, Any]:
    """Build the decoded logical extract (global node/edge indexing).

    Returns {"metadata", "nodes", "edges", "waterways"} where nodes
    are [lat_e5, lng_e5] pairs, edges/waterways are dicts whose "n"
    lists hold GLOBAL node indices (an item's position in the list is
    its global id). ``build_tiles`` cuts this into format-3 tiles.
    """
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

    south, west, north, east = bbox
    metadata = {
        "dataset": "HydroCast Hyderabad offline OpenStreetMap extract",
        "format": 3,
        "extract_date": extract_date,
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
        "hw_values": sorted({edge["hw"] for edge in edges}),
    }
    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
        "waterways": waterways,
    }


# ---------------------------------------------------------------------------
# Format 3: geographic tiles
# ---------------------------------------------------------------------------

def _delta_encode(sequence: Sequence[int]) -> List[int]:
    encoded = [sequence[0]]
    for previous, current in zip(sequence, sequence[1:]):
        encoded.append(current - previous)
    return encoded


def _tiles_for_bbox(
    item_bbox: Tuple[float, float, float, float],
    origin_south: float,
    origin_west: float,
    tile_degrees: float,
    tile_rows: int,
    tile_cols: int,
) -> List[Tuple[int, int]]:
    """All tile (row, col) whose area intersects the item's bbox."""
    s, w, n, e = item_bbox
    r0 = max(0, int(math.floor((s - origin_south) / tile_degrees)))
    r1 = min(tile_rows - 1, int(math.floor((n - origin_south) / tile_degrees)))
    c0 = max(0, int(math.floor((w - origin_west) / tile_degrees)))
    c1 = min(tile_cols - 1, int(math.floor((e - origin_west) / tile_degrees)))
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def build_tiles(
    extract: Dict[str, Any], tile_degrees: float = DEFAULT_TILE_DEGREES
) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], Dict[str, Any]]]:
    """Cut a decoded extract into format-3 tiles + a manifest dict.

    Every edge/waterway is assigned to each tile its geometry bbox
    intersects. Within a tile, items are sorted by global id (stored
    delta-encoded), nodes are the union of the items' nodes ordered
    by global id (coords and global ids both delta-encoded), and
    names are interned tile-locally. Highway codes index the shared
    ``hw_values`` table kept in the manifest metadata.
    """
    metadata = extract["metadata"]
    nodes = extract["nodes"]
    edges = extract["edges"]
    waterways = extract["waterways"]
    south, west, north, east = (float(v) for v in metadata["bbox"])
    tile_rows = int(math.ceil((north - south) / tile_degrees - 1e-9))
    tile_cols = int(math.ceil((east - west) / tile_degrees - 1e-9))
    scale = float(metadata["coordinate_scale"])
    hw_code = {value: idx for idx, value in enumerate(metadata["hw_values"])}

    def item_bbox(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
        lats = [nodes[i][0] for i in item["n"]]
        lngs = [nodes[i][1] for i in item["n"]]
        return (
            min(lats) / scale,
            min(lngs) / scale,
            max(lats) / scale,
            max(lngs) / scale,
        )

    tile_edges: Dict[Tuple[int, int], List[int]] = {}
    tile_waterways: Dict[Tuple[int, int], List[int]] = {}
    for global_id, edge in enumerate(edges):
        for tile in _tiles_for_bbox(
            item_bbox(edge), south, west, tile_degrees, tile_rows, tile_cols
        ):
            tile_edges.setdefault(tile, []).append(global_id)
    for global_id, way in enumerate(waterways):
        for tile in _tiles_for_bbox(
            item_bbox(way), south, west, tile_degrees, tile_rows, tile_cols
        ):
            tile_waterways.setdefault(tile, []).append(global_id)

    tiles: Dict[Tuple[int, int], Dict[str, Any]] = {}
    tile_index: Dict[str, Dict[str, int]] = {}
    for tile in sorted(set(tile_edges) | set(tile_waterways)):
        edge_ids = sorted(tile_edges.get(tile, []))
        way_ids = sorted(tile_waterways.get(tile, []))

        node_set = set()
        for gid in edge_ids:
            node_set.update(edges[gid]["n"])
        for gid in way_ids:
            node_set.update(waterways[gid]["n"])
        ordered_nodes = sorted(node_set)
        local_index = {gid: pos for pos, gid in enumerate(ordered_nodes)}

        flat_nodes: List[int] = []
        prev_lat = prev_lng = 0
        for gid in ordered_nodes:
            lat_i, lng_i = nodes[gid]
            flat_nodes.append(lat_i - prev_lat)
            flat_nodes.append(lng_i - prev_lng)
            prev_lat, prev_lng = lat_i, lng_i

        names: List[str] = []
        name_index: Dict[str, int] = {}

        def intern(name: str) -> int:
            idx = name_index.get(name)
            if idx is None:
                idx = len(names)
                name_index[name] = idx
                names.append(name)
            return idx

        encoded_edges: List[List[Any]] = []
        prev_gid = 0
        for gid in edge_ids:
            edge = edges[gid]
            row: List[Any] = [
                gid - prev_gid,
                _delta_encode([local_index[i] for i in edge["n"]]),
                intern(edge["name"]),
                hw_code[edge["hw"]],
            ]
            prev_gid = gid
            tail = [edge.get("sf", ""), edge.get("tun", ""), edge.get("loc", "")]
            while tail and tail[-1] == "":
                tail.pop()
            row.extend(tail)
            encoded_edges.append(row)

        encoded_waterways: List[List[Any]] = []
        prev_gid = 0
        for gid in way_ids:
            way = waterways[gid]
            encoded_waterways.append(
                [
                    gid - prev_gid,
                    _delta_encode([local_index[i] for i in way["n"]]),
                    intern(way["name"]),
                    way["ww"],
                ]
            )
            prev_gid = gid

        tiles[tile] = {
            "tile": [tile[0], tile[1]],
            "names": names,
            "nodes": flat_nodes,
            "node_ids": _delta_encode(ordered_nodes),
            "edges": encoded_edges,
            "waterways": encoded_waterways,
        }
        tile_index[f"{tile[0]}_{tile[1]}"] = {
            "edges": len(edge_ids),
            "waterways": len(way_ids),
            "nodes": len(ordered_nodes),
        }

    manifest = {
        "dataset": metadata["dataset"],
        "format": 3,
        "extract_date": metadata["extract_date"],
        "bbox": metadata["bbox"],
        "source": metadata["source"],
        "source_attribution": metadata["source_attribution"],
        "license": metadata["license"],
        "coordinate_scale": metadata["coordinate_scale"],
        "grid_cell_degrees": metadata["grid_cell_degrees"],
        "hw_values": metadata["hw_values"],
        "road_way_count": metadata["road_way_count"],
        "waterway_count": metadata["waterway_count"],
        "edge_count": metadata["edge_count"],
        "node_count": metadata["node_count"],
        "tile_degrees": tile_degrees,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_dir": TILE_DIR_NAME,
        "tile_file_template": TILE_NAME_TEMPLATE,
        "tiles": tile_index,
    }
    return manifest, tiles


# ---------------------------------------------------------------------------
# Output: tile files + manifest (see OUTPUT FORMAT in the docstring)
# ---------------------------------------------------------------------------

def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def write_extract_tiles(
    extract: Dict[str, Any],
    output_dir: str,
    tile_degrees: float = DEFAULT_TILE_DEGREES,
) -> Tuple[Dict[str, Any], List[str]]:
    """Write format-3 tile files + manifest into output_dir.

    Returns (manifest, tile file names written). Stale tile files,
    legacy format-2 part files and a legacy single-file extract in
    the same directory are removed so the directory always matches
    the manifest.
    """
    manifest, tiles = build_tiles(extract, tile_degrees)
    os.makedirs(output_dir, exist_ok=True)
    tile_dir = os.path.join(output_dir, TILE_DIR_NAME)
    os.makedirs(tile_dir, exist_ok=True)

    written: List[str] = []
    for (row, col), tile in sorted(tiles.items()):
        name = TILE_NAME_TEMPLATE.format(row=row, col=col)
        payload = _dump(tile).encode("utf-8")
        with open(os.path.join(tile_dir, name), "wb") as handle:
            handle.write(payload)
        written.append(name)
        manifest["tiles"][f"{row}_{col}"]["bytes"] = len(payload)

    # Remove stale tile files from previous builds.
    keep = set(written)
    for name in os.listdir(tile_dir):
        if name.endswith(".json") and name not in keep:
            os.remove(os.path.join(tile_dir, name))
    # Remove legacy format-2 artefacts (part files, single file).
    for name in os.listdir(output_dir):
        if name.startswith("hyderabad_osm.part") and name.endswith(".json"):
            os.remove(os.path.join(output_dir, name))
    single = os.path.join(output_dir, LEGACY_SINGLE_NAME)
    if os.path.exists(single):
        os.remove(single)

    with open(os.path.join(output_dir, MANIFEST_NAME), "wb") as handle:
        handle.write(_dump(manifest).encode("utf-8"))
    return manifest, written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["pbf", "overpass"], required=True)
    parser.add_argument("--input", help="PBF file path (for --source pbf)")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        ),
        help="Directory for the tile files + manifest (default: ../data)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=list(DEFAULT_BBOX),
    )
    parser.add_argument(
        "--tile-degrees",
        type=float,
        default=DEFAULT_TILE_DEGREES,
        help="Geographic tile size in degrees (default: 0.05)",
    )
    parser.add_argument(
        "--extract-date",
        default=None,
        help="Snapshot date stored in the manifest (default: today). "
        "Pass the source file's date when re-cutting an older snapshot.",
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

    extract_date = args.extract_date or date.today().isoformat()
    print(f"[..] {len(roads)} road ways, {len(waterways)} waterways; building extract")
    extract = build_extract(roads, waterways, bbox, source_label, extract_date)

    output_dir = os.path.abspath(args.output_dir)
    manifest, tile_names = write_extract_tiles(
        extract, output_dir, tile_degrees=args.tile_degrees
    )
    tile_dir = os.path.join(output_dir, TILE_DIR_NAME)
    total_bytes = sum(
        os.path.getsize(os.path.join(tile_dir, name)) for name in tile_names
    )
    largest = max(
        os.path.getsize(os.path.join(tile_dir, name)) for name in tile_names
    )
    print(
        f"[ok] Wrote {len(tile_names)} tile files + {MANIFEST_NAME} to "
        f"{output_dir} ({total_bytes / 1e6:.1f} MB total, largest tile "
        f"{largest / 1e6:.2f} MB, grid {manifest['tile_rows']}x"
        f"{manifest['tile_cols']}): {manifest['edge_count']} edges, "
        f"{manifest['node_count']} nodes, "
        f"{manifest['waterway_count']} waterways, "
        f"extract date {manifest['extract_date']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
