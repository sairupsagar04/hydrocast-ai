# services/osm_service.py
"""
HydroCast AI - OpenStreetMap infrastructure service (real-data replacement).

This module replaces the previous Gemini-based implementation of
``fetch_roads_near_point()``. The old version asked a large language model
to *invent* road segments, stormwater conduits and manholes for a given
latitude/longitude, which meant the infrastructure shown to the user was
fabricated, non-deterministic and impossible to verify.

This version works differently:

ROADS - REAL DATA
    Road segments are fetched from OpenStreetMap via OSMnx (which queries
    the Overpass API). The same point therefore returns the same real
    roads, with their real names, geometry, highway class and surface
    tags, on every request. Results are cached in memory so repeated
    requests for the same area do not re-hit Overpass.

DRAINAGE - READ THIS BEFORE CLAIMING OTHERWISE IN A DEMO / VIVA
    There is no public API (government or otherwise) that returns real
    underground storm-sewer layouts - pipe diameters, manhole locations
    and capacities - for an arbitrary point in India. That is municipal
    asset data. So this module is explicit about where drainage comes
    from, and tags every item with a ``source`` field:

    1. Open waterways that ARE mapped in OpenStreetMap
       (``waterway=river/stream/canal/drain/ditch``) are returned as
       conduits with ``source="openstreetmap"`` and
       ``is_modelled=False``. Their geometry and names are real.
    2. Underground storm drainage in urban areas is *modelled* from the
       real road network topology, in the same spirit as
       ``backend/ml/synthetic_drainage.py``: road intersections become
       junction nodes (returned as "manholes"), and major-road segments
       between them become conduits, sized by road hierarchy
       (trunk/primary -> 1200 mm, secondary/tertiary -> 800 mm,
       other -> 450 mm) with a default slope. These items carry
       ``source="modelled_from_road_network"`` and ``is_modelled=True``.
       Their locations follow real roads; their diameters and
       capacities are engineering estimates, NOT surveyed municipal
       data. Present them as a modelled drainage network.
    3. Rural areas return no modelled drainage (matching the previous
       behaviour), though real OSM waterways are still returned.

OFFLINE HYDERABAD EXTRACT (LOCAL-FIRST)
    In production (Render), Overpass requests time out often enough
    that live fetching fell back to the synthetic layout and the flood
    router reported ``route_source="no_path"``. So for the Hyderabad
    area this module first tries a precomputed offline extract
    (``backend/data/hyderabad_osm.part*.json`` plus a
    ``hyderabad_osm.manifest.json`` index, built by
    ``backend/scripts/build_hyderabad_extract.py`` from OpenStreetMap
    data, refreshed monthly or on demand; the extract is split into
    <= ~4.5 MB parts because the GitHub transport used for this repo
    rejects larger single files). If the requested point and
    its full search radius lie inside the extract coverage, roads and
    waterways are served from that file - same shaping code, same
    contract, ``data_source="openstreetmap"`` plus a
    ``data_extract_date`` field so the snapshot's age is visible.
    Points outside coverage use the live Overpass path below. If the
    extract files are missing or corrupt, the module logs a warning and
    falls through to the live path; it never crashes on a bad file.

FALLBACK
    If OSMnx is not installed, Overpass is unreachable, or no roads are
    found, the module falls back to ``get_fast_fallback()`` - the same
    small synthetic layout the old implementation used - and marks the
    result with ``data_source="synthetic_fallback"`` so callers and the
    UI can tell it apart from real data.

RETURN CONTRACT (unchanged - backend/main.py and the frontend depend on it)
    {
        "zone_classification": str,
        "is_urban": bool,
        "streets": [
            {"id", "name", "lat", "lng", "coordinates", "is_subway",
             "surface", ...},
            ...
        ],
        "conduits": [
            {"id", "name", "from_node", "to_node", "length_m",
             "diameter_mm", "slope", "coordinates", "capacity_used_pct",
             "status", ...},
            ...
        ],
        "manholes": [
            {"id", "name", "lat", "lng", "type", "capacity_used_pct",
             "flow_rate_m3s", "surcharge_status", "is_surcharging",
             "is_blocked_inlet", ...},
            ...
        ],
        # Extra metadata keys (additive only, safe for existing callers):
        "data_source": "openstreetmap" | "synthetic_fallback",
        "drainage_source": str,
        "source_attribution": "(c) OpenStreetMap contributors",
        # Present only when served from the offline extract:
        "data_extract_date": "YYYY-MM-DD",
    }

CONFIGURATION (environment variables, all optional)
    HYDROCAST_OSM_NETWORK_TYPE   OSMnx network type. Default "drive".
                                 Use "all" to also include service roads,
                                 paths and tracks (larger, slower).
    HYDROCAST_OSM_MAX_STREETS    Max road segments returned. Default 150.
    HYDROCAST_OSM_MAX_CONDUITS   Max modelled conduits. Default 80.
    HYDROCAST_OSM_MAX_MANHOLES   Max modelled junction nodes. Default 50.
    HYDROCAST_OSM_TIMEOUT        Overpass request timeout (s). Default 180.
    HYDROCAST_OSM_CACHE_TTL      Seconds a cached area stays valid.
                                 Default 1800 (30 minutes).
    HYDROCAST_OSM_EXTRACT        Path to a single-file offline Hyderabad
                                 extract JSON (local experiments). When
                                 unset, the split extract in backend/data/
                                 (hyderabad_osm.manifest.json +
                                 hyderabad_osm.part*.json) is used.
    HYDROCAST_OSM_EXTRACT_MANIFEST
                                 Path to the split extract's manifest.
                                 Default backend/data/hyderabad_osm.manifest.json.
    HYDROCAST_OVERPASS_URL       Overpass base URL (no "/interpreter").
                                 Default https://overpass.private.coffee/api
                                 (global mirror, no rate limit stated on
                                 the OSM wiki). Falls back to the main
                                 instance https://overpass-api.de/api if
                                 the primary endpoint fails. A full
                                 ".../api/interpreter" URL is accepted
                                 and normalised to its base.

DEPENDENCIES
    ``osmnx`` is already listed in backend/requirements.txt. It pulls in
    geopandas, networkx, pandas and shapely. This module no longer needs
    ``google-genai`` or a GEMINI_API_KEY (the SitRep service still does).

OpenStreetMap data is (c) OpenStreetMap contributors, available under
the Open Database License (ODbL). Keep the attribution if you publish
maps built from this data.
"""

import copy
import gc
import gzip
import json
import math
import os
import time
from array import array
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["fetch_roads_near_point", "get_fast_fallback"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable, or return the default."""
    raw = os.getenv(name, "")
    try:
        value = int(raw)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


NETWORK_TYPE = os.getenv("HYDROCAST_OSM_NETWORK_TYPE", "drive").strip() or "drive"
MAX_STREETS = _env_int("HYDROCAST_OSM_MAX_STREETS", 150)
MAX_CONDUITS = _env_int("HYDROCAST_OSM_MAX_CONDUITS", 80)
MAX_MANHOLES = _env_int("HYDROCAST_OSM_MAX_MANHOLES", 50)
MAX_WATERWAY_CONDUITS = _env_int("HYDROCAST_OSM_MAX_WATERWAYS", 30)
OVERPASS_TIMEOUT_SECONDS = _env_int("HYDROCAST_OSM_TIMEOUT", 180)
CACHE_TTL_SECONDS = _env_int("HYDROCAST_OSM_CACHE_TTL", 1800)
CACHE_MAX_ENTRIES = 32

# Overpass endpoint (base URL, no "/interpreter"). Default is the
# private.coffee mirror: global coverage, 4 servers, and the OSM wiki
# states no rate limit is in place (it asks to be notified before
# large-scale use). The main FOSSGIS instance is the fallback, so a
# mirror outage degrades to the old behaviour instead of the synthetic
# fallback. Set HYDROCAST_OVERPASS_URL to try another mirror, or to
# force the main instance, with no code change.
OVERPASS_URL = (
    os.getenv("HYDROCAST_OVERPASS_URL", "https://overpass.private.coffee/api").strip()
    or "https://overpass.private.coffee/api"
)
OVERPASS_FALLBACK_URL = "https://overpass-api.de/api"


def _overpass_base(url: str) -> str:
    """Normalise an Overpass URL to its base (strip "/interpreter")."""
    base = url.strip().rstrip("/")
    if base.endswith("/interpreter"):
        base = base[: -len("/interpreter")]
    return base


def _overpass_endpoints() -> List[str]:
    """Primary endpoint first, main instance second, de-duplicated."""
    endpoints: List[str] = []
    for url in (OVERPASS_URL, OVERPASS_FALLBACK_URL):
        base = _overpass_base(url)
        if base and base not in endpoints:
            endpoints.append(base)
    return endpoints

# Highway classes in priority order: major corridors are the roads that
# matter most for flood blockage reporting, so they are returned first.
HIGHWAY_RANK: Dict[str, int] = {
    "motorway": 0,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 3,
    "primary": 4,
    "primary_link": 5,
    "secondary": 6,
    "secondary_link": 7,
    "tertiary": 8,
    "tertiary_link": 9,
    "unclassified": 10,
    "residential": 11,
    "living_street": 12,
    "service": 13,
    "track": 14,
}

# Roads at or above this rank threshold (i.e. tertiary or bigger) carry
# the modelled storm-drainage network.
MAJOR_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}

# Nominal conduit diameter (mm) by highway class, mirroring the sizing
# logic already used in backend/ml/synthetic_drainage.py.
DIAMETER_BY_HIGHWAY: Dict[str, int] = {
    "motorway": 1200,
    "motorway_link": 1200,
    "trunk": 1200,
    "trunk_link": 1200,
    "primary": 1200,
    "primary_link": 1200,
    "secondary": 800,
    "secondary_link": 800,
    "tertiary": 800,
    "tertiary_link": 800,
}
DEFAULT_DIAMETER_MM = 450

# Nominal channel size (mm equivalent) for real OSM open waterways.
DIAMETER_BY_WATERWAY: Dict[str, int] = {
    "river": 1500,
    "canal": 1200,
    "stream": 800,
    "drain": 600,
    "ditch": 450,
}

DEFAULT_CONDUIT_SLOPE = 0.005
DEFAULT_WATERWAY_SLOPE = 0.003
BASELINE_CAPACITY_USED_PCT = 40
BASELINE_FLOW_RATE_M3S = 3.0

# OSM surface values that backend/main.py treats as flood-vulnerable
# ("unpaved", "soil", "dirt") plus the paved defaults.
_UNPAVED_SURFACES = {"unpaved", "gravel", "sand", "grass", "compacted", "fine_gravel"}
_SOIL_SURFACES = {"dirt", "soil", "earth", "ground", "mud"}


# ---------------------------------------------------------------------------
# Small geospatial / tag helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS84 points, in metres."""
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.asin(math.sqrt(a))


def _coords_length_m(coords: Sequence[Sequence[float]]) -> float:
    """Total polyline length in metres for a [[lat, lng], ...] list."""
    total = 0.0
    for first, second in zip(coords, coords[1:]):
        total += _haversine_m(first[0], first[1], second[0], second[1])
    return total


def _downsample_coords(
    coords: List[List[float]], max_points: int = 25
) -> List[List[float]]:
    """Evenly reduce a polyline to at most ``max_points`` points.

    Keeps the first and last points so the geometry still starts and
    ends in the right place, and keeps payloads small for the frontend.
    """
    if len(coords) <= max_points:
        return coords
    step = (len(coords) - 1) / float(max_points - 1)
    sampled = [coords[round(i * step)] for i in range(max_points)]
    sampled[0] = coords[0]
    sampled[-1] = coords[-1]
    return sampled


def _clean_tag(value: Any) -> str:
    """Normalise an OSM tag value to a single clean string.

    OSMnx may return a tag as a string, a list of strings (multi-value
    tags), a float NaN (missing), or None.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            cleaned = _clean_tag(item)
            if cleaned:
                return cleaned
        return ""
    return str(value).strip()


def _highway_rank(highway: str) -> int:
    return HIGHWAY_RANK.get(highway, 15)


def _surface_from_tags(surface_tag: str, highway: str) -> str:
    """Map OSM surface/highway tags to the surface values main.py expects."""
    surface = surface_tag.lower()
    if surface in _SOIL_SURFACES:
        return "soil" if surface == "soil" else "dirt"
    if surface in _UNPAVED_SURFACES:
        return "unpaved"
    if surface:
        # asphalt, concrete, paving_stones, sett, etc. are all paved.
        return "asphalt"
    if highway == "track":
        return "unpaved"
    return "asphalt"


def _is_subway_from_tags(name: str, data: Dict[str, Any]) -> bool:
    """Best-effort detection of underpasses/subways from OSM tags.

    OSM does not have a dedicated "subway" (underpass) tag for roads;
    tunnels and underground locations, plus names containing
    "subway"/"underpass", are the available signals. These low points
    are exactly what main.py floods first.
    """
    tunnel = _clean_tag(data.get("tunnel")).lower()
    location = _clean_tag(data.get("location")).lower()
    lowered_name = name.lower()
    return (
        tunnel not in ("", "no")
        or location == "underground"
        or "subway" in lowered_name
        or "underpass" in lowered_name
    )


def _diameter_for_highway(highway: str) -> int:
    return DIAMETER_BY_HIGHWAY.get(highway, DEFAULT_DIAMETER_MM)


# ---------------------------------------------------------------------------
# In-memory cache (Overpass is a shared public service - be a good citizen)
# ---------------------------------------------------------------------------

_CACHE: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()


def _cache_key(lat: float, lng: float, radius_meters: int) -> str:
    # Round to ~110 m so nearby clicks reuse the same fetched area.
    return f"{lat:.3f}:{lng:.3f}:{radius_meters}:{NETWORK_TYPE}"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, payload = entry
    if time.time() - stored_at > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    # main.py mutates conduit dicts in place, so never hand out the
    # cached object itself.
    return copy.deepcopy(payload)


def _cache_put(key: str, payload: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), copy.deepcopy(payload))
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX_ENTRIES:
        _CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Offline Hyderabad extract (local-first; see module docstring)
# ---------------------------------------------------------------------------

_EXTRACT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
)

EXTRACT_PATH = os.getenv(
    "HYDROCAST_OSM_EXTRACT",
    os.path.join(_EXTRACT_DIR, "hyderabad_osm.json"),
)

EXTRACT_MANIFEST_PATH = os.getenv(
    "HYDROCAST_OSM_EXTRACT_MANIFEST",
    os.path.join(_EXTRACT_DIR, "hyderabad_osm.manifest.json"),
)

_EXTRACT: Optional[Dict[str, Any]] = None
_EXTRACT_LOAD_ATTEMPTED = False


def _read_extract_json(path: str) -> Any:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_extract_parts() -> Iterator[Dict[str, Any]]:
    """Yield the extract's parsed JSON parts in storage order.

    The committed extract is split into <= ~4.5 MB part files listed in
    a manifest (the GitHub transport used for this repo rejects larger
    single files). Each part holds consecutive chunks of the top-level
    arrays, so consuming them in manifest order keeps every stored
    node/name index reference valid. A single-file extract
    (HYDROCAST_OSM_EXTRACT) is still honoured when set.
    """
    if not os.getenv("HYDROCAST_OSM_EXTRACT") and os.path.exists(
        EXTRACT_MANIFEST_PATH
    ):
        manifest = _read_extract_json(EXTRACT_MANIFEST_PATH)
        base_dir = os.path.dirname(os.path.abspath(EXTRACT_MANIFEST_PATH))
        for part_name in manifest["parts"]:
            yield _read_extract_json(
                os.path.join(base_dir, str(part_name))
            )
        return
    yield _read_extract_json(EXTRACT_PATH)


class _Vocab:
    """Intern short repeated tag strings (surface/tunnel/location)."""

    def __init__(self) -> None:
        self.values: List[str] = []
        self._index: Dict[str, int] = {}

    def intern(self, value: Any) -> int:
        text = str(value or "")
        idx = self._index.get(text)
        if idx is None:
            idx = len(self.values)
            self.values.append(text)
            self._index[text] = idx
        return idx


def _grid_key(cell_x: int, cell_y: int) -> int:
    """Integer grid-cell key (cell_y is < 100000 for any real lng)."""
    return cell_x * 100000 + cell_y


def _build_compact_extract() -> Dict[str, Any]:
    """Parse the extract and convert it to compact flat arrays.

    MEMORY (why this looks the way it does): the parsed JSON object
    graph for the whole city is ~170 MB of Python objects (471k edge
    rows as nested lists, 1.6M int objects). The first loader kept
    that graph AND a decoded list-of-[lat, lng]-lists node table,
    peaking at ~310 MB for the extract alone - which OOM-killed the
    512 MB production instance on its first prediction. So here each
    part is converted into flat ``array`` tables as soon as it is
    parsed and the parsed chunk is released before the next part is
    read: node coordinates become two array('i') columns, edge node
    sequences one concatenated array('i') plus an offsets table, tag
    strings become vocab indices, and the midpoint grid maps integer
    cell keys to array('i') edge lists. Retained footprint is ~25 MB;
    the per-query code resolves only the edges in the query's cells.
    """
    metadata: Optional[Dict[str, Any]] = None
    names: List[str] = []
    waterways: List[Any] = []
    node_lat: array = array("i")
    node_lng: array = array("i")
    edge_offsets: array = array("i", [0])
    edge_nodes: array = array("i")
    edge_name: array = array("i")
    edge_hw: array = array("b")
    edge_sf: array = array("i")
    edge_tun: array = array("i")
    edge_loc: array = array("i")
    sf_vocab = _Vocab()
    tun_vocab = _Vocab()
    loc_vocab = _Vocab()
    grid_lists: Dict[int, List[int]] = {}
    run_lat = 0
    run_lng = 0
    edge_index = 0
    seen_nodes = False
    seen_edges = False

    for part in _iter_extract_parts():
        if "metadata" in part:
            metadata = part["metadata"]
        if "names" in part:
            names = part["names"]
        if "waterways" in part:
            waterways = part["waterways"]
        flat = part.get("nodes")
        if flat:
            seen_nodes = True
            # The flat table is interleaved [dlat, dlng, ...] deltas;
            # the running sums carry across part boundaries because
            # parts hold consecutive chunks of the one table.
            for i in range(0, len(flat) - 1, 2):
                run_lat += int(flat[i])
                run_lng += int(flat[i + 1])
                node_lat.append(run_lat)
                node_lng.append(run_lng)
        raw_edges = part.get("edges")
        if raw_edges:
            seen_edges = True
            if metadata is None:
                raise ValueError("extract edges precede metadata")
            coord_scale = float(metadata["coordinate_scale"])
            cell_degrees = float(metadata["grid_cell_degrees"])
            for raw in raw_edges:
                sequence = raw[0]
                node_idx = int(sequence[0])
                edge_nodes.append(node_idx)
                midpoint_pos = len(sequence) // 2
                midpoint_idx = node_idx
                for pos in range(1, len(sequence)):
                    node_idx += int(sequence[pos])
                    edge_nodes.append(node_idx)
                    if pos == midpoint_pos:
                        midpoint_idx = node_idx
                edge_offsets.append(len(edge_nodes))
                edge_name.append(int(raw[1]))
                edge_hw.append(int(raw[2]))
                edge_sf.append(
                    sf_vocab.intern(raw[3] if len(raw) > 3 else "")
                )
                edge_tun.append(
                    tun_vocab.intern(raw[4] if len(raw) > 4 else "")
                )
                edge_loc.append(
                    loc_vocab.intern(raw[5] if len(raw) > 5 else "")
                )
                # Bucket the edge by its midpoint, as the file's
                # grid_cell_degrees intends (the grid is deliberately
                # not stored in the file; rebuilding it costs a
                # fraction of a second and saves ~3 MB of JSON).
                mid_lat = node_lat[midpoint_idx] / coord_scale
                mid_lng = node_lng[midpoint_idx] / coord_scale
                key = _grid_key(
                    math.floor(mid_lat / cell_degrees),
                    math.floor(mid_lng / cell_degrees),
                )
                grid_lists.setdefault(key, []).append(edge_index)
                edge_index += 1
        # Release this part's parsed object graph before reading the
        # next one, so load-time peak stays near one part + arrays.
        del part

    if metadata is None or not seen_nodes or not seen_edges or not names:
        raise ValueError("extract is missing required sections")
    # Touch the required metadata keys so a truncated/foreign file
    # fails here rather than at query time.
    _ = (
        metadata["bbox"],
        metadata["coordinate_scale"],
        metadata["grid_cell_degrees"],
        metadata["hw_values"],
    )
    grid = {key: array("i", idxs) for key, idxs in grid_lists.items()}
    del grid_lists
    gc.collect()
    return {
        "metadata": metadata,
        "names": names,
        "waterways": waterways,
        "node_lat": node_lat,
        "node_lng": node_lng,
        "edge_offsets": edge_offsets,
        "edge_nodes": edge_nodes,
        "edge_name": edge_name,
        "edge_hw": edge_hw,
        "edge_sf": edge_sf,
        "sf_values": sf_vocab.values,
        "edge_tun": edge_tun,
        "tun_values": tun_vocab.values,
        "edge_loc": edge_loc,
        "loc_values": loc_vocab.values,
        "grid": grid,
        "edge_count_total": edge_index,
    }


def _load_extract() -> Optional[Dict[str, Any]]:
    """Load the offline extract once, lazily. None if unavailable.

    A missing or corrupt file must never break the service: log and
    return None so callers fall through to the live Overpass path.
    """
    global _EXTRACT, _EXTRACT_LOAD_ATTEMPTED
    if _EXTRACT_LOAD_ATTEMPTED:
        return _EXTRACT
    _EXTRACT_LOAD_ATTEMPTED = True
    try:
        data = _build_compact_extract()
        metadata = data["metadata"]
        if os.getenv("HYDROCAST_OSM_EXTRACT"):
            extract_source = EXTRACT_PATH
        else:
            extract_source = (
                f"{EXTRACT_MANIFEST_PATH} (split part files)"
            )
        _EXTRACT = data
        print(
            f"[i] Offline OSM extract loaded: {extract_source} "
            f"({metadata.get('edge_count', data['edge_count_total'])} "
            f"edges, extract date "
            f"{metadata.get('extract_date', 'unknown')})."
        )
    except Exception as error:  # noqa: BLE001 - fall through to live path
        print(
            f"[!] Warning: offline OSM extract unavailable ({error}). "
            "Using live Overpass fetching."
        )
        _EXTRACT = None
    return _EXTRACT


def _extract_covers(lat: float, lng: float, radius_meters: int) -> bool:
    """True if the point AND its full search circle sit inside coverage."""
    data = _load_extract()
    if data is None:
        return False
    south, west, north, east = (float(v) for v in data["metadata"]["bbox"])
    lat_margin = radius_meters / 111320.0
    lng_margin = radius_meters / (
        111320.0 * max(0.1, math.cos(math.radians(lat)))
    )
    return (
        south + lat_margin <= lat <= north - lat_margin
        and west + lng_margin <= lng <= east - lng_margin
    )


def _extract_coords(
    data: Dict[str, Any], node_indices: Sequence[int]
) -> List[List[float]]:
    """Materialise [[lat, lng], ...] for a stored node-index sequence."""
    scale = float(data["metadata"]["coordinate_scale"])
    node_lat = data["node_lat"]
    node_lng = data["node_lng"]
    return [
        [node_lat[idx] / scale, node_lng[idx] / scale]
        for idx in node_indices
    ]


def _delta_decode(sequence: Sequence[int]) -> List[int]:
    decoded = [int(sequence[0])]
    for delta in sequence[1:]:
        decoded.append(decoded[-1] + int(delta))
    return decoded


def _decode_extract_edge(
    data: Dict[str, Any], edge_index: int
) -> Dict[str, Any]:
    """Decode one format-2 edge, by index, into the builder's dict shape."""
    start = data["edge_offsets"][edge_index]
    end = data["edge_offsets"][edge_index + 1]
    return {
        "n": list(data["edge_nodes"][start:end]),
        "name": data["names"][data["edge_name"][edge_index]],
        "hw": data["metadata"]["hw_values"][data["edge_hw"][edge_index]],
        "sf": data["sf_values"][data["edge_sf"][edge_index]],
        "tun": data["tun_values"][data["edge_tun"][edge_index]],
        "loc": data["loc_values"][data["edge_loc"][edge_index]],
    }


def _decode_extract_waterway(data: Dict[str, Any], raw: Any) -> Dict[str, Any]:
    """Decode one format-2 waterway row into the builder's dict shape."""
    return {
        "n": _delta_decode(raw[0]),
        "name": data["names"][int(raw[1])],
        "ww": raw[2],
    }


def _extract_local_edges(
    lat: float, lng: float, radius_meters: int
) -> Tuple[List[Dict[str, Any]], Any]:
    """Road edges near a point from the offline extract.

    Returns edges in exactly the shape ``_extract_edges`` produces for
    the live path, plus a minimal graph object (``.nodes`` mapping
    node-index -> {"x": lng, "y": lat}) so ``_build_modelled_drainage``
    works unchanged. Raises if the extract cannot serve the point.
    """
    data = _load_extract()
    if data is None:
        raise RuntimeError("Offline OSM extract is not available")

    cell_degrees = float(data["metadata"]["grid_cell_degrees"])
    lat_margin = radius_meters / 111320.0
    lng_margin = radius_meters / (
        111320.0 * max(0.1, math.cos(math.radians(lat)))
    )

    candidate_indices: set = set()
    cell_x_range = range(
        math.floor((lat - lat_margin) / cell_degrees),
        math.floor((lat + lat_margin) / cell_degrees) + 1,
    )
    cell_y_range = range(
        math.floor((lng - lng_margin) / cell_degrees),
        math.floor((lng + lng_margin) / cell_degrees) + 1,
    )
    grid = data["grid"]
    for cell_x in cell_x_range:
        for cell_y in cell_y_range:
            candidate_indices.update(
                grid.get(_grid_key(cell_x, cell_y), ())
            )

    edges: List[Dict[str, Any]] = []
    node_positions: Dict[Any, Dict[str, float]] = {}
    seen: set = set()
    for index in candidate_indices:
        raw = _decode_extract_edge(data, index)
        node_seq = raw["n"]
        if len(node_seq) < 2:
            continue
        coords = _extract_coords(data, node_seq)
        length_m = _coords_length_m(coords)
        mid_lat, mid_lng = coords[len(coords) // 2]
        distance_m = _haversine_m(lat, lng, mid_lat, mid_lng)
        # Keep edges that at least partly cross the search circle.
        if distance_m > radius_meters + length_m / 2.0:
            continue
        dedupe_key = (
            frozenset((node_seq[0], node_seq[-1])),
            str(raw.get("name", "")).lower(),
            round(coords[0][0], 5),
            round(coords[0][1], 5),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        u, v = node_seq[0], node_seq[-1]
        for node_idx, (node_lat, node_lng) in (
            (u, coords[0]),
            (v, coords[-1]),
        ):
            node_positions.setdefault(
                node_idx, {"x": float(node_lng), "y": float(node_lat)}
            )

        name = str(raw.get("name", "") or "")
        highway = str(raw.get("hw", "") or "unclassified")
        edges.append(
            {
                "u": u,
                "v": v,
                "name": name,
                "highway": highway,
                "coords": _downsample_coords(coords),
                "lat": float(mid_lat),
                "lng": float(mid_lng),
                "length_m": round(float(length_m), 1),
                "surface": _surface_from_tags(str(raw.get("sf", "") or ""), highway),
                "is_subway": _is_subway_from_tags(
                    name,
                    {
                        "tunnel": str(raw.get("tun", "") or ""),
                        "location": str(raw.get("loc", "") or ""),
                    },
                ),
                "distance_m": distance_m,
            }
        )

    if not edges:
        raise ValueError("No offline-extract roads found around this point")

    class _LocalGraph:
        """Minimal stand-in for the OSMnx graph used by drainage code."""

        def __init__(self, nodes: Dict[Any, Dict[str, float]]):
            self.nodes = nodes

    return edges, _LocalGraph(node_positions)


def _extract_local_waterways(
    lat: float, lng: float, radius_meters: int
) -> List[Dict[str, Any]]:
    """Real OSM waterways near a point, from the offline extract.

    Same conduit shape as ``_fetch_waterway_conduits``.
    """
    data = _load_extract()
    if data is None:
        return []

    found: List[Tuple[float, Dict[str, Any]]] = []
    for encoded in data["waterways"]:
        raw = _decode_extract_waterway(data, encoded)
        node_seq = raw["n"]
        if len(node_seq) < 2:
            continue
        coords = _extract_coords(data, node_seq)
        length_m = _coords_length_m(coords)
        mid_lat, mid_lng = coords[len(coords) // 2]
        distance_m = _haversine_m(lat, lng, mid_lat, mid_lng)
        if distance_m > radius_meters + length_m / 2.0:
            continue
        found.append((distance_m, raw))

    found.sort(key=lambda item: item[0])
    conduits: List[Dict[str, Any]] = []
    for _distance, raw in found[:MAX_WATERWAY_CONDUITS]:
        coords = _downsample_coords(_extract_coords(data, raw["n"]))
        waterway_type = str(raw.get("ww", "") or "stream")
        name = str(raw.get("name", "") or "") or f"Unnamed {waterway_type}"
        public_id = f"wc-{len(conduits) + 1}"
        conduits.append(
            {
                "id": public_id,
                "name": name,
                "from_node": f"{public_id}-a",
                "to_node": f"{public_id}-b",
                "length_m": int(round(_coords_length_m(coords))),
                "diameter_mm": DIAMETER_BY_WATERWAY.get(
                    waterway_type, DEFAULT_DIAMETER_MM
                ),
                "slope": DEFAULT_WATERWAY_SLOPE,
                "coordinates": coords,
                "capacity_used_pct": BASELINE_CAPACITY_USED_PCT,
                "status": "normal",
                # Additive metadata:
                "source": "openstreetmap",
                "is_modelled": False,
                "channel_type": waterway_type,
                "diameter_is_estimate": True,
            }
        )
    return conduits


# ---------------------------------------------------------------------------
# OpenStreetMap fetching (OSMnx / Overpass)
# ---------------------------------------------------------------------------

def _import_osmnx(endpoint: Optional[str] = None):
    """Import OSMnx lazily so this module still loads without it.

    Sets the Overpass endpoint for this fetch. OSMnx 2.x reads
    ``settings.overpass_url`` (base URL, no "/interpreter"); OSMnx 1.x
    reads ``settings.overpass_endpoint`` (full ".../api/interpreter"
    URL). Both are set, each guarded, so the same code works on
    either version and an unknown attribute never breaks the fetch.
    """
    import osmnx as ox

    base = _overpass_base(endpoint or OVERPASS_URL)
    full = base + "/interpreter"
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.requests_timeout = OVERPASS_TIMEOUT_SECONDS
    except Exception:
        # Settings attributes differ slightly across OSMnx versions;
        # failing to set them must not break the fetch itself.
        pass
    for attr, value in (("overpass_url", base), ("overpass_endpoint", full)):
        try:
            setattr(ox.settings, attr, value)
        except Exception:
            pass
    return ox


def _fetch_road_graph(lat: float, lng: float, radius_meters: int):
    """Download the real road graph around a point from OpenStreetMap.

    Tries each Overpass endpoint in turn (primary mirror, then the
    main instance) so one slow/down mirror does not by itself force
    the synthetic fallback.
    """
    last_error: Optional[Exception] = None
    for endpoint in _overpass_endpoints():
        try:
            ox = _import_osmnx(endpoint)
            return ox.graph_from_point(
                (lat, lng),
                dist=radius_meters,
                network_type=NETWORK_TYPE,
                simplify=True,
                retain_all=False,
                truncate_by_edge=True,
            )
        except Exception as error:  # noqa: BLE001 - try next endpoint
            last_error = error
            print(
                f"[!] Warning: Overpass endpoint {endpoint} failed "
                f"({error}); trying the next endpoint."
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Overpass endpoint configured")


def _node_lat_lng(graph, node_id: Any) -> Optional[Tuple[float, float]]:
    try:
        node = graph.nodes[node_id]
    except Exception:
        return None
    lng = node.get("x")
    lat = node.get("y")
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


def _geometry_coords(geometry: Any) -> List[List[float]]:
    """Extract [[lat, lng], ...] from a shapely-like LineString geometry."""
    if geometry is None:
        return []
    coords = getattr(geometry, "coords", None)
    if coords is not None:
        try:
            return [[float(y), float(x)] for x, y in list(coords)]
        except Exception:
            return []
    geoms = getattr(geometry, "geoms", None)
    if geoms is not None:
        # MultiLineString: use the longest part as the representative path.
        best: List[List[float]] = []
        for part in list(geoms):
            part_coords = _geometry_coords(part)
            if len(part_coords) > len(best):
                best = part_coords
        return best
    return []


def _extract_edges(graph, lat: float, lng: float) -> List[Dict[str, Any]]:
    """Normalise an OSMnx graph into a de-duplicated list of road edges.

    OSMnx drive graphs are directed, so most roads appear twice
    (u -> v and v -> u). Duplicates are removed on the unordered node
    pair + road name so each physical road segment is returned once.
    """
    raw_edges: List[Dict[str, Any]] = []
    seen: set = set()

    try:
        edge_items = graph.edges(keys=True, data=True)
    except TypeError:
        edge_items = graph.edges(data=True)

    for item in edge_items:
        if len(item) == 4:
            u, v, _key, data = item
        elif len(item) == 3:
            u, v, data = item
        else:
            continue
        if not isinstance(data, dict):
            data = {}

        start = _node_lat_lng(graph, u)
        end = _node_lat_lng(graph, v)
        if start is None or end is None:
            continue

        coords = _geometry_coords(data.get("geometry"))
        if len(coords) < 2:
            coords = [[start[0], start[1]], [end[0], end[1]]]
        coords = _downsample_coords(coords)

        name = _clean_tag(data.get("name"))
        highway = _clean_tag(data.get("highway")) or "unclassified"

        dedupe_key = (
            frozenset((str(u), str(v))),
            name.lower(),
            round(coords[0][0], 5),
            round(coords[0][1], 5),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        length_m = data.get("length")
        if not isinstance(length_m, (int, float)) or (
            isinstance(length_m, float) and math.isnan(length_m)
        ):
            length_m = _coords_length_m(coords)

        mid_lat, mid_lng = coords[len(coords) // 2]
        raw_edges.append(
            {
                "u": u,
                "v": v,
                "name": name,
                "highway": highway,
                "coords": coords,
                "lat": float(mid_lat),
                "lng": float(mid_lng),
                "length_m": round(float(length_m), 1),
                "surface": _surface_from_tags(
                    _clean_tag(data.get("surface")), highway
                ),
                "is_subway": _is_subway_from_tags(name, data),
                "distance_m": _haversine_m(lat, lng, mid_lat, mid_lng),
            }
        )

    return raw_edges


def _estimate_is_urban(edges: List[Dict[str, Any]], radius_meters: int) -> bool:
    """Estimate urban-ness from real road-network density.

    This replaces the old keyword check on the place name (which only
    recognised a handful of hand-listed neighbourhoods). Urban areas
    have a dense road network; rural catchments do not.
    """
    if not edges:
        return False
    area_km2 = math.pi * (radius_meters / 1000.0) ** 2
    if area_km2 <= 0:
        return False
    total_road_km = sum(edge["length_m"] for edge in edges) / 1000.0
    density_km_per_km2 = total_road_km / area_km2
    return density_km_per_km2 >= 3.5 or len(edges) >= 120


def _build_streets(edges: List[Dict[str, Any]], place_name: str) -> List[Dict[str, Any]]:
    """Build the ``streets`` list in the contract shape, major roads first."""
    ordered = sorted(
        edges,
        key=lambda edge: (
            _highway_rank(edge["highway"]),
            edge["distance_m"],
        ),
    )[:MAX_STREETS]

    streets: List[Dict[str, Any]] = []
    for index, edge in enumerate(ordered, start=1):
        name = edge["name"] or f"Unnamed {edge['highway'].replace('_', ' ')} road"
        streets.append(
            {
                "id": f"s-{index}",
                "name": name,
                "lat": edge["lat"],
                "lng": edge["lng"],
                "coordinates": edge["coords"],
                "is_subway": edge["is_subway"],
                "surface": edge["surface"],
                # Additive metadata:
                "highway": edge["highway"],
                "length_m": edge["length_m"],
                "source": "openstreetmap",
                "is_modelled": False,
            }
        )
    return streets


# ---------------------------------------------------------------------------
# Modelled drainage (from real road topology) + real OSM waterways
# ---------------------------------------------------------------------------

def _node_display_names(edges: List[Dict[str, Any]]) -> Dict[Any, str]:
    """Map each graph node to the first named road that passes through it."""
    names: Dict[Any, str] = {}
    for edge in edges:
        if not edge["name"]:
            continue
        for node_id in (edge["u"], edge["v"]):
            names.setdefault(node_id, edge["name"])
    return names


def _build_modelled_drainage(
    graph,
    edges: List[Dict[str, Any]],
    lat: float,
    lng: float,
    place_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Model storm drainage from major-road intersections and segments.

    Junction nodes (manholes) are real road intersections. Conduits are
    real major-road segments between those intersections, sized by road
    hierarchy. Diameters/capacities are estimates - see module docstring.
    """
    major_edges = [e for e in edges if e["highway"] in MAJOR_HIGHWAYS]
    if not major_edges:
        return [], []

    # Candidate junction nodes: endpoints of major roads, nearest first.
    candidate_nodes: Dict[Any, Tuple[float, float, float]] = {}
    for edge in major_edges:
        for node_id in (edge["u"], edge["v"]):
            if node_id in candidate_nodes:
                continue
            position = _node_lat_lng(graph, node_id)
            if position is None:
                continue
            distance = _haversine_m(lat, lng, position[0], position[1])
            candidate_nodes[node_id] = (position[0], position[1], distance)

    selected_nodes = dict(
        sorted(candidate_nodes.items(), key=lambda item: item[1][2])[:MAX_MANHOLES]
    )
    if not selected_nodes:
        return [], []

    node_names = _node_display_names(major_edges)
    node_ids: Dict[Any, str] = {}
    manholes: List[Dict[str, Any]] = []
    for index, (node_id, (node_lat, node_lng, _dist)) in enumerate(
        selected_nodes.items(), start=1
    ):
        public_id = f"m-{index}"
        node_ids[node_id] = public_id
        street_name = node_names.get(node_id)
        display_name = (
            f"{street_name} Junction {index}"
            if street_name
            else f"{place_name} Junction {index}"
        )
        manholes.append(
            {
                "id": public_id,
                "name": display_name,
                "lat": node_lat,
                "lng": node_lng,
                "type": "manhole",
                "capacity_used_pct": BASELINE_CAPACITY_USED_PCT,
                "flow_rate_m3s": BASELINE_FLOW_RATE_M3S,
                "surcharge_status": "normal",
                "is_surcharging": False,
                "is_blocked_inlet": False,
                # Additive metadata:
                "source": "modelled_from_road_network",
                "is_modelled": True,
            }
        )

    conduit_edges = [
        edge
        for edge in major_edges
        if edge["u"] in node_ids and edge["v"] in node_ids
    ]
    conduit_edges.sort(
        key=lambda edge: (_highway_rank(edge["highway"]), edge["distance_m"])
    )

    conduits: List[Dict[str, Any]] = []
    for index, edge in enumerate(conduit_edges[:MAX_CONDUITS], start=1):
        conduits.append(
            {
                "id": f"c-{index}",
                "name": (
                    f"Stormwater Trunk - {edge['name']}"
                    if edge["name"]
                    else f"Stormwater Trunk {index}"
                ),
                "from_node": node_ids[edge["u"]],
                "to_node": node_ids[edge["v"]],
                "length_m": int(round(edge["length_m"])),
                "diameter_mm": _diameter_for_highway(edge["highway"]),
                "slope": DEFAULT_CONDUIT_SLOPE,
                "coordinates": edge["coords"],
                "capacity_used_pct": BASELINE_CAPACITY_USED_PCT,
                "status": "normal",
                # Additive metadata:
                "source": "modelled_from_road_network",
                "is_modelled": True,
                "diameter_is_estimate": True,
            }
        )

    return conduits, manholes


def _feature_value(row: Any, key: str) -> str:
    """Read a tag from a GeoDataFrame row (Series-like) defensively."""
    try:
        getter = getattr(row, "get", None)
        if callable(getter):
            return _clean_tag(getter(key))
    except Exception:
        pass
    try:
        return _clean_tag(row[key])
    except Exception:
        return ""


def _feature_geometry(row: Any):
    geometry = getattr(row, "geometry", None)
    if geometry is not None:
        return geometry
    try:
        return row["geometry"]
    except Exception:
        return None


def _fetch_waterway_conduits(
    lat: float, lng: float, radius_meters: int
) -> List[Dict[str, Any]]:
    """Fetch real open waterways (rivers/streams/canals/drains) from OSM.

    These are genuine mapped features, returned as conduits. Failures
    here must never break the road result, so callers wrap this in
    try/except and this function itself returns [] on empty results.
    """
    last_error: Optional[Exception] = None
    for endpoint in _overpass_endpoints():
        try:
            ox = _import_osmnx(endpoint)
            tags = {"waterway": ["river", "stream", "canal", "drain", "ditch"]}
            features = ox.features_from_point((lat, lng), tags=tags, dist=radius_meters)
            break
        except Exception as error:  # noqa: BLE001 - try next endpoint
            last_error = error
            print(
                f"[!] Warning: Overpass endpoint {endpoint} failed for "
                f"waterways ({error}); trying the next endpoint."
            )
            features = None
    else:
        return []
    if features is None:
        return []

    try:
        rows = list(features.iterrows())
    except Exception:
        return []

    conduits: List[Dict[str, Any]] = []
    for _index, row in rows:
        if len(conduits) >= MAX_WATERWAY_CONDUITS:
            break
        coords = _downsample_coords(_geometry_coords(_feature_geometry(row)))
        if len(coords) < 2:
            continue
        waterway_type = _feature_value(row, "waterway") or "stream"
        name = _feature_value(row, "name") or f"Unnamed {waterway_type}"
        public_id = f"wc-{len(conduits) + 1}"
        conduits.append(
            {
                "id": public_id,
                "name": name,
                "from_node": f"{public_id}-a",
                "to_node": f"{public_id}-b",
                "length_m": int(round(_coords_length_m(coords))),
                "diameter_mm": DIAMETER_BY_WATERWAY.get(
                    waterway_type, DEFAULT_DIAMETER_MM
                ),
                "slope": DEFAULT_WATERWAY_SLOPE,
                "coordinates": coords,
                "capacity_used_pct": BASELINE_CAPACITY_USED_PCT,
                "status": "normal",
                # Additive metadata:
                "source": "openstreetmap",
                "is_modelled": False,
                "channel_type": waterway_type,
                "diameter_is_estimate": True,
            }
        )
    return conduits


# ---------------------------------------------------------------------------
# Shared result assembly (live Overpass path and offline extract path)
# ---------------------------------------------------------------------------

def _assemble_result(
    graph,
    edges: List[Dict[str, Any]],
    lat: float,
    lng: float,
    radius_meters: int,
    resolved_place: str,
    waterway_conduits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the contract result from normalised edges.

    Used by both the live Overpass path and the offline extract path so
    streets, modelled drainage and metadata are shaped identically
    regardless of where the road geometry came from.
    """
    is_urban = _estimate_is_urban(edges, radius_meters)
    zone_type = (
        f"Urban Municipal Sector ({resolved_place})"
        if is_urban
        else f"Rural Catchment / Outskirts ({resolved_place})"
    )

    streets = _build_streets(edges, resolved_place)

    conduits: List[Dict[str, Any]] = []
    manholes: List[Dict[str, Any]] = []
    drainage_notes = ["no modelled drainage (rural area)"]
    if is_urban:
        conduits, manholes = _build_modelled_drainage(
            graph, edges, lat, lng, resolved_place
        )
        drainage_notes = ["modelled from road network topology"]

    if waterway_conduits:
        conduits = conduits + waterway_conduits
        drainage_notes.append(f"{len(waterway_conduits)} real OSM waterway(s)")

    return {
        "zone_classification": zone_type,
        "is_urban": is_urban,
        "streets": streets,
        "conduits": conduits,
        "manholes": manholes,
        # Additive metadata:
        "data_source": "openstreetmap",
        "drainage_source": "; ".join(drainage_notes),
        "source_attribution": "(c) OpenStreetMap contributors",
    }


# ---------------------------------------------------------------------------
# Public entry point (signature and return shape unchanged)
# ---------------------------------------------------------------------------

def fetch_roads_near_point(
    lat: float, lng: float, radius_meters: int = 2000, place_name: str = ""
) -> dict:
    """Return roads and drainage infrastructure around a point.

    Roads and open waterways are real OpenStreetMap data. Underground
    drainage is modelled from the road network and tagged as such.
    Falls back to a small synthetic layout if real data cannot be
    fetched, with ``data_source="synthetic_fallback"``.
    """
    if not (-90.0 <= float(lat) <= 90.0) or not (-180.0 <= float(lng) <= 180.0):
        raise ValueError(f"Invalid coordinates: lat={lat}, lng={lng}")
    if radius_meters <= 0:
        raise ValueError(f"radius_meters must be positive, got {radius_meters}")

    resolved_place = (
        place_name.strip()
        if place_name and place_name.strip()
        else f"[{lat:.4f}, {lng:.4f}]"
    )

    cache_key = _cache_key(float(lat), float(lng), int(radius_meters))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Local-first: serve from the offline Hyderabad extract when the
    # point and its full search radius are inside extract coverage.
    # Any failure here falls through to the live path below.
    if _extract_covers(float(lat), float(lng), int(radius_meters)):
        try:
            local_edges, local_graph = _extract_local_edges(
                float(lat), float(lng), int(radius_meters)
            )
            local_waterways = _extract_local_waterways(
                float(lat), float(lng), int(radius_meters)
            )
            local_result = _assemble_result(
                local_graph,
                local_edges,
                float(lat),
                float(lng),
                int(radius_meters),
                resolved_place,
                local_waterways,
            )
            extract = _load_extract()
            local_result["data_extract_date"] = str(
                (extract or {}).get("metadata", {}).get("extract_date", "")
            )
            _cache_put(cache_key, local_result)
            return local_result
        except Exception as local_error:
            print(
                f"[!] Warning: offline OSM extract lookup failed "
                f"({local_error}). Falling back to live Overpass fetching."
            )

    try:
        graph = _fetch_road_graph(float(lat), float(lng), int(radius_meters))
        edges = _extract_edges(graph, float(lat), float(lng))
        if not edges:
            raise ValueError("No OpenStreetMap roads found around this point")

        waterway_conduits: List[Dict[str, Any]] = []
        try:
            waterway_conduits = _fetch_waterway_conduits(
                float(lat), float(lng), int(radius_meters)
            )
        except Exception as waterway_error:
            print(
                f"[!] Warning: OSM waterway fetch failed ({waterway_error}). "
                "Continuing with roads and modelled drainage only."
            )

        result = _assemble_result(
            graph,
            edges,
            float(lat),
            float(lng),
            int(radius_meters),
            resolved_place,
            waterway_conduits,
        )
        _cache_put(cache_key, result)
        return result

    except Exception as error:
        print(
            f"[!] Warning: OpenStreetMap fetch failed ({error}). "
            "Reverting to fast synthetic fallback."
        )
        # Urban-ness for the fallback uses the place-name hint, as the
        # old implementation did, because no network data is available.
        fallback_is_urban = any(
            keyword in resolved_place.lower()
            for keyword in [
                "kondapur", "madhapur", "hyderabad", "patancheru",
                "mumbai", "dadar", "bengaluru", "delhi", "sector", "ward",
            ]
        )
        return get_fast_fallback(
            float(lat), float(lng), resolved_place, fallback_is_urban
        )


# ---------------------------------------------------------------------------
# Synthetic fallback (retained from the previous implementation)
# ---------------------------------------------------------------------------

def get_fast_fallback(
    lat: float, lng: float, place_name: str, is_urban: bool
) -> dict:
    """Small synthetic layout used only when real OSM data is unavailable.

    Kept so the application still runs offline or if Overpass is down.
    Everything here is invented - the ``data_source`` field says so.
    """
    streets = [
        {
            "id": "f-1",
            "name": f"Main Arterial - {place_name}",
            "lat": lat,
            "lng": lng,
            "coordinates": [
                [lat - 0.003, lng - 0.003],
                [lat - 0.001, lng - 0.001],
                [lat, lng],
                [lat + 0.002, lng + 0.001],
                [lat + 0.004, lng + 0.003],
            ],
            "is_subway": is_urban,
            "surface": "asphalt" if is_urban else "dirt",
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
        {
            "id": "f-2",
            "name": f"Bypass Ridge Way - {place_name}",
            "lat": lat + 0.002,
            "lng": lng - 0.002,
            "coordinates": [
                [lat - 0.002, lng + 0.003],
                [lat + 0.001, lng + 0.001],
                [lat + 0.002, lng - 0.002],
                [lat + 0.005, lng - 0.004],
            ],
            "is_subway": False,
            "surface": "asphalt",
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
        {
            "id": "f-3",
            "name": f"Local Access Route - {place_name}",
            "lat": lat - 0.002,
            "lng": lng + 0.002,
            "coordinates": [
                [lat - 0.004, lng - 0.001],
                [lat - 0.002, lng + 0.002],
                [lat, lng + 0.003],
            ],
            "is_subway": False,
            "surface": "asphalt" if is_urban else "unpaved",
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
    ]

    conduits = [
        {
            "id": "fc-1",
            "name": "Conduit Trunk 1",
            "from_node": "m-1",
            "to_node": "m-2",
            "length_m": 120,
            "diameter_mm": 900,
            "slope": 0.005,
            "coordinates": [[lat, lng], [lat + 0.002, lng + 0.002]],
            "capacity_used_pct": 50,
            "status": "normal",
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
        {
            "id": "fc-2",
            "name": "Conduit Trunk 2",
            "from_node": "m-2",
            "to_node": "m-3",
            "length_m": 180,
            "diameter_mm": 1200,
            "slope": 0.004,
            "coordinates": [[lat + 0.002, lng + 0.002], [lat + 0.004, lng + 0.003]],
            "capacity_used_pct": 75,
            "status": "normal",
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
    ] if is_urban else []

    manholes = [
        {
            "id": "m-1",
            "name": f"{place_name} Junction A",
            "lat": lat,
            "lng": lng,
            "type": "manhole",
            "capacity_used_pct": 45,
            "flow_rate_m3s": 2.5,
            "surcharge_status": "normal",
            "is_surcharging": False,
            "is_blocked_inlet": False,
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
        {
            "id": "m-2",
            "name": f"{place_name} Junction B",
            "lat": lat + 0.002,
            "lng": lng + 0.002,
            "type": "manhole",
            "capacity_used_pct": 78,
            "flow_rate_m3s": 4.0,
            "surcharge_status": "warning",
            "is_surcharging": False,
            "is_blocked_inlet": False,
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
        {
            "id": "m-3",
            "name": f"{place_name} Junction C",
            "lat": lat + 0.004,
            "lng": lng + 0.003,
            "type": "manhole",
            "capacity_used_pct": 92,
            "flow_rate_m3s": 5.2,
            "surcharge_status": "overflow",
            "is_surcharging": True,
            "is_blocked_inlet": True,
            "source": "synthetic_fallback",
            "is_modelled": True,
        },
    ] if is_urban else []

    return {
        "zone_classification": (
            f"Urban Municipal Sector ({place_name})"
            if is_urban
            else f"Rural Outskirts ({place_name})"
        ),
        "is_urban": is_urban,
        "streets": streets,
        "conduits": conduits,
        "manholes": manholes,
        "data_source": "synthetic_fallback",
        "drainage_source": "synthetic fallback (no real data available)",
        "source_attribution": "Synthetic fallback data - not real infrastructure",
    }
