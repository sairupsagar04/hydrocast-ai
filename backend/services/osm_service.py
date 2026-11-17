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

DEPENDENCIES
    ``osmnx`` is already listed in backend/requirements.txt. It pulls in
    geopandas, networkx, pandas and shapely. This module no longer needs
    ``google-genai`` or a GEMINI_API_KEY (the SitRep service still does).

OpenStreetMap data is (c) OpenStreetMap contributors, available under
the Open Database License (ODbL). Keep the attribution if you publish
maps built from this data.
"""

import copy
import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# OpenStreetMap fetching (OSMnx / Overpass)
# ---------------------------------------------------------------------------

def _import_osmnx():
    """Import OSMnx lazily so this module still loads without it."""
    import osmnx as ox

    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.requests_timeout = OVERPASS_TIMEOUT_SECONDS
    except Exception:
        # Settings attributes differ slightly across OSMnx versions;
        # failing to set them must not break the fetch itself.
        pass
    return ox


def _fetch_road_graph(lat: float, lng: float, radius_meters: int):
    """Download the real road graph around a point from OpenStreetMap."""
    ox = _import_osmnx()
    return ox.graph_from_point(
        (lat, lng),
        dist=radius_meters,
        network_type=NETWORK_TYPE,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )


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
    ox = _import_osmnx()
    tags = {"waterway": ["river", "stream", "canal", "drain", "ditch"]}
    features = ox.features_from_point((lat, lng), tags=tags, dist=radius_meters)
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

    try:
        graph = _fetch_road_graph(float(lat), float(lng), int(radius_meters))
        edges = _extract_edges(graph, float(lat), float(lng))
        if not edges:
            raise ValueError("No OpenStreetMap roads found around this point")

        is_urban = _estimate_is_urban(edges, int(radius_meters))
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
                graph, edges, float(lat), float(lng), resolved_place
            )
            drainage_notes = ["modelled from road network topology"]

        try:
            waterway_conduits = _fetch_waterway_conduits(
                float(lat), float(lng), int(radius_meters)
            )
            if waterway_conduits:
                conduits = conduits + waterway_conduits
                drainage_notes.append(
                    f"{len(waterway_conduits)} real OSM waterway(s)"
                )
        except Exception as waterway_error:
            print(
                f"[!] Warning: OSM waterway fetch failed ({waterway_error}). "
                "Continuing with roads and modelled drainage only."
            )

        result: Dict[str, Any] = {
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
