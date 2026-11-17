# backend/routing/flood_router.py
"""
Real routing on the OpenStreetMap road graph for HydroCast AI.

Written 2026-11-17 as the replacement for the hardcoded Mumbai corridor
graph that used to live in this file (a FloodAwareRouter over fixed
Bandra/Milan Subway nodes, never called by main.py) and for the
list-slicing in main.py Stage 5, which built "routes" by taking the
first 6-8 streets in list order (not paths at all). main.py Stage 5 now
calls ``plan_routes`` below.

What it does
------------
Builds a routable graph from the streets ``fetch_roads_near_point`` already
returns (``backend/services/osm_service.py``). Each street carries its full
polyline in ``coordinates`` ([[lat, lng], ...], downsampled to at most 25
points, first/last points preserved), so the graph is joined by matching
polyline endpoints — no new Overpass/OSMnx call and no extra dependency.
Routing is Dijkstra over edge lengths (standard library ``heapq`` only).

Routes planned (both share the same endpoints, so they are comparable):
- ``standard_shortest`` — shortest path by length, ignoring flood closures.
- ``safe_bypass`` — shortest path with flooded/closed edges removed, and
  caution-depth edges (12-30 cm) penalised so it prefers dry roads.

Endpoints, because the prediction request carries only one point
---------------------------------------------------------------
The request has no destination, so the routes are evacuation corridors:
origin is the graph node nearest the queried centre; destination is the
farthest node from the centre that sits on at least one passable edge —
i.e. the edge of the mapped area, the way out. If the user later wants
point-to-point routing, pass ``destination_lat`` / ``destination_lng`` to
``plan_routes`` (nearest graph node to that point is used) instead.

Honest limits
-------------
- ``osm_service._build_streets`` drops the OSM node ids, so edges are
  joined by rounded endpoint coordinates (5 decimals, ~1 m). OSMnx splits
  edges at intersections, so shared intersections are shared endpoints;
  a junction that only touches another road mid-polyline will not join.
- Streets are capped at 150 (major roads first), so the graph covers the
  mapped area only; a destination beyond it does not exist.
- The synthetic fallback layout (``get_fast_fallback``) has three streets
  whose endpoints do not meet, so no path exists there — ``plan_routes``
  returns empty routes with ``route_source="no_path"`` and main.py keeps
  its old slice fallback for that case only, labelled via route_source.
- A "safe" route is safe only against the depths main.py computed for
  this request; it is not a surveyed evacuation route.

Contract with main.py
---------------------
main.py Stage 5 already computes, per road, ``depth_cm`` and ``is_closed``
(closed at >= 30 cm, plus submerged subways). Pass those in as
``depths_cm`` ({street id: depth in cm}) and ``closed_ids`` (set of street
ids). The return value keeps the existing response keys
``standard_shortest`` / ``safe_bypass`` as [[lat, lng], ...] lists, and
adds lengths, a source flag and whether a safe route exists, so the
frontend needs no change. It never invents coordinates: when no safe
path exists, ``safe_bypass`` is [] and ``safe_route_available`` is False
instead of the current fake diagonal ``[centre, centre + 0.01]`` fallback.
"""

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Same closure threshold main.py uses for "Closed / Flooded".
CLOSED_DEPTH_CM = 30.0
# Below this depth a road is treated as dry for routing weight.
CAUTION_DEPTH_CM = 12.0

# Node keys are endpoint coordinates rounded to 5 decimals (~1.1 m at
# Hyderabad's latitude). OSMnx geometry endpoints are the graph nodes, so
# two edges sharing an intersection share these exact coordinates.
_NODE_PRECISION = 5


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres between two lat/lng points."""
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lng / 2.0) ** 2
    )
    return 2.0 * 6371000.0 * math.asin(min(1.0, math.sqrt(a)))


def _node_key(lat: float, lng: float) -> Tuple[float, float]:
    return (round(float(lat), _NODE_PRECISION), round(float(lng), _NODE_PRECISION))


def _polyline_length_m(coords: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(coords, coords[1:]):
        total += _haversine_m(lat1, lng1, lat2, lng2)
    return total


def build_graph(streets: Sequence[Dict]) -> Tuple[Dict[Tuple[float, float], Tuple[float, float]], Dict[Tuple[float, float], list]]:
    """Build an undirected graph from osm_service street dicts.

    Returns ``(nodes, adjacency)`` where ``nodes`` maps a node key to its
    (lat, lng) and ``adjacency`` maps a node key to a list of
    ``(neighbour_key, street_index, length_m)``. Streets with fewer than
    two coordinates, or whose endpoints coincide, are skipped: a single
    point is not a routable edge.
    """
    nodes: Dict[Tuple[float, float], Tuple[float, float]] = {}
    adjacency: Dict[Tuple[float, float], list] = {}

    for index, street in enumerate(streets):
        coords = street.get("coordinates") or []
        if len(coords) < 2:
            continue
        start = _node_key(coords[0][0], coords[0][1])
        end = _node_key(coords[-1][0], coords[-1][1])
        if start == end:
            continue
        length_m = street.get("length_m")
        if not isinstance(length_m, (int, float)) or length_m <= 0:
            length_m = _polyline_length_m(coords)
        nodes[start] = (float(coords[0][0]), float(coords[0][1]))
        nodes[end] = (float(coords[-1][0]), float(coords[-1][1]))
        adjacency.setdefault(start, []).append((end, index, float(length_m)))
        adjacency.setdefault(end, []).append((start, index, float(length_m)))

    return nodes, adjacency


def _nearest_node(
    nodes: Dict[Tuple[float, float], Tuple[float, float]], lat: float, lng: float
) -> Optional[Tuple[float, float]]:
    best_key = None
    best_dist = None
    for key, (node_lat, node_lng) in nodes.items():
        dist = _haversine_m(lat, lng, node_lat, node_lng)
        if best_dist is None or dist < best_dist:
            best_key, best_dist = key, dist
    return best_key


def _exit_node(
    nodes: Dict[Tuple[float, float], Tuple[float, float]],
    adjacency: Dict[Tuple[float, float], list],
    streets: Sequence[Dict],
    centre_lat: float,
    centre_lng: float,
    closed_ids: Set[str],
) -> Optional[Tuple[float, float]]:
    """Farthest node from the centre that touches a passable edge.

    That is the edge of the mapped area on a road you can still use —
    the evacuation exit. Nodes whose only incident edges are closed are
    not eligible, so the safe route can at least end there.
    """
    best_key = None
    best_dist = -1.0
    for key, (node_lat, node_lng) in nodes.items():
        incident_open = any(
            streets[street_index].get("id") not in closed_ids
            for _, street_index, _ in adjacency.get(key, [])
        )
        if not incident_open:
            continue
        dist = _haversine_m(centre_lat, centre_lng, node_lat, node_lng)
        if dist > best_dist:
            best_key, best_dist = key, dist
    return best_key


def _edge_weight(
    length_m: float,
    street_id: Optional[str],
    depths_cm: Dict[str, float],
    closed_ids: Set[str],
    avoid_closed: bool,
) -> Optional[float]:
    """Routing cost of one edge, or None if it must not be used."""
    if avoid_closed and street_id in closed_ids:
        return None
    depth = depths_cm.get(street_id, 0.0) if street_id else 0.0
    if avoid_closed and depth >= CLOSED_DEPTH_CM:
        return None
    if depth > CAUTION_DEPTH_CM:
        # Prefer dry roads, but a caution-depth road is still usable when
        # it is the only way through: penalise, don't forbid.
        return length_m * (1.0 + depth / CLOSED_DEPTH_CM)
    return length_m


def _dijkstra(
    adjacency: Dict[Tuple[float, float], list],
    streets: Sequence[Dict],
    depths_cm: Dict[str, float],
    closed_ids: Set[str],
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    avoid_closed: bool,
) -> Optional[List[int]]:
    """Shortest path as a list of street indices, or None if unreachable."""
    dist: Dict[Tuple[float, float], float] = {origin: 0.0}
    prev: Dict[Tuple[float, float], Tuple[Tuple[float, float], int]] = {}
    queue: List[Tuple[float, Tuple[float, float]]] = [(0.0, origin)]
    visited: Set[Tuple[float, float]] = set()

    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            break
        if cost > dist.get(node, math.inf):
            continue
        for neighbour, street_index, length_m in adjacency.get(node, []):
            street_id = streets[street_index].get("id")
            weight = _edge_weight(
                length_m, street_id, depths_cm, closed_ids, avoid_closed
            )
            if weight is None:
                continue
            new_cost = cost + weight
            if new_cost < dist.get(neighbour, math.inf):
                dist[neighbour] = new_cost
                prev[neighbour] = (node, street_index)
                heapq.heappush(queue, (new_cost, neighbour))

    if destination not in prev and destination != origin:
        return None

    path: List[int] = []
    node = destination
    while node != origin:
        prev_node, street_index = prev[node]
        path.append(street_index)
        node = prev_node
    path.reverse()
    return path


def _path_coordinates(
    streets: Sequence[Dict],
    path: List[int],
    origin: Tuple[float, float],
) -> List[List[float]]:
    """Join the path's street polylines into one [[lat, lng], ...] route.

    Each polyline is oriented from the current node onwards, and the
    shared joint point is not duplicated.
    """
    coords: List[List[float]] = []
    current = origin
    for street_index in path:
        polyline = [list(point) for point in streets[street_index]["coordinates"]]
        start = _node_key(polyline[0][0], polyline[0][1])
        end = _node_key(polyline[-1][0], polyline[-1][1])
        if start == current:
            oriented = polyline
            current = end
        elif end == current:
            oriented = list(reversed(polyline))
            current = start
        else:
            # Should not happen for a graph-built path; keep the polyline
            # as-is rather than dropping the edge.
            oriented = polyline
            current = end
        if coords:
            oriented = oriented[1:]
        coords.extend(oriented)
    return coords


def plan_routes(
    streets: Sequence[Dict],
    depths_cm: Optional[Dict[str, float]] = None,
    closed_ids: Optional[Set[str]] = None,
    centre_lat: float = 0.0,
    centre_lng: float = 0.0,
    destination_lat: Optional[float] = None,
    destination_lng: Optional[float] = None,
) -> Dict:
    """Plan standard and safe-bypass routes on the real OSM road graph.

    ``streets`` is the ``streets`` list from ``fetch_roads_near_point``.
    ``depths_cm`` maps street id -> computed water depth in cm and
    ``closed_ids`` is the set of street ids main.py marked closed; both
    default to "no flooding" so the function is also usable standalone.

    Returns a dict with the existing response keys plus provenance::

        {
            "standard_shortest": [[lat, lng], ...],   # [] when no path
            "safe_bypass": [[lat, lng], ...],         # [] when no safe path
            "standard_length_m": float,
            "safe_length_m": float | None,
            "route_source": "osm_graph" | "no_path",
            "safe_route_available": bool,
            "standard_uses_closed_road": bool,
        }
    """
    depths_cm = depths_cm or {}
    closed_ids = set(closed_ids or [])

    nodes, adjacency = build_graph(streets)
    origin = _nearest_node(nodes, centre_lat, centre_lng) if nodes else None
    if origin is not None and destination_lat is not None and destination_lng is not None:
        # Explicit point-to-point routing: nearest graph node to the
        # requested destination.
        destination = _nearest_node(nodes, destination_lat, destination_lng)
    elif origin is not None:
        destination = _exit_node(
            nodes, adjacency, streets, centre_lat, centre_lng, closed_ids
        )
    else:
        destination = None
    if origin is None or destination is None or origin == destination:
        return {
            "standard_shortest": [],
            "safe_bypass": [],
            "standard_length_m": 0.0,
            "safe_length_m": None,
            "route_source": "no_path",
            "safe_route_available": False,
            "standard_uses_closed_road": False,
        }

    standard_path = _dijkstra(
        adjacency, streets, depths_cm, closed_ids, origin, destination, False
    )
    if standard_path is None:
        return {
            "standard_shortest": [],
            "safe_bypass": [],
            "standard_length_m": 0.0,
            "safe_length_m": None,
            "route_source": "no_path",
            "safe_route_available": False,
            "standard_uses_closed_road": False,
        }
    standard_coords = _path_coordinates(streets, standard_path, origin)
    standard_length = sum(
        _polyline_length_m(streets[i]["coordinates"]) for i in standard_path
    )
    standard_uses_closed = any(
        streets[i].get("id") in closed_ids
        or depths_cm.get(streets[i].get("id"), 0.0) >= CLOSED_DEPTH_CM
        for i in standard_path
    )

    safe_path = _dijkstra(
        adjacency, streets, depths_cm, closed_ids, origin, destination, True
    )
    if safe_path is None:
        return {
            "standard_shortest": standard_coords,
            "safe_bypass": [],
            "standard_length_m": round(standard_length, 1),
            "safe_length_m": None,
            "route_source": "osm_graph",
            "safe_route_available": False,
            "standard_uses_closed_road": standard_uses_closed,
        }
    safe_coords = _path_coordinates(streets, safe_path, origin)
    safe_length = sum(
        _polyline_length_m(streets[i]["coordinates"]) for i in safe_path
    )
    return {
        "standard_shortest": standard_coords,
        "safe_bypass": safe_coords,
        "standard_length_m": round(standard_length, 1),
        "safe_length_m": round(safe_length, 1),
        "route_source": "osm_graph",
        "safe_route_available": True,
        "standard_uses_closed_road": standard_uses_closed,
    }
