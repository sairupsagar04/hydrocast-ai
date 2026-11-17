# backend/gis/spatial_engine.py
"""
SpatialEngine — real terrain (DEM) grids for HydroCast AI Model 2.

Drop-in replacement for the previous implementation, whose
``fetch_real_elevation_profile()`` never fetched anything: it synthesised
terrain from coordinate-seeded sine waves, so the slope and depression
channels fed to the inundation model described imaginary ground.

This version fetches a REAL digital elevation model for the requested
bounding box, in order:

1. **Terrarium terrain tiles** (primary)
   ``https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png``
   Free, no API key. The tiles encode elevation in the PNG channels
   (``elev = R*256 + G + B/256 - 32768``) and are derived from SRTM and
   other public DEM sources (~30 m ground resolution at these zooms).
   The tiles covering the bounding box are downloaded, decoded, mosaicked,
   cropped to the exact lat/lng window and resampled to the model grid.

2. **Open-Meteo Elevation API** (fallback)
   ``https://api.open-meteo.com/v1/elevation`` — free, no API key,
   Copernicus DEM based. A coarse grid (default 16x16) is sampled over the
   bounding box and bilinearly upsampled to the model grid.

3. **Synthetic terrain** (last resort, clearly labelled)
   The original sine-wave synthesis, kept only so the service still runs
   fully offline. ``engine.elevation_source`` reports
   ``"synthetic_fallback"`` whenever this path produced the grid, so
   callers and logs can tell real terrain from placeholder terrain.

Public API (unchanged from the previous implementation):

    engine = SpatialEngine(center_lat, center_lng, grid_size=256)
    dem_grid, elev_var = engine.fetch_real_elevation_profile()
    stack = engine.get_model2_feature_stack(is_urban=True)  # (4, N, N)
    row, col = engine.latlng_to_grid(lat, lng)

Feature stack channels (order unchanged, units now physical):

    0  DEM elevation            metres above sea level (real DEM)
    1  Slope magnitude          metres per metre (rise/run, from real DEM)
    2  Surface roughness        Manning-style coefficient (constant per
                                urban/rural class, as before)
    3  Depression / ponding     0..1 — low-lying fraction derived from the
       susceptibility           real DEM: lowest cells -> retention_mult,
                                highest cells -> 0

Configuration via environment variables (all optional):

    HYDROCAST_DEM_SOURCE       auto | tiles | open_meteo | synthetic
                               (default: auto)
    HYDROCAST_DEM_ZOOM         terrain tile zoom, 10..15 (default: 14)
    HYDROCAST_DEM_TIMEOUT      per-request HTTP timeout seconds
                               (default: 10)
    HYDROCAST_DEM_API_SAMPLE   Open-Meteo sample grid size, 8..32
                               (default: 16)

Notes:
- DEM data is static, so decoded tiles are kept in a bounded in-memory
  LRU cache and each engine instance caches its own elevation grid.
- The grid convention matches the previous implementation exactly:
  row 0 is the NORTHERN edge (center_lat + dlat), column 0 is the
  WESTERN edge (center_lng - dlng), consistent with latlng_to_grid().
"""

from __future__ import annotations

import io
import math
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# --------------------------------------------------------------------------
# Constants and configuration
# --------------------------------------------------------------------------

TILE_URL_TEMPLATE = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

TILE_PIXELS = 256
MAX_MERCATOR_LAT = 85.05112878  # Web Mercator limit used by slippy maps
METERS_PER_DEGREE_LAT = 111_320.0

USER_AGENT = "HydroCast-AI/4.1 (backend DEM fetch; spatial_engine)"

# Bounded LRU cache of decoded tiles: (z, x, y) -> elevation array (float32)
_TILE_CACHE: "OrderedDict[Tuple[int, int, int], np.ndarray]" = OrderedDict()
_TILE_CACHE_MAX = 256
# Negative cache: (z, x, y) -> monotonic timestamp of last failed fetch,
# so a dead tile is retried at most once per retry window per process.
_TILE_FAILURES: Dict[Tuple[int, int, int], float] = {}
_TILE_FAILURE_RETRY_SECONDS = 600.0


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# Web Mercator / slippy-map helpers
# --------------------------------------------------------------------------


def _lng_to_world_x(lng: float, zoom: int) -> float:
    """Longitude -> horizontal position in tile units (float)."""
    return (lng + 180.0) / 360.0 * (2**zoom)


def _lat_to_world_y(lat: float, zoom: int) -> float:
    """Latitude -> vertical position in tile units (float, grows south)."""
    lat = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, lat))
    rad = math.radians(lat)
    return (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * (
        2**zoom
    )


# --------------------------------------------------------------------------
# Image decoding and resampling helpers
# --------------------------------------------------------------------------


def _decode_terrarium_png(content: bytes) -> np.ndarray:
    """
    Decode a Terrarium-format PNG into an elevation array in metres.

    Terrarium encoding: elevation = R * 256 + G + B / 256 - 32768.
    OpenCV is preferred (it is already a backend dependency); Pillow is
    used when OpenCV is unavailable. Note OpenCV loads images as BGR,
    so its channel order is reversed relative to the PNG's RGB.
    """
    try:
        import cv2  # opencv-python-headless, already in requirements.txt

        bgr = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None and bgr.ndim == 3 and bgr.shape[2] >= 3:
            r = bgr[:, :, 2].astype(np.float32)
            g = bgr[:, :, 1].astype(np.float32)
            b = bgr[:, :, 0].astype(np.float32)
            return (r * 256.0 + g + b / 256.0 - 32768.0).astype(np.float32)
    except ImportError:
        pass

    from PIL import Image  # Pillow

    with Image.open(io.BytesIO(content)) as img:
        rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (r * 256.0 + g + b / 256.0 - 32768.0).astype(np.float32)


def _resize_bilinear(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resample a 2-D float grid to (out_h, out_w) with bilinear interpolation."""
    in_h, in_w = grid.shape
    if in_h == out_h and in_w == out_w:
        return grid.astype(np.float32, copy=True)

    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (in_w / out_w) - 0.5
    ys = np.clip(ys, 0.0, in_h - 1)
    xs = np.clip(xs, 0.0, in_w - 1)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]

    src = grid.astype(np.float32, copy=False)
    top = src[np.ix_(y0, x0)] * (1.0 - wx) + src[np.ix_(y0, x1)] * wx
    bottom = src[np.ix_(y1, x0)] * (1.0 - wx) + src[np.ix_(y1, x1)] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _despike(grid: np.ndarray, threshold_m: float = 75.0) -> np.ndarray:
    """
    Repair single-pixel elevation spikes. Terrarium/SRTM tiles contain
    occasional void pixels encoded as garbage (e.g. ~-1,100 m in the
    middle of Hyderabad). A pixel that differs from the median of its
    8 neighbours by more than ``threshold_m`` metres is replaced by that
    median — real terrain at ~30 m resolution essentially never does
    this, while the void artifacts are hundreds of metres off.
    """
    h, w = grid.shape
    padded = np.pad(grid, 1, mode="edge")
    neighbours = np.stack(
        [
            padded[di : di + h, dj : dj + w]
            for di in range(3)
            for dj in range(3)
            if not (di == 1 and dj == 1)
        ]
    )
    local_median = np.median(neighbours, axis=0).astype(np.float32)
    mask = np.abs(grid - local_median) > threshold_m
    if not mask.any():
        return grid
    repaired = grid.copy()
    repaired[mask] = local_median[mask]
    print(f"[!] DEM despike: repaired {int(mask.sum())} spike pixel(s)")
    return repaired


def _validate_dem(grid: np.ndarray, grid_size: int) -> np.ndarray:
    """Ensure a candidate DEM is the right shape and physically plausible."""
    if grid.shape != (grid_size, grid_size):
        raise ValueError(f"DEM shape {grid.shape} != {(grid_size, grid_size)}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("DEM contains non-finite values")
    # Highest terrain on Earth is ~8,849 m; ocean trenches are not land DEM.
    if float(grid.min()) < -500.0 or float(grid.max()) > 9000.0:
        raise ValueError("DEM values outside plausible elevation range")
    return grid.astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# SpatialEngine
# --------------------------------------------------------------------------


class SpatialEngine:
    def __init__(self, center_lat: float, center_lng: float, grid_size: int = 256):
        self.center_lat = float(center_lat)
        self.center_lng = float(center_lng)
        self.grid_size = int(grid_size)
        self.dlat = 0.02  # ~2.2 km bounding box window
        self.dlng = 0.02

        # Which source produced the current grid: one of
        # "terrarium_tiles", "open_meteo", "synthetic_fallback", or None
        # before the first fetch.
        self.elevation_source: Optional[str] = None

        self._dem_grid: Optional[np.ndarray] = None

        self._source_mode = os.environ.get("HYDROCAST_DEM_SOURCE", "auto").strip().lower()
        if self._source_mode not in ("auto", "tiles", "open_meteo", "synthetic"):
            self._source_mode = "auto"
        self._tile_zoom = _env_int("HYDROCAST_DEM_ZOOM", 14, 10, 15)
        self._timeout = _env_float("HYDROCAST_DEM_TIMEOUT", 10.0, 2.0, 30.0)
        self._api_sample = _env_int("HYDROCAST_DEM_API_SAMPLE", 16, 8, 32)

    # ------------------------------------------------------------------
    # Geographic mapping (unchanged semantics)
    # ------------------------------------------------------------------

    def latlng_to_grid(self, lat: float, lng: float) -> Tuple[int, int]:
        """Maps geographic (lat, lng) to (row, col) grid coordinates."""
        r = int(((self.center_lat + self.dlat - lat) / (2 * self.dlat)) * self.grid_size)
        c = int(((lng - (self.center_lng - self.dlng)) / (2 * self.dlng)) * self.grid_size)
        return max(0, min(self.grid_size - 1, r)), max(0, min(self.grid_size - 1, c))

    def _bbox(self) -> Tuple[float, float, float, float]:
        """(lat_north, lat_south, lng_west, lng_east) of the grid window."""
        return (
            self.center_lat + self.dlat,
            self.center_lat - self.dlat,
            self.center_lng - self.dlng,
            self.center_lng + self.dlng,
        )

    # ------------------------------------------------------------------
    # Source 1: Terrarium terrain tiles (real DEM)
    # ------------------------------------------------------------------

    def _fetch_tile(self, z: int, x: int, y: int) -> Optional[np.ndarray]:
        """Download and decode one terrain tile, with LRU + failure caching."""
        key = (z, x, y)
        cached = _TILE_CACHE.get(key)
        if cached is not None:
            _TILE_CACHE.move_to_end(key)
            return cached

        last_failure = _TILE_FAILURES.get(key)
        if (
            last_failure is not None
            and (time.monotonic() - last_failure) < _TILE_FAILURE_RETRY_SECONDS
        ):
            return None

        url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
        for attempt in range(2):
            try:
                res = requests.get(
                    url, timeout=self._timeout, headers={"User-Agent": USER_AGENT}
                )
                if res.status_code == 200 and res.content:
                    tile = _decode_terrarium_png(res.content)
                    if tile.shape == (TILE_PIXELS, TILE_PIXELS):
                        _TILE_CACHE[key] = tile
                        _TILE_CACHE.move_to_end(key)
                        while len(_TILE_CACHE) > _TILE_CACHE_MAX:
                            _TILE_CACHE.popitem(last=False)
                        _TILE_FAILURES.pop(key, None)
                        return tile
            except Exception as exc:  # network error, decode error, ...
                print(f"[!] DEM tile {z}/{x}/{y} attempt {attempt + 1} failed: {exc}")
            if attempt == 0:
                time.sleep(0.3)

        _TILE_FAILURES[key] = time.monotonic()
        return None

    def _elevation_from_tiles(self) -> np.ndarray:
        """Build the DEM grid for the bounding box from terrain tiles."""
        zoom = self._tile_zoom
        lat_north, lat_south, lng_west, lng_east = self._bbox()

        # Bounding box in global pixel coordinates.
        px_west = _lng_to_world_x(lng_west, zoom) * TILE_PIXELS
        px_east = _lng_to_world_x(lng_east, zoom) * TILE_PIXELS
        px_north = _lat_to_world_y(lat_north, zoom) * TILE_PIXELS
        px_south = _lat_to_world_y(lat_south, zoom) * TILE_PIXELS

        max_tile = 2**zoom - 1
        tx_min = max(0, min(max_tile, int(math.floor(px_west / TILE_PIXELS))))
        tx_max = max(0, min(max_tile, int(math.floor((px_east - 1) / TILE_PIXELS))))
        ty_min = max(0, min(max_tile, int(math.floor(px_north / TILE_PIXELS))))
        ty_max = max(0, min(max_tile, int(math.floor((px_south - 1) / TILE_PIXELS))))

        tiles_x = tx_max - tx_min + 1
        tiles_y = ty_max - ty_min + 1
        mosaic = np.full(
            (tiles_y * TILE_PIXELS, tiles_x * TILE_PIXELS), np.nan, dtype=np.float32
        )

        fetched = 0
        total = tiles_x * tiles_y
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile = self._fetch_tile(zoom, tx, ty)
                if tile is None:
                    continue
                fetched += 1
                row0 = (ty - ty_min) * TILE_PIXELS
                col0 = (tx - tx_min) * TILE_PIXELS
                mosaic[row0 : row0 + TILE_PIXELS, col0 : col0 + TILE_PIXELS] = tile

        if fetched == 0:
            raise RuntimeError("no terrain tiles could be downloaded")

        # Fill any missing tiles with the mean of what we have; a small
        # flat patch is better than failing the whole request. If most of
        # the mosaic is missing, the tile source is effectively down.
        nan_fraction = float(np.isnan(mosaic).mean())
        if nan_fraction > 0.25:
            raise RuntimeError(f"terrain tiles mostly unavailable ({nan_fraction:.0%} missing)")
        if nan_fraction > 0.0:
            mosaic[np.isnan(mosaic)] = float(np.nanmean(mosaic))

        # Repair SRTM void spikes BEFORE resampling — afterwards they
        # would be diluted into plausible-looking but wrong values.
        mosaic = _despike(mosaic)

        # Crop the mosaic to the exact bounding box and resample.
        origin_x = tx_min * TILE_PIXELS
        origin_y = ty_min * TILE_PIXELS
        cx0 = int(math.floor(px_west - origin_x))
        cx1 = int(math.ceil(px_east - origin_x))
        cy0 = int(math.floor(px_north - origin_y))
        cy1 = int(math.ceil(px_south - origin_y))
        crop = mosaic[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            raise RuntimeError("terrain tile crop is empty")

        grid = _despike(_resize_bilinear(crop, self.grid_size, self.grid_size))
        return _validate_dem(grid, self.grid_size)

    # ------------------------------------------------------------------
    # Source 2: Open-Meteo Elevation API (real DEM, sampled)
    # ------------------------------------------------------------------

    def _elevation_from_open_meteo(self) -> np.ndarray:
        """Sample a coarse elevation grid from the Open-Meteo Elevation API."""
        sample = self._api_sample
        lat_north, lat_south, lng_west, lng_east = self._bbox()
        lats = np.linspace(lat_north, lat_south, sample)
        lngs = np.linspace(lng_west, lng_east, sample)

        points: List[Tuple[float, float]] = [
            (float(lat), float(lng)) for lat in lats for lng in lngs
        ]

        elevations: List[float] = []
        chunk_size = 100  # coordinates per request
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            res = requests.get(
                OPEN_METEO_ELEVATION_URL,
                params={
                    "latitude": ",".join(f"{p[0]:.6f}" for p in chunk),
                    "longitude": ",".join(f"{p[1]:.6f}" for p in chunk),
                },
                timeout=self._timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if res.status_code != 200:
                raise RuntimeError(f"Open-Meteo elevation HTTP {res.status_code}")
            values = res.json().get("elevation")
            if not isinstance(values, list) or len(values) != len(chunk):
                raise RuntimeError("Open-Meteo elevation returned unexpected payload")
            elevations.extend(0.0 if v is None else float(v) for v in values)

        coarse = np.array(elevations, dtype=np.float32).reshape(sample, sample)
        return _validate_dem(
            _resize_bilinear(coarse, self.grid_size, self.grid_size), self.grid_size
        )

    # ------------------------------------------------------------------
    # Source 3: synthetic terrain (offline last resort, labelled)
    # ------------------------------------------------------------------

    def _synthetic_elevation_profile(self) -> np.ndarray:
        """
        The original coordinate-seeded sine-wave terrain. Kept ONLY as an
        offline last resort; grids from here are placeholders, not terrain.
        """
        seed = int(abs(self.center_lat * 10000 + self.center_lng * 10000))
        np.random.seed(seed)

        y, x = np.ogrid[: self.grid_size, : self.grid_size]
        freq1, freq2 = 25.0 + (seed % 15), 50.0 + (seed % 20)
        dem_grid = (
            np.sin(x / freq1 + self.center_lng * 50) * 15.0
            + np.cos(y / freq2 + self.center_lat * 50) * 22.0
            + np.sin((x + y) / 40.0) * 8.0
            + 450.0
            + (seed % 100)
        ).astype(np.float32)
        return dem_grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_real_elevation_profile(self) -> Tuple[np.ndarray, float]:
        """
        Return (dem_grid, elevation_std) for this engine's bounding box,
        from a real DEM whenever one is reachable. The grid is cached per
        instance; ``self.elevation_source`` names the source that produced
        it ("terrarium_tiles", "open_meteo" or "synthetic_fallback").
        """
        if self._dem_grid is not None:
            return self._dem_grid.copy(), float(np.std(self._dem_grid))

        dem: Optional[np.ndarray] = None
        mode = self._source_mode

        if mode in ("auto", "tiles"):
            try:
                dem = self._elevation_from_tiles()
                self.elevation_source = "terrarium_tiles"
            except Exception as exc:
                print(f"[!] Terrain tiles unavailable, trying Open-Meteo: {exc}")
                if mode == "tiles":
                    dem = None

        if dem is None and mode in ("auto", "open_meteo"):
            try:
                dem = self._elevation_from_open_meteo()
                self.elevation_source = "open_meteo"
            except Exception as exc:
                print(f"[!] Open-Meteo elevation unavailable: {exc}")

        if dem is None:
            dem = self._synthetic_elevation_profile()
            self.elevation_source = "synthetic_fallback"
            if mode == "synthetic":
                print(
                    "[!] HYDROCAST_DEM_SOURCE=synthetic — using SYNTHETIC "
                    "terrain. Flood depths for this area are placeholders."
                )
            else:
                print(
                    "[!] WARNING: using SYNTHETIC terrain — no DEM source was "
                    "reachable. Flood depths for this area are placeholders."
                )

        self._dem_grid = dem
        return dem.copy(), float(np.std(dem))

    def get_model2_feature_stack(self, is_urban: bool = True) -> np.ndarray:
        """
        Build the (4, grid, grid) Model 2 input stack from the real DEM:
        [elevation (m), slope (m/m), roughness, depression (0..1)].
        """
        dem_grid, _ = self.fetch_real_elevation_profile()

        # Physical cell spacing, so slope is a true rise/run ratio.
        height_m = (2 * self.dlat) * METERS_PER_DEGREE_LAT
        width_m = (
            (2 * self.dlng)
            * METERS_PER_DEGREE_LAT
            * max(0.01, math.cos(math.radians(self.center_lat)))
        )
        cell_h = height_m / self.grid_size
        cell_w = width_m / self.grid_size

        grad_y, grad_x = np.gradient(dem_grid, cell_h, cell_w)
        slope = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        # Urban concrete vs rural soil friction (unchanged from before).
        if is_urban:
            roughness = np.full(
                (self.grid_size, self.grid_size), fill_value=0.015, dtype=np.float32
            )
            retention_mult = 0.92  # High runoff, low absorption
        else:
            roughness = np.full(
                (self.grid_size, self.grid_size), fill_value=0.140, dtype=np.float32
            )
            retention_mult = 0.18  # High absorption, low runoff

        # Depression/ponding susceptibility from the REAL DEM: the lowest
        # cells in the window get the full retention multiplier, the
        # highest get zero. (The previous implementation derived this
        # from the synthetic terrain and added coordinate-seeded noise;
        # the noise injection has been removed — the layer is now a pure
        # function of measured elevation.)
        min_elev = float(np.min(dem_grid))
        max_elev = float(np.max(dem_grid))
        elevation_range = max_elev - min_elev + 1e-5
        normalised = (dem_grid - min_elev) / elevation_range
        depression = np.clip((1.0 - normalised) * retention_mult, 0.0, 1.0).astype(
            np.float32
        )

        return np.stack([dem_grid, slope, roughness, depression], axis=0)
