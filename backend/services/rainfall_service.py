# backend/services/rainfall_service.py
"""
Rainfall service — measured precipitation for HydroCast AI live mode.

Replaces main.py's old single-source helper, which had three problems:

1. Without an OPENWEATHER_API_KEY it returned 0.0 unconditionally, so
   live mode permanently reported "Clear Sky" regardless of the actual
   weather. There was no fallback source.
2. If OpenWeather's text description mentioned rain while the measured
   value was 0.0 mm, it invented a rate of 15.5 mm/hr.
3. It returned only an instantaneous rate, and main.py converted it to
   an event accumulation with a fudge factor (rate x 0.15), ignoring the
   request's lead_time_minutes entirely.

This service returns a RainfallReport built from two real sources:

- **OpenWeather** (primary for current rate, when a key is configured):
  the measured ``rain.1h`` value, used exactly as reported — never
  adjusted, never invented.
- **Open-Meteo forecast API** (no key required): hourly precipitation
  for the point, used for the lead-time accumulation, antecedent
  (past 3h / 24h) totals, and as the current-rate source whenever
  OpenWeather is unavailable or reports 0 while the model shows rain
  for the current hour.

The report distinguishes three states that used to be conflated:
actually raining (rate > 0), measured dry (a source responded with 0),
and no data (``data_available`` is False — rate 0 then means
"unknown", and callers should say so rather than claim clear skies).

Configuration via environment variables (all optional):

    OPENWEATHER_API_KEY          existing key; absence is fine now
    HYDROCAST_RAINFALL_SOURCE    auto | openweather | open_meteo
                                 (default: auto)

Source responses are cached per ~1 km cell for 5 minutes; the DEM-style
provenance string is carried on the report (``report.source``) so API
responses can show exactly where the rain numbers came from.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "HydroCast-AI/4.1 (backend rainfall fetch; rainfall_service)"

_CACHE_TTL_SECONDS = 300.0
# (lat2, lng2) -> (monotonic_ts, openweather_rate_or_None, meteo_series_or_None)
_SOURCE_CACHE: Dict[Tuple[float, float], Tuple[float, Optional[float], Optional[Tuple[List[datetime], List[float]]]]] = {}


@dataclass
class RainfallReport:
    """Precipitation picture for one point. All depths in mm, rate in mm/hr."""

    rate_mm_hr: float  # current precipitation rate
    accumulation_mm: float  # forecast accumulation over the lead-time window
    antecedent_3h_mm: float  # measured rain in the 3 hours before now
    antecedent_24h_mm: float  # measured rain in the 24 hours before now
    source: str  # e.g. "openweather+open_meteo", "open_meteo", "none"
    data_available: bool  # False => no source responded; zeros mean "unknown"
    observed_at: str  # ISO-8601 UTC timestamp of the report

    def to_dict(self) -> dict:
        return {
            "rate_mm_hr": self.rate_mm_hr,
            "accumulation_mm": self.accumulation_mm,
            "antecedent_3h_mm": self.antecedent_3h_mm,
            "antecedent_24h_mm": self.antecedent_24h_mm,
            "source": self.source,
            "data_available": self.data_available,
            "observed_at": self.observed_at,
        }


# ---------------------------------------------------------------------------
# Source fetchers (return None when the source cannot answer)
# ---------------------------------------------------------------------------


def _resolve_openweather_key(explicit: Optional[str]) -> Optional[str]:
    key = explicit if explicit is not None else os.environ.get("OPENWEATHER_API_KEY", "")
    key = (key or "").strip()
    if not key or key == "your_actual_api_key_here":
        return None
    return key


def _fetch_openweather_rate(lat: float, lng: float, api_key: str, timeout: float) -> Optional[float]:
    """
    Current rain rate (mm/hr) from OpenWeather's measured ``rain.1h``.
    Returns None on any failure. The value is used exactly as reported;
    a text description of rain with no measured amount is 0.0, not an
    invented constant (the previous implementation returned 15.5).
    """
    try:
        res = requests.get(
            OPENWEATHER_URL,
            params={"lat": lat, "lon": lng, "appid": api_key, "units": "metric"},
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        if res.status_code != 200:
            print(f"[!] OpenWeather HTTP {res.status_code} for rainfall")
            return None
        data = res.json()
        rain_data = data.get("rain") or {}
        return float(rain_data.get("1h", 0.0) or 0.0)
    except Exception as exc:
        print(f"[!] OpenWeather rainfall fetch failed: {exc}")
        return None


def _fetch_open_meteo_series(
    lat: float, lng: float, timeout: float
) -> Optional[Tuple[List[datetime], List[float]]]:
    """
    Hourly precipitation series (UTC) from Open-Meteo: 2 days past,
    2 days forecast. Each value is the precipitation total for that
    hour in mm, i.e. the mean rate in mm/hr over that hour.
    Returns (hour_start_times, mm_per_hour) or None on failure.
    """
    try:
        res = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lng,
                "hourly": "precipitation",
                "timezone": "GMT",
                "past_days": 2,
                "forecast_days": 2,
            },
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        if res.status_code != 200:
            print(f"[!] Open-Meteo HTTP {res.status_code} for rainfall")
            return None
        hourly = res.json().get("hourly") or {}
        raw_times = hourly.get("time") or []
        raw_values = hourly.get("precipitation") or []
        if not raw_times or len(raw_times) != len(raw_values):
            return None
        times = [
            datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            for t in raw_times
        ]
        values = [0.0 if v is None else float(v) for v in raw_values]
        return times, values
    except Exception as exc:
        print(f"[!] Open-Meteo rainfall fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Series maths (pure functions — each hourly value is a constant mm/hr
# rate over its hour bucket, so window totals are overlap integrals)
# ---------------------------------------------------------------------------


def _overlap_mm(
    times: List[datetime],
    values: List[float],
    start: datetime,
    end: datetime,
) -> float:
    """Precipitation total (mm) over [start, end] from an hourly series."""
    if end <= start:
        return 0.0
    total = 0.0
    one_hour = timedelta(hours=1)
    for t, v in zip(times, values):
        bucket_end = t + one_hour
        overlap_start = max(t, start)
        overlap_end = min(bucket_end, end)
        if overlap_end > overlap_start:
            total += v * (overlap_end - overlap_start).total_seconds() / 3600.0
    return total


def _rate_at(times: List[datetime], values: List[float], when: datetime) -> Optional[float]:
    """Rate (mm/hr) of the hour bucket containing ``when``."""
    one_hour = timedelta(hours=1)
    for t, v in zip(times, values):
        if t <= when < t + one_hour:
            return v
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _get_sources(
    lat: float, lng: float, mode: str, api_key: Optional[str], timeout: float
) -> Tuple[Optional[float], Optional[Tuple[List[datetime], List[float]]]]:
    """Fetch (or reuse cached) raw source data for a ~1 km cell."""
    cache_key = (round(lat, 2), round(lng, 2))
    cached = _SOURCE_CACHE.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1], cached[2]

    ow_rate: Optional[float] = None
    if mode in ("auto", "openweather") and api_key:
        ow_rate = _fetch_openweather_rate(lat, lng, api_key, timeout)

    series = None
    if mode in ("auto", "open_meteo"):
        series = _fetch_open_meteo_series(lat, lng, timeout)

    _SOURCE_CACHE[cache_key] = (time.monotonic(), ow_rate, series)
    return ow_rate, series


def fetch_rainfall(
    lat: float,
    lng: float,
    lead_time_minutes: int = 30,
    api_key: Optional[str] = None,
    timeout: float = 8.0,
    now: Optional[datetime] = None,
) -> RainfallReport:
    """
    Build the RainfallReport for a point.

    Current rate: OpenWeather's measured value when it reports rain;
    otherwise Open-Meteo's current-hour value (OpenWeather stations can
    report 0 between showers); otherwise OpenWeather's measured 0.
    Accumulation and antecedent totals come from the Open-Meteo hourly
    series; without it, accumulation falls back to persisting the
    current rate over the lead window (documented assumption).
    """
    mode = os.environ.get("HYDROCAST_RAINFALL_SOURCE", "auto").strip().lower()
    if mode not in ("auto", "openweather", "open_meteo"):
        mode = "auto"
    resolved_key = _resolve_openweather_key(api_key)
    if mode == "openweather" and not resolved_key:
        mode = "open_meteo"  # requested source cannot answer without a key

    now = now or datetime.now(timezone.utc)
    lead = max(0, int(lead_time_minutes))

    ow_rate, series = _get_sources(lat, lng, mode, resolved_key, timeout)

    sources: List[str] = []
    if ow_rate is not None:
        sources.append("openweather")
    if series is not None:
        sources.append("open_meteo")

    om_rate: Optional[float] = None
    accumulation_mm = 0.0
    antecedent_3h_mm = 0.0
    antecedent_24h_mm = 0.0
    if series is not None:
        times, values = series
        om_rate = _rate_at(times, values, now)
        accumulation_mm = _overlap_mm(times, values, now, now + timedelta(minutes=lead))
        antecedent_3h_mm = _overlap_mm(times, values, now - timedelta(hours=3), now)
        antecedent_24h_mm = _overlap_mm(times, values, now - timedelta(hours=24), now)

    # Current-rate decision (see module docstring).
    if ow_rate is not None and ow_rate > 0.0:
        rate = ow_rate
    elif om_rate is not None:
        rate = om_rate
    elif ow_rate is not None:
        rate = ow_rate
    else:
        rate = 0.0

    if series is None and (ow_rate is not None or om_rate is not None):
        # No hourly series: persist the current rate over the lead window.
        accumulation_mm = rate * (lead / 60.0)

    data_available = ow_rate is not None or series is not None

    return RainfallReport(
        rate_mm_hr=round(float(rate), 2),
        accumulation_mm=round(float(accumulation_mm), 2),
        antecedent_3h_mm=round(float(antecedent_3h_mm), 2),
        antecedent_24h_mm=round(float(antecedent_24h_mm), 2),
        source="+".join(sources) if sources else "none",
        data_available=data_available,
        observed_at=now.isoformat(),
    )
