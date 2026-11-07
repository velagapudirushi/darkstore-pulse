"""Free live weather fetcher using Open-Meteo (zero API key required)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherSnapshot:
    """Current neighborhood weather joined onto each hourly storefront check."""

    temp_c: float
    rain_mm: float


def fetch_live_weather(lat: float, lon: float, timeout_sec: float = 8.0) -> WeatherSnapshot:
    """Query Open-Meteo for current temperature (°C) and precipitation/rain (mm).

    Falls back cleanly to mild dry Bengaluru baseline if offline so hourly collection
    never crashes due to transient network blips.
    """
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": "temperature_2m,rain",
                "timezone": "auto",
            },
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})
        return WeatherSnapshot(
            temp_c=float(current.get("temperature_2m", 25.0)),
            rain_mm=max(0.0, float(current.get("rain", 0.0))),
        )
    except Exception as exc:
        logger.warning("Open-Meteo unavailable (%s); using neutral dry fallback.", exc)
        return WeatherSnapshot(temp_c=25.0, rain_mm=0.0)
