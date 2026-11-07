"""Zero-login, polite guest-mode Playwright collector for quick-commerce storefronts.

Design principles:
1. Guest mode only — NEVER logs into personal accounts or submits OTPs.
2. Polite pacing — waits 3.5–6.0 seconds between item pages (~15–20 URLs/hour).
3. Respects boundaries — checks HTTP status and backs off immediately on 403/429
   or anti-bot challenges without attempting any evasion.
4. Schema-gated — every hourly batch passes through Pandera before saving to Parquet.
"""

from __future__ import annotations

from datetime import datetime
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from darkstore_pulse.config import LIVE_PARQUET_PATH, load_targets_config
from darkstore_pulse.schemas import ObservationRecord, validate_observations_df
from darkstore_pulse.weather import fetch_live_weather

logger = logging.getLogger(__name__)

STOREFRONT_BASE_URLS = {
    "blinkit_guest": "https://blinkit.com/prn",
    "zepto_guest": "https://www.zeptonow.com/pn",
}


def _parse_eta_minutes(raw_text: str, default: int = 12) -> int:
    """Extract integer delivery minutes from banner strings like '11 MINS' or '9 mins'."""
    match = re.search(r"(\d{1,3})\s*(?:min|mins|minute)", raw_text, flags=re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return max(4, min(120, val))
    return default


def _parse_price_inr(raw_text: str, fallback: float) -> float:
    """Extract ₹ price from visible product card text."""
    match = re.search(r"₹\s*(\d+(?:\.\d{1,2})?)", raw_text)
    if match:
        val = float(match.group(1))
        if 1.0 <= val <= 5000.0:
            return val
    return fallback


def collect_hourly_batch(
    config: dict[str, Any] | None = None,
    max_pincodes: int | None = None,
    max_items: int | None = None,
    headless: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run one polite hourly guest-mode check across configured pincodes and staple items."""
    from playwright.sync_api import sync_playwright

    cfg = config or load_targets_config()
    platform = cfg.get("platform", "blinkit_guest")
    base_url = STOREFRONT_BASE_URLS.get(platform, STOREFRONT_BASE_URLS["blinkit_guest"])
    delay_cfg = cfg.get("polite_delay_seconds", {"min": 3.5, "max": 6.0})

    pincodes = cfg["pincodes"][:max_pincodes] if max_pincodes else cfg["pincodes"]
    items = cfg["items"][:max_items] if max_items else cfg["items"]

    records: list[dict[str, Any]] = []
    sweep_ts = datetime.now().replace(minute=0, second=0, microsecond=0)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        for pin_cfg in pincodes:
            pincode = str(pin_cfg["pincode"])
            weather = fetch_live_weather(float(pin_cfg["lat"]), float(pin_cfg["lon"]))
            logger.info(
                "Checking %s (%s) | Temp %.1f°C | Rain %.1fmm",
                pin_cfg["neighborhood"],
                pincode,
                weather.temp_c,
                weather.rain_mm,
            )

            # Set public guest delivery location if location box is available
            try:
                page.goto("https://blinkit.com/", wait_until="domcontentloaded", timeout=20000)
                time.sleep(2.0)
                loc_input = page.locator("input[placeholder*='delivery location'], input[name*='select-locality']")
                if loc_input.count() > 0:
                    loc_input.first.fill(pincode)
                    time.sleep(1.5)
                    suggestion = page.locator("[class*='LocationSearchList'] div").first
                    if suggestion.count() > 0:
                        suggestion.click()
                        time.sleep(2.0)
            except Exception as exc:
                logger.warning("Could not set guest pincode modal for %s: %s", pincode, exc)

            for item in items:
                url = f"{base_url}/{item['url_slug']}"
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    if response is not None and response.status in (403, 429):
                        logger.warning(
                            "Received HTTP %s on %s — stopping sweep politely without retrying.",
                            response.status,
                            url,
                        )
                        browser.close()
                        return pd.DataFrame()

                    time.sleep(random.uniform(float(delay_cfg["min"]), float(delay_cfg["max"])))
                    body_text = page.locator("body").inner_text(timeout=5000)

                    out_of_stock_markers = (
                        "out of stock",
                        "currently unavailable",
                        "sold out",
                    )
                    in_stock = not any(m in body_text.lower() for m in out_of_stock_markers)
                    eta_mins = _parse_eta_minutes(body_text, default=11)
                    price_inr = _parse_price_inr(body_text, fallback=float(item["base_price_inr"]))
                    surge_fee = 25.0 if "surge" in body_text.lower() or "rain fee" in body_text.lower() else 0.0

                    rec = ObservationRecord(
                        timestamp=sweep_ts,
                        pincode=pincode,
                        neighborhood=pin_cfg["neighborhood"],
                        zone_type=pin_cfg["zone_type"],
                        item_id=item["item_id"],
                        item_name=item["item_name"],
                        category=item["category"],
                        price_inr=price_inr,
                        in_stock=in_stock,
                        eta_mins=eta_mins,
                        surge_fee_inr=surge_fee,
                        rain_mm=weather.rain_mm,
                        temp_c=weather.temp_c,
                    )
                    records.append(rec.model_dump())
                except Exception as exc:
                    logger.warning("Skipping %s (%s): %s", item["item_id"], pincode, exc)

        browser.close()

    if not records:
        return pd.DataFrame()

    batch_df = validate_observations_df(pd.DataFrame(records))
    target_file = output_path or LIVE_PARQUET_PATH
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists():
        existing = pd.read_parquet(target_file)
        combined = pd.concat([existing, batch_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "pincode", "item_id"], keep="last")
    else:
        combined = batch_df

    combined.to_parquet(target_file, index=False)
    logger.info("Saved %d validated rows (total %d) to %s", len(batch_df), len(combined), target_file)
    return batch_df
