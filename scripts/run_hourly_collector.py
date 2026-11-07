"""CLI entry point for running one polite hourly guest-mode storefront sweep."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from darkstore_pulse.collector import collect_hourly_batch
from darkstore_pulse.config import load_targets_config
from darkstore_pulse.weather import fetch_live_weather


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one zero-login, polite hourly dark-store inventory & ETA check."
    )
    parser.add_argument(
        "--max-pincodes",
        type=int,
        default=None,
        help="Optional limit on number of pincodes to check (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit on number of items per pincode.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show browser window visibly instead of running silently in headless background mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify config & Open-Meteo weather connectivity without launching Playwright.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = load_targets_config()
    if args.dry_run:
        first_pin = cfg["pincodes"][0]
        weather = fetch_live_weather(float(first_pin["lat"]), float(first_pin["lon"]))
        print(
            f"[DRY RUN OK] Loaded {len(cfg['pincodes'])} pincodes x {len(cfg['items'])} items. "
            f"Sample weather for {first_pin['neighborhood']} ({first_pin['pincode']}): "
            f"{weather.temp_c}°C, rain={weather.rain_mm}mm"
        )
        return

    df = collect_hourly_batch(
        config=cfg,
        max_pincodes=args.max_pincodes,
        max_items=args.max_items,
        headless=not args.headed,
    )
    print(f"Completed hourly sweep: {len(df)} validated rows captured.")


if __name__ == "__main__":
    main()
