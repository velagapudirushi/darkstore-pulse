"""Generate a realistic 14-day hourly dark-store dataset (53,760 rows) for Day-1 execution.

Simulates finite shelf inventory across 8 Bengaluru dark stores × 20 staple SKUs over 336 hours:
- Family residential zones deplete Dairy/Breakfast & Fresh Produce rapidly from 06:30–09:30 AM.
- IT-Hub zones deplete Late-Night Snacks/Drinks heavily from 20:30–01:30 AM (especially Fri/Sat).
- Rain storms increase order surges, drive Delivery ETA congestion upward BEFORE stock hits 0,
  and trigger ₹15–₹35 rain/surge fees.
- Scheduled supply trucks arrive at each store's morning & afternoon window, replenishing stock.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from darkstore_pulse.config import SEED_PARQUET_PATH, load_targets_config
from darkstore_pulse.schemas import validate_observations_df


def generate_14d_seed_dataset(
    days: int = 14,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Simulate 14 days of hourly observations and validate with Pandera."""
    rng = np.random.default_rng(seed)
    cfg = load_targets_config()
    pincodes = cfg["pincodes"]
    items = cfg["items"]

    start_ts = datetime(2026, 10, 24, 0, 0, 0)
    total_hours = days * 24

    # Generate realistic citywide weather timeline with localized rain bursts
    city_rain_base = np.zeros(total_hours)
    rain_windows = [
        (18, 22, 4.8),
        (41, 45, 6.2),
        (90, 95, 8.5),
        (138, 142, 5.4),
        (186, 191, 9.1),
        (233, 237, 4.2),
        (281, 286, 7.6),
        (306, 310, 5.9),
    ]
    for h_start, h_end, peak_mm in rain_windows:
        for h in range(h_start, min(h_end, total_hours)):
            city_rain_base[h] = peak_mm

    # Finite shelf capacity per SKU category
    base_capacity = {
        "Dairy_Breakfast": 24,
        "LateNight_Snacks": 22,
        "Fresh_Produce": 20,
        "Weather_Household": 18,
    }

    # Initialize shelf inventory per (pincode, item_id)
    inventory: dict[tuple[str, str], int] = {}
    for pin in pincodes:
        for item in items:
            cap = base_capacity[item["category"]]
            inventory[(pin["pincode"], item["item_id"])] = int(rng.integers(cap // 2, cap + 1))

    rows: list[dict] = []

    for step in range(total_hours):
        ts = start_ts + timedelta(hours=step)
        hour = ts.hour
        dow = ts.weekday()  # 0=Mon ... 6=Sun
        is_fri_sat_sun = dow in (4, 5, 6)

        # Diurnal Bengaluru temperature curve (19°C at 5 AM -> 29°C at 3 PM)
        diurnal_temp = 24.0 + 4.8 * np.sin((hour - 9) * np.pi / 12.0)

        for pin in pincodes:
            pin_code = str(pin["pincode"])
            zone = pin["zone_type"]
            morning_truck = int(pin["morning_truck_hour"])
            afternoon_truck = int(pin["afternoon_truck_hour"])

            # Localize rainfall & temperature slightly by neighborhood
            rain_mm = (
                round(max(0.0, city_rain_base[step] * rng.uniform(0.75, 1.25)), 1)
                if city_rain_base[step] > 0
                else 0.0
            )
            temp_c = round(diurnal_temp - (2.2 if rain_mm > 0 else 0.0) + rng.normal(0, 0.5), 1)

            # 1. Supply truck replenishment at store's morning & afternoon restock hours
            if hour == morning_truck or hour == afternoon_truck:
                for item in items:
                    if rng.random() < 0.93:  # 93% of SKUs replenished on each scheduled truck
                        cap = base_capacity[item["category"]]
                        inventory[(pin_code, item["item_id"])] = int(
                            rng.integers(int(cap * 0.85), cap + 4)
                        )

            # 2. Calculate storewide order load (drives Delivery ETA & surge fee)
            if zone == "IT_Hub":
                rush_load = (
                    1.35
                    if hour in (19, 20, 21, 22, 23, 0)
                    else (0.95 if hour in (8, 9, 13, 14) else 0.55)
                )
                if is_fri_sat_sun and hour in (20, 21, 22, 23, 0, 1):
                    rush_load += 0.55
            else:
                rush_load = (
                    1.40
                    if hour in (6, 7, 8, 9, 17, 18, 19, 20)
                    else (0.85 if hour in (10, 11, 12) else 0.40)
                )
                if is_fri_sat_sun and hour in (7, 8, 9, 18, 19):
                    rush_load += 0.35

            rain_load_boost = 0.85 * min(rain_mm / 4.0, 2.0) if rain_mm > 0 else 0.0
            total_store_load = rush_load + rain_load_boost + rng.normal(0, 0.08)

            surge_fee = (
                25.0
                if rain_mm >= 3.0
                else (15.0 if (rain_mm > 0.5 or total_store_load > 1.85) else 0.0)
            )

            # 3. Deplete each item's shelf inventory & emit hourly observation
            for item in items:
                iid = item["item_id"]
                cat = item["category"]
                current_units = inventory[(pin_code, iid)]

                # Category × Zone × Hour hourly demand rate (expected units sold)
                lam = 1.1 * max(0.25, total_store_load)
                if cat == "Dairy_Breakfast":
                    if hour in (6, 7, 8, 9):
                        lam *= 2.8 if zone == "Family_Residential" else 1.8
                    elif hour in (18, 19, 20):
                        lam *= 1.6
                elif cat == "LateNight_Snacks":
                    if hour in (20, 21, 22, 23, 0, 1):
                        lam *= 3.1 if zone == "IT_Hub" else 1.15
                        if is_fri_sat_sun and zone == "IT_Hub":
                            lam *= 1.35
                    elif hour in (6, 7, 8, 9, 10):
                        lam *= 0.45
                elif cat == "Fresh_Produce":
                    if hour in (7, 8, 9, 10, 17, 18, 19):
                        lam *= 2.2 if zone == "Family_Residential" else 1.5
                elif cat == "Weather_Household":
                    if iid == "umbrella_compact":
                        lam = 5.8 if rain_mm > 1.0 else 0.25
                    elif iid == "ginger_tea_250g" and rain_mm > 1.0:
                        lam *= 2.3
                    else:
                        lam *= 0.85

                sold = int(rng.poisson(max(0.05, lam)))
                new_units = max(0, current_units - sold)
                inventory[(pin_code, iid)] = new_units
                in_stock = new_units > 0

                # Realistic precursor signal: low remaining shelf stock + high store load pushes ETA up!
                scarcity_pressure = 3.2 if (0 < new_units <= 4) else (1.5 if (4 < new_units <= 8) else 0.0)
                eta_mins = int(
                    np.clip(
                        round(7.5 + 4.8 * total_store_load + scarcity_pressure + rng.normal(0, 1.1)),
                        6,
                        45,
                    )
                )

                # Occasional ₹1–₹4 dynamic promo discount during midday lull
                base_price = float(item["base_price_inr"])
                price_inr = (
                    round(base_price * 0.95, 2)
                    if (hour in (11, 12, 14, 15) and base_price >= 50 and rng.random() < 0.15)
                    else base_price
                )

                rows.append(
                    {
                        "timestamp": ts,
                        "pincode": pin_code,
                        "neighborhood": pin["neighborhood"],
                        "zone_type": zone,
                        "item_id": iid,
                        "item_name": item["item_name"],
                        "category": cat,
                        "price_inr": price_inr,
                        "in_stock": in_stock,
                        "eta_mins": eta_mins,
                        "surge_fee_inr": surge_fee,
                        "rain_mm": rain_mm,
                        "temp_c": temp_c,
                    }
                )

    df = validate_observations_df(pd.DataFrame(rows))
    out = Path(output_path or SEED_PARQUET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    generated = generate_14d_seed_dataset()
    oos_pct = (1.0 - generated["in_stock"].mean()) * 100
    print(
        f"Generated {len(generated):,} validated rows -> {SEED_PARQUET_PATH} "
        f"(Overall stockout rate: {oos_pct:.2f}%)"
    )
