"""Unit tests for Pydantic & Pandera observation validation gates."""

from datetime import datetime
import pandas as pd
import pytest

from darkstore_pulse.schemas import ObservationRecord, validate_observations_df


def _valid_row() -> dict:
    return {
        "timestamp": datetime(2026, 11, 1, 20, 0, 0),
        "pincode": "560038",
        "neighborhood": "Indiranagar",
        "zone_type": "IT_Hub",
        "item_id": "milk_amul_500ml",
        "item_name": "Amul Taaza Toned Milk (500ml)",
        "category": "Dairy_Breakfast",
        "price_inr": 28.0,
        "in_stock": True,
        "eta_mins": 12,
        "surge_fee_inr": 15.0,
        "rain_mm": 2.4,
        "temp_c": 23.1,
    }


def test_valid_observation_passes_pydantic_and_pandera() -> None:
    row = _valid_row()
    rec = ObservationRecord(**row)
    assert rec.pincode == "560038"

    df = pd.DataFrame([row])
    validated = validate_observations_df(df)
    assert len(validated) == 1
    assert bool(validated.iloc[0]["in_stock"]) is True


def test_corrupt_zero_price_rejected() -> None:
    bad = _valid_row()
    bad["price_inr"] = 0.0
    with pytest.raises(Exception):
        validate_observations_df(pd.DataFrame([bad]))


def test_malformed_pincode_and_impossible_eta_rejected() -> None:
    bad = _valid_row()
    bad["pincode"] = "5600"  # Invalid 4-digit pincode
    bad["eta_mins"] = 450    # Impossible ETA
    with pytest.raises(Exception):
        ObservationRecord(**bad)
