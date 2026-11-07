"""Data contracts and Pandera schema validation for dark-store observations.

Blocks corrupt scrapes (e.g., ₹0 prices from broken DOM selectors, impossible ETAs,
or malformed pincodes) before they enter DuckDB / Parquet storage.
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, field_validator

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera versions
    import pandera as pa


VALID_ZONE_TYPES = {"IT_Hub", "Family_Residential"}
VALID_CATEGORIES = {
    "Dairy_Breakfast",
    "LateNight_Snacks",
    "Fresh_Produce",
    "Weather_Household",
}


class ObservationRecord(BaseModel):
    """Single item-pincode observation captured during an hourly sweep."""

    timestamp: datetime
    pincode: str = Field(..., pattern=r"^\d{6}$")
    neighborhood: str = Field(..., min_length=2)
    zone_type: str
    item_id: str = Field(..., min_length=2)
    item_name: str = Field(..., min_length=2)
    category: str
    price_inr: float = Field(..., gt=0.0, le=5000.0)
    in_stock: bool
    eta_mins: int = Field(..., ge=4, le=120)
    surge_fee_inr: float = Field(..., ge=0.0, le=150.0)
    rain_mm: float = Field(..., ge=0.0, le=200.0)
    temp_c: float = Field(..., ge=5.0, le=50.0)

    @field_validator("zone_type")
    @classmethod
    def check_zone_type(cls, v: str) -> str:
        if v not in VALID_ZONE_TYPES:
            raise ValueError(f"zone_type must be one of {VALID_ZONE_TYPES}, got {v!r}")
        return v

    @field_validator("category")
    @classmethod
    def check_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}, got {v!r}")
        return v


OBSERVATION_DF_SCHEMA = pa.DataFrameSchema(
    columns={
        "timestamp": pa.Column(pa.DateTime, nullable=False),
        "pincode": pa.Column(str, pa.Check.str_matches(r"^\d{6}$"), nullable=False),
        "neighborhood": pa.Column(str, nullable=False),
        "zone_type": pa.Column(str, pa.Check.isin(VALID_ZONE_TYPES), nullable=False),
        "item_id": pa.Column(str, nullable=False),
        "item_name": pa.Column(str, nullable=False),
        "category": pa.Column(str, pa.Check.isin(VALID_CATEGORIES), nullable=False),
        "price_inr": pa.Column(float, [pa.Check.gt(0.0), pa.Check.le(5000.0)], nullable=False),
        "in_stock": pa.Column(bool, nullable=False),
        "eta_mins": pa.Column(int, [pa.Check.ge(4), pa.Check.le(120)], coerce=True, nullable=False),
        "surge_fee_inr": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(150.0)], coerce=True, nullable=False),
        "rain_mm": pa.Column(float, [pa.Check.ge(0.0), pa.Check.le(200.0)], coerce=True, nullable=False),
        "temp_c": pa.Column(float, [pa.Check.ge(5.0), pa.Check.le(50.0)], coerce=True, nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_observations_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an observation DataFrame against strict Pandera bounds."""
    df_checked = df.copy()
    df_checked["timestamp"] = pd.to_datetime(df_checked["timestamp"])
    return OBSERVATION_DF_SCHEMA.validate(df_checked, lazy=True)
