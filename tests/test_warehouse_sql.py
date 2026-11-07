"""Unit tests for DuckDB SQL restock-truck inference and SQL safety guardrails."""

import pytest

from darkstore_pulse.config import SEED_PARQUET_PATH, load_targets_config
from darkstore_pulse.warehouse_sql import DarkStoreWarehouseSQL


def test_sql_lag_recovers_ground_truth_restock_hours() -> None:
    cfg = load_targets_config()
    expected = {
        str(p["pincode"]): (int(p["morning_truck_hour"]), int(p["afternoon_truck_hour"]))
        for p in cfg["pincodes"]
    }

    sql = DarkStoreWarehouseSQL(SEED_PARQUET_PATH)
    summary = sql.get_top_restock_windows_summary()
    assert len(summary) == len(expected)

    for row in summary.itertuples():
        exp_morn, exp_aft = expected[str(row.pincode)]
        assert int(row.inferred_morning_truck_hr) == exp_morn
        assert int(row.inferred_afternoon_truck_hr) == exp_aft


def test_sql_explorer_blocks_destructive_queries() -> None:
    sql = DarkStoreWarehouseSQL(SEED_PARQUET_PATH)
    with pytest.raises(ValueError, match="read-only"):
        sql.run_custom_query("DROP VIEW obs;")


def test_ml_feature_table_filters_only_currently_in_stock() -> None:
    sql = DarkStoreWarehouseSQL(SEED_PARQUET_PATH)
    feat_df = sql.build_ml_feature_table()
    assert len(feat_df) > 30_000
    assert set(feat_df["will_stockout_next_2h"].unique()).issubset({0, 1})
    assert feat_df["eta_delta_1h"].isna().sum() == 0
