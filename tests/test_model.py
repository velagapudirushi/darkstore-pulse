"""Unit tests for leak-free LightGBM stockout prediction & TreeSHAP explainability."""

import pandas as pd

from darkstore_pulse.config import SEED_PARQUET_PATH
from darkstore_pulse.model import explain_single_prediction, train_and_evaluate_model
from darkstore_pulse.stats_audit import run_statistical_audit
from darkstore_pulse.warehouse_sql import DarkStoreWarehouseSQL


def test_model_beats_naive_baseline_and_explains_single_row(tmp_path) -> None:
    sql = DarkStoreWarehouseSQL(SEED_PARQUET_PATH)
    feat_df = sql.build_ml_feature_table()
    metrics_file = tmp_path / "metrics.json"

    bundle = train_and_evaluate_model(feat_df, n_splits=3, save_metrics_path=metrics_file)
    assert metrics_file.exists()
    assert bundle.metrics["holdout_roc_auc"] > 0.85
    assert bundle.metrics["pr_auc_lift_over_naive_x"] > 3.0

    sample_row = feat_df.iloc[[100]]
    prob, breakdown = explain_single_prediction(bundle, sample_row)
    assert 0.0 <= prob <= 1.0
    assert len(breakdown) == 17
    assert "shap_impact" in breakdown.columns


def test_statistical_hypothesis_audit_runs_cleanly() -> None:
    raw_df = pd.read_parquet(SEED_PARQUET_PATH)
    results = run_statistical_audit(raw_df)
    assert set(results.keys()) == {"rain_stockout", "rain_eta", "zone_latenight_snacks"}
    assert results["rain_eta"].significant_at_0_01 is True
    assert results["zone_latenight_snacks"].relative_lift_x > 3.0
