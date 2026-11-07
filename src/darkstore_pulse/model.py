"""Chronological LightGBM 2-Hour Stockout Predictor + SHAP TreeExplainer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from darkstore_pulse.config import ARTIFACTS_DIR

CATEGORICAL_COLS = ["zone_type", "category", "pincode", "item_id"]
NUMERIC_COLS = [
    "price_inr",
    "eta_mins",
    "eta_delta_1h",
    "eta_rolling_3h",
    "warehouse_oos_share_now",
    "hours_since_last_restock",
    "item_hist_oos_rate",
    "surge_fee_inr",
    "rain_mm",
    "temp_c",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "will_stockout_next_2h"

HUMAN_FEATURE_LABELS = {
    "eta_mins": "Current Delivery ETA (mins)",
    "eta_delta_1h": "1-Hour ETA Jump (Congestion Signal)",
    "eta_rolling_3h": "3-Hour Rolling Average ETA",
    "warehouse_oos_share_now": "Storewide Out-of-Stock Share Right Now",
    "hours_since_last_restock": "Hours Elapsed Since Last Supply Truck",
    "item_hist_oos_rate": "Item Past 48h Stockout Rate",
    "rain_mm": "Live Rainfall (mm/hr)",
    "surge_fee_inr": "Active Surge / Rain Fee (₹)",
    "temp_c": "Ambient Temperature (°C)",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Flag",
    "zone_type": "Neighborhood Archetype (IT Hub vs Family)",
    "category": "Product Category",
    "pincode": "Dark-Store Pincode",
    "item_id": "Specific SKU",
    "price_inr": "Item Price (₹)",
}


@dataclass
class TrainedStockoutBundle:
    """In-memory trained LightGBM model, category levels, metrics, and SHAP explainer."""

    model: lgb.LGBMClassifier
    explainer: shap.TreeExplainer
    category_levels: dict[str, list[str]]
    metrics: dict[str, Any]
    global_shap_importance: pd.DataFrame


def _prepare_features(
    df: pd.DataFrame,
    category_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Encode categorical columns consistently between train and live inference."""
    X = df[FEATURE_COLS].copy()
    levels_out: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLS:
        if category_levels and col in category_levels:
            cats = category_levels[col]
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)
            levels_out[col] = cats
        else:
            cats = sorted(X[col].astype(str).unique().tolist())
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)
            levels_out[col] = cats
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X, levels_out


def train_and_evaluate_model(
    feature_df: pd.DataFrame,
    n_splits: int = 4,
    save_metrics_path: Path | None = None,
) -> TrainedStockoutBundle:
    """Train LightGBM using strict chronological splits (zero future leakage)."""
    ordered = feature_df.sort_values("timestamp").reset_index(drop=True)
    X_all, category_levels = _prepare_features(ordered)
    y_all = ordered[TARGET_COL].astype(int).to_numpy()

    # 1. Cross-validate strictly forward in time via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_roc_aucs: list[float] = []
    cv_pr_aucs: list[float] = []

    params = {
        "n_estimators": 240,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1,
    }

    for train_idx, val_idx in tscv.split(X_all):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_all.iloc[train_idx], y_all[train_idx])
        probs = m.predict_proba(X_all.iloc[val_idx])[:, 1]
        cv_roc_aucs.append(float(roc_auc_score(y_all[val_idx], probs)))
        cv_pr_aucs.append(float(average_precision_score(y_all[val_idx], probs)))

    # 2. Final chronological holdout split: First 75% days train -> Final 25% days test
    split_point = int(len(ordered) * 0.75)
    X_train, X_test = X_all.iloc[:split_point], X_all.iloc[split_point:]
    y_train, y_test = y_all[:split_point], y_all[split_point:]

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_train, y_train)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    base_rate = float(y_test.mean())

    # Naive baseline: item's past 48h stockout rate
    naive_probs = np.clip(X_test["item_hist_oos_rate"].to_numpy(), 0.001, 0.999)

    holdout_roc_auc = float(roc_auc_score(y_test, test_probs))
    holdout_pr_auc = float(average_precision_score(y_test, test_probs))
    naive_roc_auc = float(roc_auc_score(y_test, naive_probs))
    naive_pr_auc = float(average_precision_score(y_test, naive_probs))
    brier = float(brier_score_loss(y_test, test_probs))

    # 3. Compute exact TreeSHAP explanations on chronological test sample
    explainer = shap.TreeExplainer(final_model)
    sample_X = X_test.sample(n=min(1200, len(X_test)), random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        shap_vals = explainer.shap_values(sample_X)
    if isinstance(shap_vals, list):
        shap_matrix = np.asarray(shap_vals[1])
    else:
        shap_matrix = np.asarray(shap_vals)

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    shap_df = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "label": [HUMAN_FEATURE_LABELS[c] for c in FEATURE_COLS],
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    metrics: dict[str, Any] = {
        "total_in_stock_rows_evaluated": int(len(ordered)),
        "train_rows": int(len(X_train)),
        "chronological_test_rows": int(len(X_test)),
        "positive_2h_stockout_base_rate_pct": round(base_rate * 100, 2),
        "cv_mean_roc_auc": round(float(np.mean(cv_roc_aucs)), 4),
        "cv_mean_pr_auc": round(float(np.mean(cv_pr_aucs)), 4),
        "holdout_roc_auc": round(holdout_roc_auc, 4),
        "naive_baseline_roc_auc": round(naive_roc_auc, 4),
        "holdout_pr_auc": round(holdout_pr_auc, 4),
        "naive_baseline_pr_auc": round(naive_pr_auc, 4),
        "pr_auc_lift_over_naive_x": round(holdout_pr_auc / max(naive_pr_auc, 1e-6), 2),
        "holdout_brier_score": round(brier, 4),
        "top_5_shap_features": shap_df.head(5)[["feature", "label", "mean_abs_shap"]].to_dict(
            orient="records"
        ),
    }

    target_metrics_path = save_metrics_path or (ARTIFACTS_DIR / "model_metrics.json")
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainedStockoutBundle(
        model=final_model,
        explainer=explainer,
        category_levels=category_levels,
        metrics=metrics,
        global_shap_importance=shap_df,
    )


def explain_single_prediction(
    bundle: TrainedStockoutBundle,
    single_row_df: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Predict 2-hour stockout probability and return per-feature SHAP drivers for 1 scenario."""
    X_single, _ = _prepare_features(single_row_df, category_levels=bundle.category_levels)
    prob = float(bundle.model.predict_proba(X_single)[0, 1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        raw_shap = bundle.explainer.shap_values(X_single)
    if isinstance(raw_shap, list):
        vec = np.asarray(raw_shap[1])[0]
    else:
        vec = np.asarray(raw_shap)[0]

    breakdown = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "driver": [HUMAN_FEATURE_LABELS[c] for c in FEATURE_COLS],
                "value": [str(single_row_df.iloc[0][c]) for c in FEATURE_COLS],
                "shap_impact": vec,
            }
        )
        .assign(abs_impact=lambda d: d["shap_impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .drop(columns=["abs_impact"])
        .reset_index(drop=True)
    )
    return prob, breakdown
