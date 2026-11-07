"""Interactive Streamlit dashboard for DarkStore Pulse."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from darkstore_pulse.config import LIVE_PARQUET_PATH, SEED_PARQUET_PATH, load_targets_config
from darkstore_pulse.model import explain_single_prediction, train_and_evaluate_model
from darkstore_pulse.stats_audit import run_statistical_audit
from darkstore_pulse.warehouse_sql import DarkStoreWarehouseSQL

st.set_page_config(
    page_title="DarkStore Pulse | Quick-Commerce Forensics",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner="Training chronological LightGBM + TreeSHAP explainer...")
def load_pipeline_bundle(parquet_str: str):
    sql_engine = DarkStoreWarehouseSQL(Path(parquet_str))
    raw_df = pd.read_parquet(parquet_str)
    feat_df = sql_engine.build_ml_feature_table()
    bundle = train_and_evaluate_model(feat_df)
    stats_results = run_statistical_audit(raw_df)
    restock_summary = sql_engine.get_top_restock_windows_summary()
    restock_all = sql_engine.infer_restock_schedules()
    return raw_df, feat_df, bundle, stats_results, restock_summary, restock_all


def main() -> None:
    st.title("⚡ DarkStore Pulse: Reverse-Engineering Quick-Commerce Warehouses")
    st.caption(
        "Uncovering hidden supply-truck schedules, neighborhood demand splits, and 2-hour-ahead "
        "stockout probabilities from public storefront signals + live rainfall."
    )

    with st.sidebar:
        st.header("Data Source")
        has_live = LIVE_PARQUET_PATH.exists()
        source_choice = st.radio(
            "Active Dataset",
            options=(
                ["14-Day Seed Dataset (53,760 rows)", "Live Collected Parquet"]
                if has_live
                else ["14-Day Seed Dataset (53,760 rows)"]
            ),
            index=0,
        )
        active_path = (
            LIVE_PARQUET_PATH
            if source_choice == "Live Collected Parquet"
            else SEED_PARQUET_PATH
        )
        cfg = load_targets_config()
        st.divider()
        st.markdown("### Pipeline Snapshot")
        st.markdown(
            f"- **City:** `{cfg['city']}`\n"
            f"- **Tracked Dark Stores:** `{len(cfg['pincodes'])}` pincodes\n"
            f"- **Tracked Staples:** `{len(cfg['items'])}` SKUs\n"
            "- **Validation:** `Pandera` strict schema\n"
            "- **Warehouse Engine:** `DuckDB` SQL\n"
            "- **Predictor:** `LightGBM` + `TreeSHAP`"
        )

    raw_df, feat_df, bundle, stats_results, restock_summary, restock_all = load_pipeline_bundle(
        str(active_path)
    )
    metrics = bundle.metrics

    # Top KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Validated Hourly Checks", f"{len(raw_df):,}")
    k2.metric(
        "2-Hr Stockout ROC-AUC (Holdout)",
        f"{metrics['holdout_roc_auc']:.3f}",
        delta=f"+{metrics['holdout_roc_auc'] - metrics['naive_baseline_roc_auc']:.3f} vs Naive",
    )
    k3.metric(
        "PR-AUC Lift Over Baseline",
        f"{metrics['pr_auc_lift_over_naive_x']}x",
        delta=f"PR-AUC {metrics['holdout_pr_auc']:.3f}",
    )
    k4.metric(
        "Rain Stockout Multiplier",
        f"{stats_results['rain_stockout'].relative_lift_x}x",
        delta="p < 0.0001 (Chi-Square)",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 2-Hour Stockout Risk Predictor (SHAP)",
            "🚚 Secret Supply-Truck Clock (SQL LAG)",
            "🌧️ Neighborhood & Rain Forensics",
            "🧮 Live DuckDB SQL Playground",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Interactive Stockout Risk Simulator + Local SHAP Waterfall
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Will Your Item Sell Out in the Next 2 Hours?")
        st.markdown(
            "Adjust live conditions below to see how **rising Delivery ETA**, **rainfall**, "
            "and **hours since the last supply truck** shift stockout probability."
        )

        c_left, c_right = st.columns([1, 1.45])
        with c_left:
            pin_options = {
                f"{p['neighborhood']} ({p['pincode']}) — {p['zone_type']}": p
                for p in cfg["pincodes"]
            }
            selected_pin_label = st.selectbox("1. Neighborhood Dark Store", list(pin_options.keys()))
            pin_obj = pin_options[selected_pin_label]

            item_options = {f"{i['item_name']} (₹{i['base_price_inr']:.0f})": i for i in cfg["items"]}
            selected_item_label = st.selectbox("2. Everyday Item (Currently In Stock)", list(item_options.keys()))
            item_obj = item_options[selected_item_label]

            hour_val = st.slider("3. Current Hour of Day (24h)", 0, 23, 20)
            is_wknd = st.toggle("Weekend (Fri–Sun Evening Surge)", value=True)
            rain_val = st.slider("4. Live Rainfall (mm/hr from Open-Meteo)", 0.0, 12.0, 4.5, 0.5)
            eta_now = st.slider("5. Quoted Delivery ETA Right Now (mins)", 6, 38, 19)
            eta_jump = st.slider("6. ETA Jump Over Past 1 Hour (mins)", -5, 15, 6)
            hrs_since_truck = st.slider("7. Hours Since Last Supply Truck", 0, 14, 6)
            store_oos_pct = st.slider("8. Other SKUs Already Out of Stock in Store (%)", 0, 60, 25)

        scenario_row = pd.DataFrame(
            [
                {
                    "zone_type": pin_obj["zone_type"],
                    "category": item_obj["category"],
                    "pincode": str(pin_obj["pincode"]),
                    "item_id": item_obj["item_id"],
                    "price_inr": float(item_obj["base_price_inr"]),
                    "eta_mins": eta_now,
                    "eta_delta_1h": eta_jump,
                    "eta_rolling_3h": max(6.0, eta_now - eta_jump * 0.4),
                    "warehouse_oos_share_now": store_oos_pct / 100.0,
                    "hours_since_last_restock": hrs_since_truck,
                    "item_hist_oos_rate": 0.18,
                    "surge_fee_inr": 25.0 if rain_val >= 3.0 else (15.0 if rain_val > 0.5 else 0.0),
                    "rain_mm": rain_val,
                    "temp_c": 22.5 if rain_val > 0 else 26.0,
                    "hour_of_day": hour_val,
                    "day_of_week": 5 if is_wknd else 2,
                    "is_weekend": int(is_wknd),
                }
            ]
        )

        prob, shap_breakdown = explain_single_prediction(bundle, scenario_row)

        with c_right:
            risk_pct = prob * 100
            if risk_pct >= 60:
                badge = "🔴 HIGH STOCKOUT RISK — Order Immediately"
            elif risk_pct >= 30:
                badge = "🟠 MODERATE STOCKOUT RISK — Stock Depleting"
            else:
                badge = "🟢 LOW RISK — Shelf Well Stocked"

            st.markdown(f"### {badge}")
            st.metric("Predicted 2-Hour Sellout Probability", f"{risk_pct:.1f}%")

            top_drivers = shap_breakdown.head(8).iloc[::-1]
            colors = ["#ef4444" if v > 0 else "#10b981" for v in top_drivers["shap_impact"]]
            fig_shap = go.Figure(
                go.Bar(
                    x=top_drivers["shap_impact"],
                    y=[f"{r.driver} = {r.value}" for r in top_drivers.itertuples()],
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig_shap.update_layout(
                title="Exact TreeSHAP Feature Contributions (Red = Pushes Toward Stockout)",
                xaxis_title="SHAP Log-Odds Impact",
                height=380,
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: Reverse-Engineered Warehouse Restock Schedule
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Reverse-Engineering Hidden Dark-Store Supply Trucks")
        st.markdown(
            "Using `DuckDB` SQL `LAG(in_stock)`, every transition from **`Out of Stock` → `In Stock`** "
            "reveals when a replenishment truck unloaded inventory at each neighborhood warehouse."
        )
        st.dataframe(restock_summary, use_container_width=True, hide_index=True)

        fig_trucks = px.bar(
            restock_all,
            x="hour_of_day",
            y="restock_flip_count",
            color="neighborhood",
            barmode="group",
            labels={
                "hour_of_day": "Hour of Day (0–23)",
                "restock_flip_count": "Observed Out-of-Stock → In-Stock Flips",
            },
            title="Detected Replenishment Spikes Across All 8 Bengaluru Pincodes",
        )
        st.plotly_chart(fig_trucks, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: Statistical Hypothesis Tests & Behavioral Heatmaps
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Rigorous Statistical Proof (`SciPy` Hypothesis Audit)")
        for res in stats_results.values():
            st.info(f"**{res.test_name}** — {res.plain_english_verdict}")

        sql_engine = DarkStoreWarehouseSQL(active_path)
        col_a, col_b = st.columns(2)
        with col_a:
            it_heat = sql_engine.get_hourly_heatmap("IT_Hub")
            piv_it = it_heat.pivot(index="category", columns="hour_of_day", values="stockout_rate_pct")
            fig_it = px.imshow(
                piv_it,
                aspect="auto",
                color_continuous_scale="Reds",
                title="IT-Hub Pincodes: Stockout % by Hour & Category",
            )
            st.plotly_chart(fig_it, use_container_width=True)

        with col_b:
            fam_heat = sql_engine.get_hourly_heatmap("Family_Residential")
            piv_fam = fam_heat.pivot(index="category", columns="hour_of_day", values="stockout_rate_pct")
            fig_fam = px.imshow(
                piv_fam,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Family Residential Pincodes: Stockout % by Hour & Category",
            )
            st.plotly_chart(fig_fam, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: Live DuckDB SQL Explorer
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Query the Parquet Warehouse Directly With `DuckDB` SQL")
        default_sql = """SELECT
    neighborhood,
    zone_type,
    COUNT(*) AS observations,
    ROUND(100.0 * AVG(CASE WHEN NOT in_stock THEN 1.0 ELSE 0.0 END), 2) AS stockout_pct,
    ROUND(AVG(eta_mins), 1) AS avg_eta_mins
FROM obs
GROUP BY 1, 2
ORDER BY stockout_pct DESC;"""
        user_sql = st.text_area("Analytical SQL (`obs` view)", value=default_sql, height=180)
        if st.button("Run DuckDB Query"):
            try:
                out_df = DarkStoreWarehouseSQL(active_path).run_custom_query(user_sql)
                st.dataframe(out_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Query rejected: {exc}")


if __name__ == "__main__":
    main()
