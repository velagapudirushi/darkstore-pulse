# ⚡ DarkStore Pulse: Reverse-Engineering Quick-Commerce Warehouses

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?owner=velagapudirushi&repo=darkstore-pulse&branch=main&mainModule=app/streamlit_app.py)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DuckDB SQL](https://img.shields.io/badge/SQL-DuckDB%202.0-FFF000.svg)](https://duckdb.org/)
[![LightGBM + SHAP](https://img.shields.io/badge/ML-LightGBM%20%2B%20TreeSHAP-10b981.svg)](https://lightgbm.readthedocs.io/)
[![Pandera Validated](https://img.shields.io/badge/Data%20Quality-Pandera-purple.svg)](https://pandera.readthedocs.io/)
[![Tests: 9 Passing](https://img.shields.io/badge/pytest-9%20passed-brightgreen.svg)](tests/)

**DarkStore Pulse** is an end-to-end operational Data Science system inspired by viral reverse-engineering audits like [`McBroken`](https://mcbroken.com/). Instead of training another model on a static Kaggle CSV, this project tracks **public guest-mode quick-commerce storefronts** (Blinkit / Zepto—zero login required) joined with **live neighborhood rainfall** (`Open-Meteo`) across **8 Bengaluru pincodes × 20 everyday staples**.

Using `DuckDB` analytical SQL window functions, `SciPy` hypothesis tests, and a leak-free chronological `LightGBM` + `TreeSHAP` pipeline, it uncovers **hidden supply-truck arrival schedules** and predicts **whether an in-stock item will sell out within the next 2 hours (`0.928` ROC-AUC, `7.25x` PR-AUC lift over naive baselines)**.

---

## 🐐 Explain It So Simple Even a Goat Can Understand It

Imagine there is a tiny, windowless grocery warehouse (**a "dark store"**) hidden down the street in your neighborhood. Nobody outside the company is allowed inside. You have no idea how many cartons of **Milk**, **Eggs**, or **Maggi** are sitting on its shelves, or what time their delivery truck comes to refill them.

**So how do we figure out everything happening inside that locked building—without ever stepping foot in it or logging into an account?**

We watch **three public clues** on the store's public website once every hour:

1. **Clue #1 — The "Magic Refill" (`Out of Stock` → `In Stock`):**
   * At **9:00 PM**, Eggs say `Out of Stock`.
   * At **2:00 AM**, Eggs still say `Out of Stock`.
   * At **6:00 AM sharp**, Eggs, Milk, and Bread all flip back to `In Stock` at the exact same hour!
   * **What we just discovered:** Nobody magically laid eggs on the shelf—**the morning supply truck arrives at 6:00 AM in that neighborhood!** Using a simple SQL `LAG()` check across days, we map every neighborhood warehouse's secret morning and afternoon truck schedule.

2. **Clue #2 — The "Traffic Jam Warning" (Rising Delivery Minutes *Before* Sellout):**
   * Normally, the app promises **"Delivery in 9 mins."**
   * At 7:30 PM on a rainy Friday, it creeps up to **14 mins**, then **21 mins**. Milk *still* says `In Stock`—for now!
   * Why did the minutes go up? Because orders are piling up inside that warehouse faster than packers can bag them! **Rising delivery minutes is an early-warning smoke alarm that shelves are about to hit zero.**

3. **Clue #3 — Neighborhood Personalities:**
   * **Family neighborhoods** (Jayanagar, Malleshwaram) wipe out Milk, Curd, Bread, and Coriander between **6:30 AM and 9:00 AM**.
   * **IT-Hub / Bachelor neighborhoods** (Indiranagar, Koramangala, HSR, Bellandur) wipe out Coke Zero, Ice Cubes, Chips, and Maggi between **9:00 PM and 2:00 AM** (`6.88x` higher late-night stockouts!).

Put those clues into a **Machine Learning model (`LightGBM`)**, and our app tells you:
> *"Amul Milk is In Stock right now in Indiranagar, **but there is an 81% chance it sells out within 2 hours** because rainfall is 4.5 mm/hr, Delivery ETA jumped +6 mins in the last hour, and it has been 6 hours since the 2:00 PM supply truck."*

---

## 🏗️ System Architecture & Repository Structure

```mermaid
flowchart LR
    subgraph Ingestion["1. Polite Hourly Ingestion (Zero Login)"]
        A["Public Guest Storefront (Playwright)"]
        B["Open-Meteo Weather API (Free, No Key)"]
    end

    subgraph Quality["2. Data Quality Gate"]
        C["Pydantic + Pandera Schema Check"]
    end

    subgraph Storage["3. Analytical Warehouse"]
        D["Parquet Lake + DuckDB SQL Engine"]
    end

    subgraph Intelligence["4. Data Science & ML Core"]
        E["SQL LAG() Truck Forensics"]
        F["SciPy Hypothesis Audit"]
        G["LightGBM + TreeSHAP (TimeSeriesSplit)"]
    end

    subgraph App["5. Interactive Product"]
        H["4-Tab Streamlit Dashboard"]
    end

    A --> C
    B --> C
    C -->|"Blocks ₹0 / Corrupt Rows"| D
    D --> E
    D --> F
    D --> G
    E --> H
    F --> H
    G --> H
```

### Clean Modular Folder Layout (Zero Messy Root Notebooks)

```text
darkstore-pulse/
├── .github/workflows/ci.yml           # Automated pytest runner on every push/PR
├── configs/
│   └── targets.yaml                   # 8 Bengaluru pincodes (IT Hub vs Family) & 20 staple SKUs
├── data/
│   ├── processed/
│   │   └── darkstore_14d_seed.parquet # 53,760-row 14-day dataset so the entire app runs Day 1
│   └── raw/                           # Live hourly Playwright Parquet observations land here
├── src/darkstore_pulse/
│   ├── config.py                      # Canonical paths & YAML config loader
│   ├── schemas.py                     # Pydantic + Pandera validation contracts
│   ├── weather.py                     # Zero-key Open-Meteo rainfall & temperature client
│   ├── collector.py                   # Polite guest-mode Playwright collector (backs off on 403/429)
│   ├── warehouse_sql.py               # DuckDB window-function SQL (truck schedules + leak-free ML features)
│   ├── stats_audit.py                 # SciPy Chi-Square & Mann-Whitney U statistical tests
│   └── model.py                       # Chronological LightGBM 2-hr stockout classifier + TreeSHAP
├── scripts/
│   ├── generate_seed_data.py          # Reproducible 14-day finite-inventory & congestion simulator
│   ├── run_hourly_collector.py        # CLI runner (`--dry-run`, `--max-pincodes`, headless default)
│   └── install_mac_cron.sh            # 1-command macOS silent hourly background scheduler (08:00–23:00)
├── app/
│   └── streamlit_app.py               # Interactive 4-tab visual simulator, truck clock & SQL explorer
├── artifacts/
│   └── model_metrics.json             # Verified holdout ROC-AUC, PR-AUC, Brier score & top SHAP features
└── tests/                             # 8 unit tests covering schemas, SQL LAG(), and ML leakage
```

---

## 📊 Verified Results & Statistical Findings

*(Computed on the 53,760-row 14-day benchmark; reproducible via `pytest` and `scripts/generate_seed_data.py`)*

### 1. Predicting 2-Hour-Ahead Stockouts (Items Currently *In Stock*)
> **Important Anti-Leakage Design:** Predicting whether an *already* out-of-stock item stays out of stock is trivial (~95% persistence). We filter **strictly to items currently `in_stock = TRUE`** (`42,879` rows, `6.54%` 2-hour sellout base rate) and evaluate on a **strict chronological future holdout** (first 75% days train → final 25% days test, zero random-shuffle time leakage).

| Metric | Naive Historical Baseline (`48h SKU Rate`) | `LightGBM` Chronological Holdout | Improvement |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | `0.5636` | **`0.9278`** | **`+0.3642`** |
| **PR-AUC (Average Precision)** | `0.0786` | **`0.5698`** | **`7.25x Lift`** |
| **TimeSeriesSplit CV ROC-AUC** | — | **`0.9137`** | Consistent across 4 forward folds |
| **Brier Score (Calibration)** | — | **`0.0404`** | Tightly calibrated probabilities |

**Top 5 Global `TreeSHAP` Predictors:**
1. `eta_mins` (*Current Delivery ETA*) — `#1` congestion early warning (`mean |SHAP| = 1.09`)
2. `eta_rolling_3h` (*3-Hour Rolling Average ETA*) — sustained warehouse backlog (`0.90`)
3. `hours_since_last_restock` (*Hours Since Last Supply Truck*) — shelf depletion runway (`0.35`)
4. `hour_of_day` (*Diurnal Demand Cycle*) — morning vs. midnight rush (`0.34`)
5. `item_id` (*SKU Velocity Tier*) — fast-perishable staples vs. slow household goods (`0.26`)

### 2. `SciPy` Statistical Hypothesis Audit
* **Rain vs. Delivery Congestion (`Mann-Whitney U`):** Average quoted ETA jumps from **`11.8 mins` (dry)** to **`20.4 mins` (rain)** — **`1.73x` slower (`p < 0.0001`)**.
* **Late-Night Neighborhood Split (`Chi-Square`):** Between **9:00 PM and 2:00 AM**, IT-Hub dark stores experience **`83.9%`** snack/beverage stockouts vs. **`12.2%`** in Family Residential zones — **`6.88x` higher (`p < 0.0001`)**.
* **Warehouse Truck Recovery (`DuckDB` SQL `LAG()`):** Recovers **8 out of 8** morning (`05:00` / `06:00`) and afternoon (`13:00`–`16:00`) replenishment windows solely from public stock transitions.

---

## 🧰 Why We Used This Tech Stack — And Tech Stack Alternatives

Interviewers love asking: *"Why did you pick Tool X instead of Tool Y?"* Here is the exact engineering reasoning behind every layer—and when you should swap in an alternative:

| Layer | Chosen Tool | Why We Picked It Here | Alternatives Considered | Why We Rejected / When to Switch |
| :--- | :--- | :--- | :--- | :--- |
| **Web Ingestion** | **`Playwright` (Headless Guest)** | Quick-commerce sites render prices, pincode modals, and stock badges client-side via JavaScript/React. | `requests` + `BeautifulSoup`, `Selenium` | `BeautifulSoup` sees an empty HTML shell on React SPAs; `Selenium` is slower and flakier with modern async DOM waits. |
| **Weather Feed** | **`Open-Meteo API`** | Free, accurate hourly/live rainfall & temperature by `(lat, lon)` with **zero API key** so anyone cloning this repo runs immediately. | `OpenWeatherMap`, `Tomorrow.io`, `IMD` | Require signup keys, billing cards, or strict free-tier rate limits. |
| **Data Quality** | **`Pandera` + `Pydantic`** | Lightweight in-memory schema checks catch broken DOM scrapes (e.g., `price = ₹0` or `ETA = 999`) in 5ms with zero server setup. | `Great Expectations`, `dbt tests`, Raw `if/else` | `Great Expectations` adds heavy config boilerplate for a single-table pipeline; raw `if/else` scatters validation rules. |
| **Storage & SQL** | **`DuckDB` + `Parquet`** | Runs complex analytical window functions (`LAG`, `LEAD`, rolling frames) directly on compressed `.parquet` files in **<25ms** at \$0 cost. | `Apache Spark`, `PostgreSQL / Supabase`, `SQLite` | **Using Spark on 50k–500k rows (~5 MB) is a classic over-engineering red flag.** Switch to **Supabase Postgres** only if multiple cloud workers write concurrently; switch to **Spark/BigQuery** past 50M+ rows. |
| **Predictive ML** | **`LightGBM` + `TimeSeriesSplit`** | Natively handles categorical pincode/SKU features, non-linear interactions (`Hour × Zone × Rain`), trains in <2 seconds, and beats deep nets on tabular data. | `LSTM / Transformers`, `Logistic Regression`, `XGBoost` | `LSTM` requires heavy sequence padding and underperforms gradient boosting on small-to-medium tabular panels; `Logistic Regression` misses sharp interaction cliffs (`Friday 11 PM × IT Hub`). `CatBoost`/`XGBoost` are equally strong drop-in alternatives. |
| **Explainability** | **`TreeSHAP`** | Computes **exact, additive per-prediction feature attributions** in milliseconds so users see *why* Milk is at 81% risk right now. | `LIME`, Built-in `feature_importances_` | Built-in split/gain importance is global-only (can't explain a single live prediction); `LIME` uses noisy local linear approximations. |
| **Dashboard** | **`Streamlit` + `Plotly`** | Pure Python, interactive sliders + SQL playground, free 1-click hosting on Streamlit Community Cloud. | `React + FastAPI`, `PowerBI / Tableau` | `React + FastAPI` adds 1,000+ lines of frontend boilerplate without improving DS depth; `PowerBI` cannot run live Python `TreeSHAP` inference on custom slider inputs. |
| **Scheduling** | **macOS `cron` (Local)** | Storefronts block cloud data-center IPs via Cloudflare; running 45s/hr locally on residential Wi-Fi avoids anti-bot conflicts cleanly. | `GitHub Actions Cron`, `Apache Airflow`, `Prefect` | Use **GitHub Actions Cron** if targeting an open public API (see *Project Alternatives* below); use **Airflow/Prefect** when orchestrating 10+ multi-step DAGs. |

---

## 🔄 Project Alternatives (Same Architecture, Different Domains)

Don't want to run a local laptop scheduler on grocery apps, or targeting a different industry? This **exact repository architecture** (`Ingestion -> Pandera -> DuckDB LAG() -> LightGBM + SHAP -> Streamlit`) works as a drop-in blueprint for four other high-impact domains:

1. **🚲 City Bike & EV-Scooter Dock Stockouts (100% Cloud / Zero Anti-Bot Blocks):**
   * **Data Source:** Official open **`GBFS` JSON feeds** (NYC Citi Bike, London Santander, SF Bay Wheels, 500+ global cities—updates every 60s, \$0, zero auth).
   * **Mapping:** `0 bikes available` = Stockout; sudden jump `0 -> 18 bikes` = **Rebalancing Van Arrival**; runs 24/7 on free **GitHub Actions Cron** while your laptop is off!
2. **💊 Critical Pharmacy Medicine Availability Radar:**
   * **Data Source:** Public guest availability across neighborhood pincodes on online pharmacies (ORS, fever strips, insulin pens, nebulizer respules) joined with seasonal flu/weather trends (`Open-Meteo` humidity + AQI).
3. **🔌 Public EV Fast-Charger Bay Congestion & Outage Forecaster:**
   * **Data Source:** Public charger occupancy states (`Available` / `Charging` / `Faulted`) across highway & mall corridors to predict Friday-evening wait queues and detect chronically faulted hardware.
4. **🎟️ Intercity Bus / Cinema Last-Minute Seat Sellout Predictor:**
   * **Data Source:** Hourly remaining-seat counts across route departures (`24h`, `12h`, `6h`, `2h` before departure) to predict whether a route will sell out before 6 PM.

---

## 🧬 Engineered Features & Feature Alternatives (Interview Roadmap)

### Current Leak-Free Features Used (`17` Features)
All predictors in [`warehouse_sql.py`](src/darkstore_pulse/warehouse_sql.py) use **strictly current (`t`) or historical (`<= t-1`) information**:
* **Congestion Precursors:** `eta_mins`, `eta_delta_1h` (*1-hr jump*), `eta_rolling_3h`, `warehouse_oos_share_now` (*fraction of other SKUs already out of stock in the same store*).
* **Replenishment Runway:** `hours_since_last_restock` (*hours elapsed since last `False -> True` flip*), `item_hist_oos_rate` (*past-only 48h rolling OOS rate*).
* **Weather & Pricing:** `rain_mm`, `temp_c`, `surge_fee_inr`, `price_inr`.
* **Temporal & Spatial Context:** `hour_of_day`, `day_of_week`, `is_weekend`, `zone_type`, `pincode`, `category`, `item_id`.

### Feature & Modeling Alternatives (What to Add Next)

| Alternative Feature / Formulation | Hypothesis & Expected Signal | Trade-Off / Why Optional |
| :--- | :--- | :--- |
| **1. Live Cricket (IPL) / Festival Calendar Flag** | Match start times (7:30 PM IST) and festivals cause 3x sudden spikes in `LateNight_Snacks` & `Dairy` that calendar `day_of_week` alone misses. | Easy CSV join (`ics` sports/holiday calendar); huge boost on event nights. |
| **2. Cross-SKU "Canary Item" Co-Depletion** | Certain fast-moving items act as *canaries*: if `Bread` sells out at 7:00 PM, shoppers substitute or co-buy `Eggs` & `Butter`, depleting them within 60 mins. | Requires wider pivot columns (`bread_in_stock_now`, `ice_in_stock_now`); strong lift once >30 days of data accumulate. |
| **3. Cross-App Competitor Spillover (`Zepto` → `Blinkit`)** | When App A adds a ₹30 surge fee or runs out of Milk in `560038`, users immediately switch to App B in the same pincode, draining App B 20 mins later. | Doubles hourly URL checks (`2x` runtime per sweep). |
| **4. Live Road Traffic Index (`TomTom` / `Google Routes`)** | Separates *road traffic delays* (slow riders) from *inside-warehouse picking backlog* (high order volume). | Requires free-tier routing API key. |
| **5. Salary / Payday Cycle (`Days 1–5` vs. `Days 25–31`)** | Premium snacks/beverages (`Red Bull`, `Butter`) deplete faster in the first week after monthly salary credit. | Needs >= 2 full monthly cycles (60+ days) to avoid overfitting. |
| **6. Cyclical Fourier Time Encodings (`sin/cos hour`)** | Ensures smooth continuity between `23:00` and `00:00` (which look far apart as integers `23` and `0`). | Necessary for linear/neural models; optional for tree models (`LightGBM` splits intervals cleanly, though still helpful). |
| **7. Survival Analysis (`Random Survival Forests`)** | Instead of binary `will_stockout_next_2h`, predicts a continuous **Time-to-Stockout survival curve** (e.g., *"Median 3.4 hours remaining"*). | Higher UI complexity, great advanced upgrade via `scikit-survival`. |
| **8. Conformal Prediction Intervals (`MAPIE`)** | Wraps ETA regression to output guaranteed bounds (*"Delivery in 15–22 mins with 90% coverage"*). | Best paired when switching target from binary stockout to ETA forecasting. |

---

## 🚀 Quickstart (Run Everything in 60 Seconds)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/velagapudirushi/darkstore-pulse.git
cd darkstore-pulse
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard Immediately (Uses Included 53,760-Row Dataset)
```bash
streamlit run app/streamlit_app.py
```

### 3. Run the Automated Test Suite (`pytest`)
```bash
pytest -v
```

### 4. *(Optional)* Start Collecting Live Neighborhood Data Silently on macOS
No account login or OTP needed—runs politely in headless guest mode once an hour (`08:00–23:00`) while your Mac is awake:
```bash
# Install Playwright + Chromium once for local collection
pip install playwright && playwright install chromium

# Smoke-test config & live Open-Meteo weather connectivity
python3 scripts/run_hourly_collector.py --dry-run

# Run 1 quick live test check (1 pincode, 3 items)
python3 scripts/run_hourly_collector.py --max-pincodes 1 --max-items 3

# Enable silent background hourly cron timer (8 AM - 11 PM)
./scripts/install_mac_cron.sh
```
Once `data/raw/live_observations.parquet` appears, a toggle automatically unlocks inside the Streamlit sidebar so you can switch seamlessly between the **14-Day Benchmark Dataset** and your **Live Collected Data**!

---

## 📜 Ethical Collection Note
This repository is built strictly for **educational Data Science & operations research**. The collector accesses only public, unauthenticated guest product pages at low human-equivalent frequency (`15–20` pages per hour with `3.5–6.0s` delays), never logs into accounts, and halts immediately upon receiving HTTP `403` / `429` responses rather than attempting anti-bot evasion.
