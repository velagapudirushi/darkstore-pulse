"""Statistical hypothesis audits proving weather & neighborhood behavioral effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class HypothesisResult:
    """Structured result of a statistical hypothesis test."""

    test_name: str
    null_hypothesis: str
    metric_group_a: float
    metric_group_b: float
    relative_lift_x: float
    statistic: float
    p_value: float
    significant_at_0_01: bool
    plain_english_verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_statistical_audit(df: pd.DataFrame) -> dict[str, HypothesisResult]:
    """Run three operational hypothesis tests on collected dark-store observations."""
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work["hour"] = work["timestamp"].dt.hour
    work["is_oos"] = (~work["in_stock"]).astype(int)

    # 1. Chi-Square Test: Rain (>0.5mm) vs Dry Stockout Rate
    rainy = work[work["rain_mm"] > 0.5]
    dry = work[work["rain_mm"] <= 0.5]

    rain_oos_rate = float(rainy["is_oos"].mean()) if len(rainy) else 0.0
    dry_oos_rate = float(dry["is_oos"].mean()) if len(dry) else 0.0
    contingency_rain = np.array(
        [
            [int(rainy["is_oos"].sum()), int((1 - rainy["is_oos"]).sum())],
            [int(dry["is_oos"].sum()), int((1 - dry["is_oos"]).sum())],
        ]
    )
    chi2_rain, p_rain, _, _ = stats.chi2_contingency(contingency_rain)
    rain_lift = round(rain_oos_rate / max(dry_oos_rate, 1e-6), 2)

    h1 = HypothesisResult(
        test_name="Chi-Square Test: Rain vs. Dry Stockout Incidence",
        null_hypothesis="Rainfall (>0.5 mm/hr) has no association with item stockout rates.",
        metric_group_a=round(rain_oos_rate * 100, 2),
        metric_group_b=round(dry_oos_rate * 100, 2),
        relative_lift_x=rain_lift,
        statistic=round(float(chi2_rain), 2),
        p_value=float(p_rain),
        significant_at_0_01=bool(p_rain < 0.01),
        plain_english_verdict=(
            f"Rainy hours show a {rain_oos_rate*100:.1f}% stockout rate vs. "
            f"{dry_oos_rate*100:.1f}% in dry weather ({rain_lift}x higher, p = {p_rain:.2e})."
        ),
    )

    # 2. Mann-Whitney U Test: Non-parametric Delivery ETA shift during Rain
    u_stat, p_eta = stats.mannwhitneyu(
        rainy["eta_mins"].to_numpy(),
        dry["eta_mins"].to_numpy(),
        alternative="greater",
    )
    rain_eta_mean = float(rainy["eta_mins"].mean())
    dry_eta_mean = float(dry["eta_mins"].mean())
    eta_lift = round(rain_eta_mean / max(dry_eta_mean, 1e-6), 2)

    h2 = HypothesisResult(
        test_name="Mann-Whitney U Test: Rain vs. Dry Delivery ETA Congestion",
        null_hypothesis="Delivery ETA distributions are identical during rainy and dry hours.",
        metric_group_a=round(rain_eta_mean, 2),
        metric_group_b=round(dry_eta_mean, 2),
        relative_lift_x=eta_lift,
        statistic=round(float(u_stat), 2),
        p_value=float(p_eta),
        significant_at_0_01=bool(p_eta < 0.01),
        plain_english_verdict=(
            f"Average delivery ETA jumps from {dry_eta_mean:.1f} mins (dry) to "
            f"{rain_eta_mean:.1f} mins during rain ({eta_lift}x slower, p = {p_eta:.2e})."
        ),
    )

    # 3. Chi-Square Test: Late-Night (21:00–02:00) Snack Stockouts in IT Hubs vs Family Zones
    late_snacks = work[
        (work["category"] == "LateNight_Snacks")
        & ((work["hour"] >= 21) | (work["hour"] <= 2))
    ]
    it_late = late_snacks[late_snacks["zone_type"] == "IT_Hub"]
    fam_late = late_snacks[late_snacks["zone_type"] == "Family_Residential"]

    it_rate = float(it_late["is_oos"].mean()) if len(it_late) else 0.0
    fam_rate = float(fam_late["is_oos"].mean()) if len(fam_late) else 0.0
    contingency_zone = np.array(
        [
            [int(it_late["is_oos"].sum()), int((1 - it_late["is_oos"]).sum())],
            [int(fam_late["is_oos"].sum()), int((1 - fam_late["is_oos"]).sum())],
        ]
    )
    chi2_zone, p_zone, _, _ = stats.chi2_contingency(contingency_zone)
    zone_lift = round(it_rate / max(fam_rate, 1e-6), 2)

    h3 = HypothesisResult(
        test_name="Chi-Square Test: Late-Night Snack Depletion (IT Hubs vs. Family Zones)",
        null_hypothesis="Late-night snack stockouts occur equally in IT-Hub and Family pincodes.",
        metric_group_a=round(it_rate * 100, 2),
        metric_group_b=round(fam_rate * 100, 2),
        relative_lift_x=zone_lift,
        statistic=round(float(chi2_zone), 2),
        p_value=float(p_zone),
        significant_at_0_01=bool(p_zone < 0.01),
        plain_english_verdict=(
            f"Between 9 PM and 2 AM, IT-Hub dark stores run out of snacks/drinks "
            f"{it_rate*100:.1f}% of the time vs. {fam_rate*100:.1f}% in family zones "
            f"({zone_lift}x higher, p = {p_zone:.2e})."
        ),
    )

    return {
        "rain_stockout": h1,
        "rain_eta": h2,
        "zone_latenight_snacks": h3,
    }
