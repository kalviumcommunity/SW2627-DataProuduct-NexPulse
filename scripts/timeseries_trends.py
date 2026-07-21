"""
timeseries_trends.py
--------------------
Time-Series Trend & Rolling Metrics for the NexPulse workflow.

Responsibilities
----------------
1. Rolling windows — 7-day, 14-day, 30-day moving averages to smooth noise
   and reveal underlying trends.
2. Resampling — aggregate daily data to weekly, monthly, quarterly for
   period-over-period analysis.
3. Period-over-period change — MoM, WoW, QoQ percentage change using .pct_change().
4. Cumulative sums — running totals for revenue, orders, customers.
5. Trend identification — detect uptrend, downtrend, flat, acceleration/deceleration.
6. Visualizations — raw vs smoothed, resampled bar charts, cumulative growth curves.
7. Business interpretations — connect patterns to actionable insights.

Usage
-----
    # Built-in demo dataset (365 days):
    python scripts/timeseries_trends.py --demo

    # Real CSV (must have a 'date' column):
    python scripts/timeseries_trends.py --input data/processed/daily_metrics.csv

    # Custom rolling window sizes:
    python scripts/timeseries_trends.py --demo --windows 7,30,90
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data(days: int = 365) -> pd.DataFrame:
    """
    Return a synthetic daily revenue dataset covering one full year with:
      - Underlying positive trend (+15 % growth over the year)
      - Weekly seasonality (weekends drop ~30 %)
      - Monthly seasonality (start-of-month spike)
      - Random noise (±20 % daily volatility)

    This exercises the scenario from the assignment: raw daily numbers are
    noisy and misleading; rolling averages reveal the true upward trend.
    """
    rng = np.random.default_rng(seed=77)
    start_date = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    # Base revenue with linear growth
    base = np.linspace(40_000, 46_000, days)

    # Weekly seasonality: weekends drop 30 %
    day_of_week = dates.dayofweek
    weekend_mask = (day_of_week >= 5)
    weekly_factor = np.where(weekend_mask, 0.7, 1.0)

    # Monthly seasonality: first 3 days of month spike +20 %
    day_of_month = dates.day
    monthly_spike = np.where(day_of_month <= 3, 1.2, 1.0)

    # Random noise ±20 %
    noise = rng.uniform(0.8, 1.2, size=days)

    revenue = (base * weekly_factor * monthly_spike * noise).round(2)

    orders = (revenue / rng.uniform(50, 150, size=days)).astype(int)
    customers = (orders * rng.uniform(0.6, 0.9, size=days)).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue,
            "orders": orders,
            "customers": customers,
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Rolling windows
# ---------------------------------------------------------------------------

def compute_rolling_metrics(
    df: pd.DataFrame,
    windows: list[int],
    value_col: str = "revenue",
) -> pd.DataFrame:
    """
    Compute rolling mean, rolling std, and rolling z-score for multiple
    window sizes.

    New columns added (per window)
    ------------------------------
    <value_col>_ma<N>     : N-day moving average
    <value_col>_std<N>    : N-day rolling standard deviation (volatility)
    <value_col>_zscore<N> : (value - ma<N>) / std<N>  → how far from trend?
    """
    enriched = df.copy()

    for w in windows:
        col_ma = f"{value_col}_ma{w}"
        col_std = f"{value_col}_std{w}"
        col_z = f"{value_col}_zscore{w}"

        enriched[col_ma] = enriched[value_col].rolling(window=w, min_periods=1).mean().round(2)
        enriched[col_std] = enriched[value_col].rolling(window=w, min_periods=1).std().round(2)

        # Z-score: (raw - ma) / std  → detects anomalies (|z| > 2)
        enriched[col_z] = (
            (enriched[value_col] - enriched[col_ma]) / enriched[col_std]
        ).round(3)

    return enriched


# ---------------------------------------------------------------------------
# Step 2 – Resampling
# ---------------------------------------------------------------------------

def resample_to_periods(
    df: pd.DataFrame,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """
    Resample daily data to weekly, monthly, quarterly views.

    Each resampled DataFrame includes:
      - revenue_sum, revenue_mean
      - orders_sum, orders_mean
      - customers_sum
      - period_label (for reporting)

    Returns
    -------
    dict keyed by "weekly", "monthly", "quarterly"
    """
    df_ts = df.set_index(date_col).sort_index()
    resampled: dict[str, pd.DataFrame] = {}

    for label, freq in [("weekly", "W"), ("monthly", "M"), ("quarterly", "Q")]:
        agg_result = df_ts.resample(freq).agg(
            revenue_sum_sum   =("revenue",   "sum"),
            revenue_mean_mean =("revenue",   "mean"),
            orders_sum_sum    =("orders",    "sum"),
            orders_mean_mean  =("orders",    "mean"),
            customers_sum     =("customers", "sum"),
        ).round(2)
        agg_result["period_label"] = agg_result.index.strftime("%Y-%m-%d")
        resampled[label] = agg_result.reset_index()

    return resampled


# ---------------------------------------------------------------------------
# Step 3 – Period-over-period change
# ---------------------------------------------------------------------------

def compute_period_changes(resampled: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Compute WoW, MoM, QoQ percentage change for revenue_sum and orders_sum.

    New columns added
    -----------------
    revenue_pct_change : % change from previous period
    orders_pct_change  : % change from previous period
    """
    for label, df in resampled.items():
        df["revenue_pct_change"] = df["revenue_sum_sum"].pct_change().mul(100).round(2)
        df["orders_pct_change"] = df["orders_sum_sum"].pct_change().mul(100).round(2)
    return resampled


# ---------------------------------------------------------------------------
# Step 4 – Cumulative sums
# ---------------------------------------------------------------------------

def add_cumulative_sums(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add running totals for revenue, orders, customers.

    New columns
    -----------
    revenue_cumsum
    orders_cumsum
    customers_cumsum
    """
    enriched = df.copy()
    enriched["revenue_cumsum"] = enriched["revenue"].cumsum().round(2)
    enriched["orders_cumsum"] = enriched["orders"].cumsum()
    enriched["customers_cumsum"] = enriched["customers"].cumsum()
    return enriched


# ---------------------------------------------------------------------------
# Step 5 – Trend identification
# ---------------------------------------------------------------------------

def identify_trends(resampled: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Detect trend direction for monthly data:
      uptrend, downtrend, flat, accelerating, decelerating

    Returns
    -------
    dict keyed by "weekly", "monthly", "quarterly" with:
      - direction: "uptrend" | "downtrend" | "flat"
      - acceleration: "accelerating" | "decelerating" | "steady"
      - interpretation: plain-English summary
    """
    trends: dict[str, dict] = {}

    for label, df in resampled.items():
        if len(df) < 3:
            trends[label] = {"direction": "insufficient data", "acceleration": "n/a"}
            continue

        rev = df["revenue_sum_sum"].values
        last_3 = rev[-3:]

        # Direction: compare last period to 3 periods ago
        trend_val = last_3[-1] - last_3[0]
        if trend_val > last_3[0] * 0.05:
            direction = "uptrend"
        elif trend_val < -last_3[0] * 0.05:
            direction = "downtrend"
        else:
            direction = "flat"

        # Acceleration: compare slope of last 2 periods vs previous 2
        if len(last_3) == 3:
            slope_recent = last_3[-1] - last_3[-2]
            slope_prior = last_3[-2] - last_3[-3]
            if slope_recent > slope_prior * 1.2:
                acceleration = "accelerating"
            elif slope_recent < slope_prior * 0.8:
                acceleration = "decelerating"
            else:
                acceleration = "steady"
        else:
            acceleration = "insufficient data"

        # Business interpretation
        if direction == "uptrend" and acceleration == "accelerating":
            interp = (
                f"{label.capitalize()} revenue is growing AND accelerating. "
                f"This is healthy momentum. Continue current strategy and scale investment."
            )
        elif direction == "uptrend" and acceleration == "decelerating":
            interp = (
                f"{label.capitalize()} revenue is still growing but momentum is slowing. "
                f"Investigate: market saturation? Competitive pressure? Need new growth levers."
            )
        elif direction == "downtrend":
            interp = (
                f"{label.capitalize()} revenue is declining. Immediate action required. "
                f"Diagnose root cause: churn spike? Sales pipeline issue? Product-market fit erosion?"
            )
        elif direction == "flat":
            interp = (
                f"{label.capitalize()} revenue is stable with no clear trend. "
                f"Monitor closely; flat can precede either growth or decline."
            )
        else:
            interp = f"{label.capitalize()} trend unclear."

        trends[label] = {
            "direction": direction,
            "acceleration": acceleration,
            "interpretation": interp,
            "last_3_periods": [round(float(v), 2) for v in last_3],
        }

    return trends


# ---------------------------------------------------------------------------
# Step 6 – Visualizations
# ---------------------------------------------------------------------------

def plot_raw_vs_smoothed(
    df: pd.DataFrame,
    windows: list[int],
    value_col: str,
    output_dir: Path,
) -> Path:
    """
    Plot raw daily values vs multiple rolling averages on the same chart.
    Shows how smoothing reveals the underlying trend.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw data (semi-transparent)
    ax.plot(
        df["date"], df[value_col],
        label="Raw daily",
        color="gray",
        alpha=0.3,
        linewidth=0.8,
    )

    # Rolling averages
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, w in enumerate(windows):
        col = f"{value_col}_ma{w}"
        if col in df.columns:
            ax.plot(
                df["date"], df[col],
                label=f"{w}-day MA",
                color=colors[i % len(colors)],
                linewidth=2.5,
            )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(value_col.capitalize(), fontsize=11)
    ax.set_title(
        f"{value_col.capitalize()}: Raw vs Smoothed (Rolling Averages)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    p = output_dir / f"timeseries_{value_col}_raw_vs_smoothed.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_resampled_bars(
    resampled: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """
    Bar charts for weekly, monthly, quarterly revenue_sum with % change annotations.
    """
    paths: list[Path] = []

    for label, df in resampled.items():
        fig, ax = plt.subplots(figsize=(12, 5))

        x = np.arange(len(df))
        revenue = df["revenue_sum_sum"].values
        pct_change = df["revenue_pct_change"].values

        bars = ax.bar(x, revenue, color="#3498db", alpha=0.8, edgecolor="black")

        # Annotate with % change
        for i, (bar, pct) in enumerate(zip(bars, pct_change)):
            if np.isnan(pct):
                continue
            y_pos = bar.get_height()
            color = "green" if pct >= 0 else "red"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(df["period_label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Revenue ($)", fontsize=11)
        ax.set_title(
            f"{label.capitalize()} Revenue with Period-over-Period % Change",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)

        p = output_dir / f"timeseries_{label}_revenue_bars.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def plot_cumulative_growth(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Line chart showing cumulative revenue, orders, customers over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax2 = ax.twinx()

    # Revenue on left axis
    ax.plot(
        df["date"], df["revenue_cumsum"],
        label="Cumulative Revenue ($)",
        color="#e74c3c",
        linewidth=2.5,
    )
    ax.set_ylabel("Cumulative Revenue ($)", fontsize=11, color="#e74c3c")
    ax.tick_params(axis="y", labelcolor="#e74c3c")

    # Orders + customers on right axis
    ax2.plot(
        df["date"], df["orders_cumsum"],
        label="Cumulative Orders",
        color="#3498db",
        linewidth=2,
    )
    ax2.plot(
        df["date"], df["customers_cumsum"],
        label="Cumulative Customers",
        color="#2ecc71",
        linewidth=2,
    )
    ax2.set_ylabel("Cumulative Count", fontsize=11)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Cumulative Growth Over Time", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    p = output_dir / "timeseries_cumulative_growth.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Step 7 – Console report
# ---------------------------------------------------------------------------

def print_report(
    df: pd.DataFrame,
    resampled: dict[str, pd.DataFrame],
    trends: dict[str, dict],
    windows: list[int],
) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("TIME-SERIES TREND & ROLLING METRICS — REPORT")
    print(sep)

    print(f"\n  Time period : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Days        : {len(df)}")
    print(f"  Total revenue: ${df['revenue'].sum():,.0f}")

    # Raw volatility
    raw_std = df["revenue"].std()
    raw_mean = df["revenue"].mean()
    cv = (raw_std / raw_mean) * 100
    print(f"\n  ── Raw daily volatility ──")
    print(f"  Mean   : ${raw_mean:,.0f}")
    print(f"  Std    : ${raw_std:,.0f}")
    print(f"  CV     : {cv:.1f}%  (coefficient of variation = std/mean)")

    # Smoothed volatility
    print(f"\n  ── Smoothed volatility (rolling windows) ──")
    for w in windows:
        col = f"revenue_ma{w}"
        if col in df.columns:
            ma_std = df[col].std()
            ma_mean = df[col].mean()
            ma_cv = (ma_std / ma_mean) * 100
            reduction = ((cv - ma_cv) / cv) * 100
            print(f"  {w}-day MA : CV={ma_cv:.1f}%  ({reduction:.0f}% noise reduction)")

    # Trend identification
    print(f"\n  ── Trend identification ──")
    for label, trend in trends.items():
        print(f"\n  [{label.upper()}]")
        print(f"    Direction    : {trend['direction']}")
        print(f"    Acceleration : {trend['acceleration']}")
        print(f"    Last 3 periods: {trend['last_3_periods']}")
        print(f"    Interpretation: {trend['interpretation']}")

    # Period-over-period highlights
    print(f"\n  ── Period-over-period highlights (monthly) ──")
    monthly = resampled["monthly"]
    if len(monthly) >= 2:
        last_2 = monthly.tail(2)
        prev_month = last_2.iloc[0]
        curr_month = last_2.iloc[1]
        mom_change = curr_month["revenue_pct_change"]
        print(f"  {prev_month['period_label']} : ${prev_month['revenue_sum_sum']:,.0f}")
        print(f"  {curr_month['period_label']} : ${curr_month['revenue_sum_sum']:,.0f}  "
              f"({mom_change:+.1f}% MoM)")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 8 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    resampled: dict[str, pd.DataFrame],
    trends: dict[str, dict],
    chart_paths: list[Path],
    output_dir: Path,
) -> None:
    """
    Files created
    -------------
    output/timeseries_analysis/daily_enriched.csv       — daily + rolling + cumsum
    output/timeseries_analysis/weekly_resampled.csv
    output/timeseries_analysis/monthly_resampled.csv
    output/timeseries_analysis/quarterly_resampled.csv
    output/timeseries_analysis/trends_report.json
    output/timeseries_analysis/*.png (5 charts)
    """
    df.to_csv(output_dir / "daily_enriched.csv", index=False)
    for label, rs_df in resampled.items():
        rs_df.to_csv(output_dir / f"{label}_resampled.csv", index=False)

    with (output_dir / "trends_report.json").open("w", encoding="utf-8") as fh:
        json.dump(trends, fh, indent=4, default=str)

    print(f"[save_outputs] All tables and charts saved → {output_dir}/")
    print(f"[save_outputs] Charts: {[p.name for p in chart_paths]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-Series Trend & Rolling Metrics — NexPulse"
    )
    parser.add_argument("--input", dest="input_path",
                        help="Path to a CSV file with a 'date' column.")
    parser.add_argument("--windows", dest="windows", default="7,14,30",
                        help="Comma-separated rolling window sizes (default: 7,14,30).")
    parser.add_argument("--demo", action="store_true",
                        help="Force the built-in demo dataset (365 days).")
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        print("Using built-in demo dataset (365 days).")
        return build_demo_data(), "demo dataset"
    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded: {path}  ({len(df)} rows)")
    return df, str(path)


def main() -> None:
    args = parse_args()
    windows = [int(w.strip()) for w in args.windows.split(",")]

    df, source_label = load_input(args)
    print(f"\nSource  : {source_label}")
    print(f"Rows    : {len(df)}")
    print(f"Windows : {windows}\n")

    output_dir = Path("output/timeseries_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rolling metrics
    print("[1/5] Computing rolling windows …")
    df = compute_rolling_metrics(df, windows)

    # 2. Resampling
    print("[2/5] Resampling to weekly / monthly / quarterly …")
    resampled = resample_to_periods(df)

    # 3. Period-over-period
    print("[3/5] Computing period-over-period changes …")
    resampled = compute_period_changes(resampled)

    # 4. Cumulative sums
    print("[4/5] Adding cumulative sums …")
    df = add_cumulative_sums(df)

    # 5. Trend identification
    print("[5/5] Identifying trends …")
    trends = identify_trends(resampled)

    # 6. Visualizations
    chart_paths = []
    chart_paths.append(plot_raw_vs_smoothed(df, windows, "revenue", output_dir))
    chart_paths.extend(plot_resampled_bars(resampled, output_dir))
    chart_paths.append(plot_cumulative_growth(df, output_dir))

    # 7. Console report
    print_report(df, resampled, trends, windows)

    # 8. Persist
    save_outputs(df, resampled, trends, chart_paths, output_dir)

    print(f"\nTime-series analysis complete.")
    print(f"  Daily rows          : {len(df)}")
    print(f"  Weekly periods      : {len(resampled['weekly'])}")
    print(f"  Monthly periods     : {len(resampled['monthly'])}")
    print(f"  Quarterly periods   : {len(resampled['quarterly'])}")
    print(f"  Charts generated    : {len(chart_paths)}")


if __name__ == "__main__":
    main()
