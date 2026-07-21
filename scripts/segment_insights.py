"""
segment_insights.py
-------------------
GroupBy Aggregation & Segment Insights for the NexPulse workflow.

Responsibilities
----------------
1. Single-dimension groupby — compute churn rate, revenue, customer count
   per customer_type, product, and region using .agg().
2. Multi-dimension groupby — slice by (customer_type × product) simultaneously.
3. Transform — broadcast segment-level metrics back to every row via
   .transform() for per-row comparisons.
4. Pivot table — two-dimensional revenue view: customer_type × product.
5. Ranking — rank segments by churn rate, revenue, and LTV.
6. Actionable insights — surface plain-English business findings per segment
   with explicit intervention recommendations.
7. Persist all summary tables and a structured JSON report to output/.

Usage
-----
    # Built-in demo dataset:
    python scripts/segment_insights.py --demo

    # Real CSV:
    python scripts/segment_insights.py --input data/processed/features.csv
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

def build_demo_data(n: int = 3000) -> pd.DataFrame:
    """
    Return a customer transaction dataset that replicates the assignment scenario:
      - Enterprise (5 % of base):  1 % churn,  high revenue  → healthy
      - SMB        (40 % of base): 12 % churn, mid revenue   → intervention needed
      - Startup    (55 % of base):  8 % churn, low revenue   → monitor

    Also includes a product dimension (Analytics / Platform / Support) and
    a region dimension (NA / EMEA / APAC) to exercise multi-level groupby.
    """
    rng = np.random.default_rng(seed=55)

    segment_config = {
        "Enterprise": {"share": 0.05, "churn_p": 0.01, "rev_mu": 12000, "rev_sd": 3000},
        "SMB":        {"share": 0.40, "churn_p": 0.12, "rev_mu": 1500,  "rev_sd": 500},
        "Startup":    {"share": 0.55, "churn_p": 0.08, "rev_mu": 400,   "rev_sd": 150},
    }

    rows = []
    cust_id = 1
    for seg, cfg in segment_config.items():
        count = int(n * cfg["share"])
        for _ in range(count):
            revenue = max(10, rng.normal(cfg["rev_mu"], cfg["rev_sd"]))
            churned = int(rng.random() < cfg["churn_p"])
            product = rng.choice(["Analytics", "Platform", "Support"],
                                 p=[0.4, 0.45, 0.15])
            region  = rng.choice(["NA", "EMEA", "APAC"], p=[0.5, 0.3, 0.2])
            rows.append({
                "customer_id":      cust_id,
                "customer_type":    seg,
                "product":          product,
                "region":           region,
                "revenue":          round(revenue, 2),
                "churned":          churned,
                "support_tickets":  int(rng.poisson(lam=2 + churned * 3)),
                "days_as_customer": int(rng.integers(30, 1460)),
                "discount_pct":     round(rng.uniform(0, 40), 1),
            })
            cust_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1 – Single-dimension .agg()
# ---------------------------------------------------------------------------

def single_dimension_agg(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute key metrics grouped by one dimension at a time.

    Returns a dict of labelled summary DataFrames.
    """
    summaries: dict[str, pd.DataFrame] = {}

    # ── by customer_type ────────────────────────────────────────────────
    by_type = (
        df.groupby("customer_type")
        .agg(
            customer_count  =("customer_id",      "count"),
            total_revenue   =("revenue",           "sum"),
            avg_revenue     =("revenue",           "mean"),
            median_revenue  =("revenue",           "median"),
            churn_count     =("churned",           "sum"),
            churn_rate      =("churned",           "mean"),
            avg_tickets     =("support_tickets",   "mean"),
            avg_discount_pct=("discount_pct",      "mean"),
            avg_days        =("days_as_customer",  "mean"),
        )
        .round(4)
    )
    by_type["revenue_share_pct"] = (
        by_type["total_revenue"] / by_type["total_revenue"].sum() * 100
    ).round(2)
    by_type["customer_share_pct"] = (
        by_type["customer_count"] / by_type["customer_count"].sum() * 100
    ).round(2)
    summaries["by_customer_type"] = by_type

    # ── by product ──────────────────────────────────────────────────────
    by_product = (
        df.groupby("product")
        .agg(
            customer_count=("customer_id", "count"),
            total_revenue =("revenue",     "sum"),
            avg_revenue   =("revenue",     "mean"),
            churn_rate    =("churned",     "mean"),
        )
        .round(4)
    )
    summaries["by_product"] = by_product

    # ── by region ───────────────────────────────────────────────────────
    by_region = (
        df.groupby("region")
        .agg(
            customer_count=("customer_id", "count"),
            total_revenue =("revenue",     "sum"),
            avg_revenue   =("revenue",     "mean"),
            churn_rate    =("churned",     "mean"),
        )
        .round(4)
    )
    summaries["by_region"] = by_region

    return summaries


# ---------------------------------------------------------------------------
# Step 2 – Multi-dimension groupby
# ---------------------------------------------------------------------------

def multi_dimension_agg(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group by two dimensions simultaneously to reveal cross-segment patterns.
    """
    summaries: dict[str, pd.DataFrame] = {}

    # customer_type × product
    by_type_product = (
        df.groupby(["customer_type", "product"])
        .agg(
            customer_count=("customer_id", "count"),
            total_revenue =("revenue",     "sum"),
            churn_rate    =("churned",     "mean"),
        )
        .round(4)
        .reset_index()
    )
    summaries["by_type_x_product"] = by_type_product

    # customer_type × region
    by_type_region = (
        df.groupby(["customer_type", "region"])
        .agg(
            customer_count=("customer_id", "count"),
            total_revenue =("revenue",     "sum"),
            churn_rate    =("churned",     "mean"),
        )
        .round(4)
        .reset_index()
    )
    summaries["by_type_x_region"] = by_type_region

    return summaries


# ---------------------------------------------------------------------------
# Step 3 – .transform() — broadcast segment metrics back to rows
# ---------------------------------------------------------------------------

def add_segment_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use .transform() to add segment-level metrics to every individual row.
    This enables per-row comparisons like 'is this customer above segment avg?'

    New columns added
    -----------------
    segment_avg_revenue     : average revenue for this customer's segment
    segment_churn_rate      : churn rate for this customer's segment
    above_segment_avg       : bool — is this customer's revenue above their segment mean?
    """
    enriched = df.copy()

    enriched["segment_avg_revenue"] = (
        df.groupby("customer_type")["revenue"].transform("mean").round(2)
    )
    enriched["segment_churn_rate"] = (
        df.groupby("customer_type")["churned"].transform("mean").round(4)
    )
    enriched["above_segment_avg"] = (
        enriched["revenue"] > enriched["segment_avg_revenue"]
    )

    return enriched


# ---------------------------------------------------------------------------
# Step 4 – Pivot table
# ---------------------------------------------------------------------------

def build_pivot_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create two-dimensional pivot tables: customer_type × product.
    """
    pivots: dict[str, pd.DataFrame] = {}

    # Revenue pivot
    revenue_pivot = pd.pivot_table(
        df,
        values="revenue",
        index="customer_type",
        columns="product",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Total",
    ).round(2)
    pivots["revenue_by_type_x_product"] = revenue_pivot

    # Churn rate pivot
    churn_pivot = pd.pivot_table(
        df,
        values="churned",
        index="customer_type",
        columns="product",
        aggfunc="mean",
        fill_value=0,
        margins=True,
        margins_name="Overall",
    ).round(4)
    pivots["churn_rate_by_type_x_product"] = churn_pivot

    # Customer count pivot
    count_pivot = pd.pivot_table(
        df,
        values="customer_id",
        index="customer_type",
        columns="product",
        aggfunc="count",
        fill_value=0,
        margins=True,
        margins_name="Total",
    )
    pivots["count_by_type_x_product"] = count_pivot

    return pivots


# ---------------------------------------------------------------------------
# Step 5 – Rank segments
# ---------------------------------------------------------------------------

def rank_segments(by_type: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranking columns to the customer_type summary table.

    Ranks
    -----
    churn_rank      : 1 = lowest churn (best), ascending
    revenue_rank    : 1 = highest revenue (best), descending
    ltv_rank        : lifetime value per customer; 1 = highest
    """
    ranked = by_type.copy()

    ranked["ltv_per_customer"] = (
        ranked["total_revenue"] / ranked["customer_count"]
    ).round(2)

    ranked["churn_rank"]   = ranked["churn_rate"].rank(ascending=True).astype(int)
    ranked["revenue_rank"] = ranked["total_revenue"].rank(ascending=False).astype(int)
    ranked["ltv_rank"]     = ranked["ltv_per_customer"].rank(ascending=False).astype(int)

    return ranked.sort_values("churn_rate", ascending=True)


# ---------------------------------------------------------------------------
# Step 6 – Surface actionable insights
# ---------------------------------------------------------------------------

def generate_segment_insights(ranked: pd.DataFrame) -> list[dict]:
    """
    Produce a plain-English business insight + intervention recommendation
    for every segment based on its churn rate and revenue share.

    Logic
    -----
    churn_rate < 0.03 and revenue_share > 50 %  → healthy, protect
    churn_rate > 0.10                            → critical, urgent intervention
    churn_rate 0.05–0.10                         → at risk, early intervention
    churn_rate < 0.05                            → healthy, monitor
    """
    insights: list[dict] = []

    total_revenue = ranked["total_revenue"].sum()

    for seg, row in ranked.iterrows():
        churn  = row["churn_rate"]
        rev    = row["total_revenue"]
        count  = row["customer_count"]
        rev_sh = row["revenue_share_pct"]
        cu_sh  = row["customer_share_pct"]
        ltv    = row.get("ltv_per_customer", rev / count)

        # Determine health status
        if churn < 0.03 and rev_sh > 40:
            status = "HEALTHY — HIGH VALUE"
            action = (
                f"Protect this segment. Prioritise dedicated account management, "
                f"proactive QBRs, and executive sponsorship. "
                f"Upsell opportunities exist given high LTV (${ltv:,.0f}/customer)."
            )
        elif churn > 0.10:
            status = "CRITICAL — URGENT INTERVENTION"
            action = (
                f"Immediate churn prevention campaign required. "
                f"Investigate root causes: onboarding gaps, product-fit issues, "
                f"support quality. Consider a dedicated success programme for "
                f"this segment. {churn*100:.1f}% churn on {count:,} customers "
                f"= {int(count*churn):,} lost customers per cycle."
            )
        elif churn > 0.05:
            status = "AT RISK — EARLY INTERVENTION"
            action = (
                f"Launch early-warning churn signals and proactive outreach. "
                f"Increase product adoption initiatives. Segment churn of "
                f"{churn*100:.1f}% is above the healthy threshold of 5%."
            )
        else:
            status = "HEALTHY — MONITOR"
            action = (
                f"Maintain current engagement programmes. "
                f"Track trend monthly; intervene if churn crosses 5%."
            )

        insights.append({
            "segment":          seg,
            "status":           status,
            "churn_rate_pct":   round(churn * 100, 2),
            "revenue_share_pct": rev_sh,
            "customer_share_pct": cu_sh,
            "ltv_per_customer": round(ltv, 2),
            "churn_rank":       int(row["churn_rank"]),
            "revenue_rank":     int(row["revenue_rank"]),
            "action":           action,
        })

    return insights


# ---------------------------------------------------------------------------
# Step 7 – Visualisations
# ---------------------------------------------------------------------------

def plot_segment_overview(ranked: pd.DataFrame, insights: list[dict], output_dir: Path) -> list[Path]:
    """
    Generate three charts:
      1. Stacked bar: revenue share vs customer share per segment
      2. Bar chart: churn rate per segment with danger threshold line
      3. Bubble chart: churn rate vs LTV sized by customer count
    """
    paths: list[Path] = []
    segments = ranked.index.tolist()
    colors   = ["#2ecc71", "#e74c3c", "#f39c12"][:len(segments)]

    # ── Chart 1: Revenue share vs customer share ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(segments))
    w = 0.35
    rev_shares  = ranked["revenue_share_pct"].values
    cust_shares = ranked["customer_share_pct"].values

    bars1 = ax.bar(x - w/2, rev_shares,  w, label="Revenue share %",  color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, cust_shares, w, label="Customer share %", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Revenue Share vs Customer Share by Segment", fontsize=13, fontweight="bold")
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    p = output_dir / "segment_revenue_vs_customer_share.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # ── Chart 2: Churn rate per segment ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    churn_vals = ranked["churn_rate"].values * 100
    bars = ax.bar(segments, churn_vals, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(5, color="red", linestyle="--", linewidth=1.8, label="5% danger threshold")
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_title("Churn Rate by Customer Segment", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    p = output_dir / "segment_churn_rate.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # ── Chart 3: Bubble — churn rate vs LTV, sized by customer count ─────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, seg in enumerate(segments):
        row = ranked.loc[seg]
        ltv   = row.get("ltv_per_customer", row["total_revenue"] / row["customer_count"])
        churn = row["churn_rate"] * 100
        count = row["customer_count"]
        ax.scatter(
            churn, ltv,
            s=count / 3,
            color=colors[i],
            alpha=0.7,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            f"{seg}\n(n={count:,})",
            (churn, ltv),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.axvline(5, color="red", linestyle="--", linewidth=1.5, label="5% churn threshold")
    ax.set_xlabel("Churn Rate (%)", fontsize=11)
    ax.set_ylabel("LTV per Customer ($)", fontsize=11)
    ax.set_title("Segment Health: Churn Rate vs LTV\n(bubble size = customer count)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    p = output_dir / "segment_churn_vs_ltv_bubble.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    return paths


def plot_pivot_heatmap(pivot: pd.DataFrame, title: str, fname: str, output_dir: Path) -> Path:
    """Save a heatmap of a pivot table."""
    import seaborn as sns

    # Exclude margins row/column for the heatmap
    data = pivot.drop(
        index=[c for c in ["Total", "Overall"] if c in pivot.index],
        columns=[c for c in ["Total", "Overall"] if c in pivot.columns],
        errors="ignore",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    p = output_dir / fname
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Step 8 – Console report
# ---------------------------------------------------------------------------

def print_report(
    single: dict[str, pd.DataFrame],
    ranked: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    insights: list[dict],
) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("GROUPBY AGGREGATION & SEGMENT INSIGHTS — REPORT")
    print(sep)

    # dataset-wide warning
    overall_churn = single["by_customer_type"]["churn_rate"].mean()
    print(f"\n  ⚠  Dataset-wide average churn = {overall_churn*100:.1f}%")
    print(f"     This single number is MISLEADING. Segment breakdown below:")

    # single-dimension: by customer type
    print(f"\n  ── BY CUSTOMER TYPE (primary segmentation) ──")
    bt = ranked[["customer_count","customer_share_pct","total_revenue",
                 "revenue_share_pct","churn_rate","ltv_per_customer",
                 "churn_rank","revenue_rank"]]
    print(bt.to_string())

    # single-dimension: by product
    print(f"\n  ── BY PRODUCT ──")
    print(single["by_product"].to_string())

    # single-dimension: by region
    print(f"\n  ── BY REGION ──")
    print(single["by_region"].to_string())

    # pivot: revenue
    print(f"\n  ── PIVOT: REVENUE by CUSTOMER TYPE × PRODUCT ──")
    print(pivots["revenue_by_type_x_product"].to_string())

    # pivot: churn rate
    print(f"\n  ── PIVOT: CHURN RATE by CUSTOMER TYPE × PRODUCT ──")
    print(pivots["churn_rate_by_type_x_product"].to_string())

    # actionable insights
    print(f"\n  ── ACTIONABLE SEGMENT INSIGHTS ──")
    for ins in insights:
        print(f"\n  [{ins['status']}]  {ins['segment']}")
        print(f"    Churn rate      : {ins['churn_rate_pct']:.1f}%  "
              f"(rank {ins['churn_rank']} — lower = healthier)")
        print(f"    Revenue share   : {ins['revenue_share_pct']:.1f}%  "
              f"(rank {ins['revenue_rank']})")
        print(f"    Customer share  : {ins['customer_share_pct']:.1f}%")
        print(f"    LTV/customer    : ${ins['ltv_per_customer']:,.0f}")
        print(f"    Action          : {ins['action']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 9 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    single: dict[str, pd.DataFrame],
    multi: dict[str, pd.DataFrame],
    pivots: dict[str, pd.DataFrame],
    ranked: pd.DataFrame,
    insights: list[dict],
    chart_paths: list[Path],
    output_dir: Path,
) -> None:
    """
    Files created
    -------------
    output/segment_analysis/segment_by_customer_type.csv
    output/segment_analysis/segment_by_product.csv
    output/segment_analysis/segment_by_region.csv
    output/segment_analysis/segment_by_type_x_product.csv
    output/segment_analysis/segment_by_type_x_region.csv
    output/segment_analysis/pivot_revenue_type_x_product.csv
    output/segment_analysis/pivot_churn_type_x_product.csv
    output/segment_analysis/segment_ranked.csv
    output/segment_analysis/report.json
    output/segment_analysis/*.png  (3 charts + 2 pivot heatmaps)
    """
    # CSV tables
    single["by_customer_type"].to_csv(output_dir / "segment_by_customer_type.csv")
    single["by_product"].to_csv(output_dir / "segment_by_product.csv")
    single["by_region"].to_csv(output_dir / "segment_by_region.csv")
    multi["by_type_x_product"].to_csv(output_dir / "segment_by_type_x_product.csv", index=False)
    multi["by_type_x_region"].to_csv(output_dir / "segment_by_type_x_region.csv", index=False)
    pivots["revenue_by_type_x_product"].to_csv(output_dir / "pivot_revenue_type_x_product.csv")
    pivots["churn_rate_by_type_x_product"].to_csv(output_dir / "pivot_churn_type_x_product.csv")
    ranked.to_csv(output_dir / "segment_ranked.csv")

    # JSON report
    report = {
        "segment_insights": insights,
        "plots": [str(p.name) for p in chart_paths],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)

    print(f"\n[save_outputs] All tables and report saved to {output_dir}/")
    print(f"[save_outputs] Charts: {[p.name for p in chart_paths]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GroupBy Aggregation & Segment Insights — NexPulse"
    )
    parser.add_argument("--input", dest="input_path",
                        help="Path to a CSV file. Omit to use the built-in demo dataset.")
    parser.add_argument("--demo", action="store_true",
                        help="Force the built-in demo dataset.")
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        print("Using built-in demo dataset (3,000 rows).")
        return build_demo_data(), "demo dataset"
    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df, str(path)


def main() -> None:
    args = parse_args()

    df, source_label = load_input(args)
    print(f"\nSource : {source_label}")
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Cols   : {list(df.columns)}\n")

    output_dir = Path("output/segment_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single-dimension aggregations
    print("[1/6] Single-dimension groupby (.agg) …")
    single = single_dimension_agg(df)

    # 2. Multi-dimension aggregations
    print("[2/6] Multi-dimension groupby (two keys) …")
    multi = multi_dimension_agg(df)

    # 3. Transform — broadcast back to rows
    print("[3/6] .transform() — adding segment benchmarks to each row …")
    df = add_segment_benchmarks(df)
    above_avg_pct = df["above_segment_avg"].mean() * 100
    print(f"  {above_avg_pct:.1f}% of customers are above their segment revenue average")

    # 4. Pivot tables
    print("[4/6] Building pivot tables …")
    pivots = build_pivot_tables(df)

    # 5. Rank segments
    print("[5/6] Ranking segments …")
    ranked = rank_segments(single["by_customer_type"])

    # 6. Actionable insights
    print("[6/6] Generating actionable segment insights …")
    insights = generate_segment_insights(ranked)

    # 7. Visualisations
    chart_paths = plot_segment_overview(ranked, insights, output_dir)
    chart_paths.append(
        plot_pivot_heatmap(
            pivots["revenue_by_type_x_product"],
            "Revenue by Customer Type × Product ($)",
            "pivot_heatmap_revenue.png",
            output_dir,
        )
    )
    chart_paths.append(
        plot_pivot_heatmap(
            pivots["churn_rate_by_type_x_product"],
            "Churn Rate by Customer Type × Product",
            "pivot_heatmap_churn.png",
            output_dir,
        )
    )

    # 8. Console report
    print_report(single, ranked, pivots, insights)

    # 9. Persist
    save_outputs(single, multi, pivots, ranked, insights, chart_paths, output_dir)

    print(f"\nSegment analysis complete.")
    print(f"  Segments analyzed   : {len(ranked)}")
    print(f"  Aggregation tables  : {len(single) + len(multi) + len(pivots)}")
    print(f"  Charts generated    : {len(chart_paths)}")
    print(f"  Insights surfaced   : {len(insights)}")


if __name__ == "__main__":
    main()
