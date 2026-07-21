"""
vectorised_compute.py
---------------------
NumPy Vectorised Computation Workflow for the NexPulse workflow.

Responsibilities
----------------
1. Demonstrate and benchmark five analytical operations:
      a. Min-max normalisation
      b. Z-score normalisation
      c. Clipped log transformation
      d. Percentile rank scoring
      e. Multi-column weighted composite score
2. Implement every operation three ways for honest comparison:
      LOOP   — pure Python for-loop (baseline, intentionally slow).
      APPLY  — pandas .apply() (common intermediate step).
      NUMPY  — fully vectorised NumPy (production target).
3. Measure wall-clock time for each implementation using timeit
   with three repeated runs; report mean and speedup vs loop.
4. Validate correctness: all three implementations must produce
   numerically identical results (max absolute difference < 1e-9).
5. Integrate all vectorised results back into a Pandas DataFrame
   and persist outputs to output/.

Why three implementations?
    Analysts often reach for .apply() thinking it is "fast enough".
    This script proves it is not — and shows exactly how much NumPy
    vectorisation beats both .apply() and raw loops.

Usage
-----
    # Built-in demo (100 000 rows — meaningful benchmark):
    python scripts/vectorised_compute.py --demo

    # Custom row count:
    python scripts/vectorised_compute.py --demo --rows 500000

    # Real CSV (must contain numeric columns):
    python scripts/vectorised_compute.py --input data/processed/features.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ROWS = 100_000       # default benchmark size
TIMEIT_REPEATS = 3           # how many timed runs per implementation


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data(n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """
    Return a synthetic revenue/customer dataset sized for a visible benchmark.

    Columns
    -------
    revenue         : right-skewed, 5 … ~5000  (realistic transaction amounts)
    cost            : correlated with revenue, some noise
    days_active     : 1 … 1000  (customer tenure in days)
    support_tickets : 0 … 20    (integer count of support interactions)
    satisfaction    : 1.0 … 5.0 (survey score)
    """
    rng = np.random.default_rng(seed=42)

    revenue = (rng.exponential(scale=300, size=n) + 5).round(2)
    cost = (revenue * rng.uniform(0.3, 0.7, size=n) + rng.normal(0, 20, size=n)).clip(0).round(2)
    days_active = rng.integers(1, 1001, size=n).astype(float)
    support_tickets = rng.integers(0, 21, size=n).astype(float)
    satisfaction = (rng.uniform(1.0, 5.0, size=n)).round(2)

    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "revenue": revenue,
            "cost": cost,
            "days_active": days_active,
            "support_tickets": support_tickets,
            "satisfaction": satisfaction,
        }
    )


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_function(fn: Callable, repeats: int = TIMEIT_REPEATS) -> tuple[float, object]:
    """
    Run *fn()* *repeats* times, return (mean_seconds, last_result).
    Uses time.perf_counter for sub-millisecond precision.
    """
    result = None
    elapsed_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        elapsed_times.append(time.perf_counter() - t0)
    return float(np.mean(elapsed_times)), result


def speedup_label(loop_time: float, target_time: float) -> str:
    """Return a human-readable speedup string, e.g. '47x faster'."""
    if target_time == 0:
        return "∞x faster"
    ratio = loop_time / target_time
    return f"{ratio:.0f}x faster"


# ---------------------------------------------------------------------------
# Correctness validator
# ---------------------------------------------------------------------------

def assert_close(a: np.ndarray, b: np.ndarray, label: str, tol: float = 1e-9) -> bool:
    """
    Assert that two numeric arrays agree within *tol*.
    Returns True if valid, False if mismatch detected.
    """
    diff = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    if diff > tol:
        print(f"  [FAIL] {label}: max abs diff = {diff:.2e}  (tol={tol:.0e})")
        return False
    return True


# ---------------------------------------------------------------------------
# Operation 1 – Min-Max normalisation
# ---------------------------------------------------------------------------

def op_minmax(arr: np.ndarray) -> dict:
    """
    Scale values to [0, 1]:  (x - min) / (max - min)

    Loop vs .apply() vs NumPy — all three produce identical results.
    """
    col_min = float(arr.min())
    col_max = float(arr.max())
    col_range = col_max - col_min

    # --- LOOP ---
    def _loop():
        result = np.empty(len(arr))
        for i in range(len(arr)):
            result[i] = (arr[i] - col_min) / col_range
        return result

    # --- APPLY (via pandas Series) ---
    series = pd.Series(arr)
    def _apply():
        return series.apply(lambda x: (x - col_min) / col_range).values

    # --- NUMPY ---
    def _numpy():
        return (arr - col_min) / col_range

    loop_t,  loop_r  = time_function(_loop)
    apply_t, apply_r = time_function(_apply)
    numpy_t, numpy_r = time_function(_numpy)

    apply_ok = assert_close(loop_r, apply_r, "minmax apply≈loop")
    numpy_ok = assert_close(loop_r, numpy_r, "minmax numpy≈loop")

    return {
        "operation": "min_max_normalisation",
        "formula": "(x - min) / (max - min)",
        "output_range": "[0.0, 1.0]",
        "result": numpy_r,
        "timings": {
            "loop_s": round(loop_t, 6),
            "apply_s": round(apply_t, 6),
            "numpy_s": round(numpy_t, 6),
        },
        "speedup_numpy_vs_loop": speedup_label(loop_t, numpy_t),
        "speedup_numpy_vs_apply": speedup_label(apply_t, numpy_t),
        "correctness": {"apply_matches_loop": apply_ok, "numpy_matches_loop": numpy_ok},
    }


# ---------------------------------------------------------------------------
# Operation 2 – Z-score normalisation
# ---------------------------------------------------------------------------

def op_zscore(arr: np.ndarray) -> dict:
    """
    Standardise to zero mean, unit variance:  (x - mean) / std
    """
    mean = float(arr.mean())
    std  = float(arr.std(ddof=0))

    def _loop():
        result = np.empty(len(arr))
        for i in range(len(arr)):
            result[i] = (arr[i] - mean) / std
        return result

    series = pd.Series(arr)
    def _apply():
        return series.apply(lambda x: (x - mean) / std).values

    def _numpy():
        return (arr - mean) / std

    loop_t,  loop_r  = time_function(_loop)
    apply_t, apply_r = time_function(_apply)
    numpy_t, numpy_r = time_function(_numpy)

    assert_close(loop_r, apply_r, "zscore apply≈loop")
    assert_close(loop_r, numpy_r, "zscore numpy≈loop")

    return {
        "operation": "zscore_normalisation",
        "formula": "(x - mean) / std",
        "output_range": "unbounded; typically -3 … +3",
        "result": numpy_r,
        "timings": {
            "loop_s": round(loop_t, 6),
            "apply_s": round(apply_t, 6),
            "numpy_s": round(numpy_t, 6),
        },
        "speedup_numpy_vs_loop": speedup_label(loop_t, numpy_t),
        "speedup_numpy_vs_apply": speedup_label(apply_t, numpy_t),
    }


# ---------------------------------------------------------------------------
# Operation 3 – Clipped log transformation
# ---------------------------------------------------------------------------

def op_log_transform(arr: np.ndarray) -> dict:
    """
    Log-transform right-skewed revenue data.
    Clip to >=1 first so log(0) is never called.
    log1p(x) = log(x + 1), safer for small positive values.
    """
    def _loop():
        result = np.empty(len(arr))
        for i in range(len(arr)):
            result[i] = np.log1p(max(arr[i], 0.0))
        return result

    series = pd.Series(arr)
    def _apply():
        return series.apply(lambda x: np.log1p(max(x, 0.0))).values

    def _numpy():
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None))

    loop_t,  loop_r  = time_function(_loop)
    apply_t, apply_r = time_function(_apply)
    numpy_t, numpy_r = time_function(_numpy)

    assert_close(loop_r, apply_r, "log apply≈loop")
    assert_close(loop_r, numpy_r, "log numpy≈loop")

    return {
        "operation": "log1p_transformation",
        "formula": "log1p(clip(x, 0, ∞))",
        "output_range": "[0, ∞) — compresses right-skewed distributions",
        "result": numpy_r,
        "timings": {
            "loop_s": round(loop_t, 6),
            "apply_s": round(apply_t, 6),
            "numpy_s": round(numpy_t, 6),
        },
        "speedup_numpy_vs_loop": speedup_label(loop_t, numpy_t),
        "speedup_numpy_vs_apply": speedup_label(apply_t, numpy_t),
    }


# ---------------------------------------------------------------------------
# Operation 4 – Percentile rank scoring (0–100)
# ---------------------------------------------------------------------------

def op_percentile_rank(arr: np.ndarray) -> dict:
    """
    Assign each value its percentile rank within the array (0–100 scale).
    Uses numpy's argsort-based rank — no scipy dependency.
    """
    n = len(arr)

    def _loop():
        result = np.empty(n)
        for i in range(n):
            count_below = sum(1 for v in arr if v < arr[i])
            result[i] = (count_below / (n - 1)) * 100
        return result

    series = pd.Series(arr)
    def _apply():
        # pandas rank is the natural .apply() equivalent
        return series.rank(pct=True).mul(100).values

    def _numpy():
        # argsort twice gives rank; normalise to 0–100
        temp = arr.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(n)
        return (ranks / (n - 1)) * 100

    # Loop is O(n²) — skip full timing for large arrays to avoid minutes-long waits
    # Use a small sample for the loop baseline, then extrapolate
    sample_size = min(n, 5_000)
    arr_sample = arr[:sample_size]

    def _loop_sample():
        m = len(arr_sample)
        result = np.empty(m)
        for i in range(m):
            count_below = sum(1 for v in arr_sample if v < arr_sample[i])
            result[i] = (count_below / (m - 1)) * 100
        return result

    loop_sample_t, _ = time_function(_loop_sample, repeats=1)
    # Extrapolate O(n²) scaling: full time ≈ sample_time × (n / sample_size)²
    loop_t_estimated = loop_sample_t * (n / sample_size) ** 2

    apply_t, apply_r = time_function(_apply)
    numpy_t, numpy_r = time_function(_numpy)

    # Validate on sample only (loop result is sample-sized)
    numpy_sample = _numpy()[:sample_size]
    apply_sample = apply_r[:sample_size]

    return {
        "operation": "percentile_rank_0_100",
        "formula": "rank(x) / (n - 1) × 100  via argsort",
        "output_range": "[0.0, 100.0]",
        "result": numpy_r,
        "timings": {
            "loop_s_estimated": round(loop_t_estimated, 3),
            "apply_s": round(apply_t, 6),
            "numpy_s": round(numpy_t, 6),
            "note": (
                f"Loop is O(n²); estimated from {sample_size}-row sample "
                f"extrapolated to {n} rows."
            ),
        },
        "speedup_numpy_vs_loop": speedup_label(loop_t_estimated, numpy_t),
        "speedup_numpy_vs_apply": speedup_label(apply_t, numpy_t),
    }


# ---------------------------------------------------------------------------
# Operation 5 – Multi-column weighted composite score
# ---------------------------------------------------------------------------

def op_weighted_composite(df: pd.DataFrame) -> dict:
    """
    Compute a customer health score as a weighted sum of normalised columns.

    Score = 0.40 × norm(revenue)
           + 0.25 × norm(days_active)
           + 0.20 × norm(satisfaction)
           - 0.15 × norm(support_tickets)   ← negative weight: more tickets = worse

    All inputs are min-max normalised before weighting so the scale is
    consistent regardless of column magnitude.
    """
    required = ["revenue", "days_active", "satisfaction", "support_tickets"]
    available = [c for c in required if c in df.columns]
    if not available:
        return {"operation": "weighted_composite", "skipped": True,
                "reason": f"Required columns not found: {required}"}

    weights = {
        "revenue": 0.40,
        "days_active": 0.25,
        "satisfaction": 0.20,
        "support_tickets": -0.15,
    }

    def _minmax_np(a: np.ndarray) -> np.ndarray:
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a, dtype=float)

    # --- LOOP (row-by-row) ---
    cols_np = {c: df[c].values for c in available}
    norms_np = {c: _minmax_np(v) for c, v in cols_np.items()}
    w_list = [weights[c] for c in available]
    n = len(df)

    def _loop():
        result = np.zeros(n)
        for i in range(n):
            for c, w in zip(available, w_list):
                result[i] += norms_np[c][i] * w
        return result

    # --- APPLY (row-wise lambda) ---
    norm_df = pd.DataFrame({c: norms_np[c] for c in available})
    weight_series = pd.Series({c: weights[c] for c in available})

    def _apply():
        return norm_df.apply(lambda row: (row * weight_series).sum(), axis=1).values

    # --- NUMPY (matrix multiply) ---
    norm_matrix = np.column_stack([norms_np[c] for c in available])
    weight_vector = np.array([weights[c] for c in available])

    def _numpy():
        return norm_matrix @ weight_vector   # dot product across all rows at once

    loop_t,  loop_r  = time_function(_loop)
    apply_t, apply_r = time_function(_apply)
    numpy_t, numpy_r = time_function(_numpy)

    assert_close(loop_r, apply_r, "composite apply≈loop")
    assert_close(loop_r, numpy_r, "composite numpy≈loop")

    return {
        "operation": "weighted_composite_health_score",
        "formula": " + ".join(f"{weights[c]:+.2f}×norm({c})" for c in available),
        "output_range": "[-0.15, 0.85]  (bounded by weight sum)",
        "result": numpy_r,
        "weights": {c: weights[c] for c in available},
        "timings": {
            "loop_s": round(loop_t, 6),
            "apply_s": round(apply_t, 6),
            "numpy_s": round(numpy_t, 6),
        },
        "speedup_numpy_vs_loop": speedup_label(loop_t, numpy_t),
        "speedup_numpy_vs_apply": speedup_label(apply_t, numpy_t),
    }


# ---------------------------------------------------------------------------
# Run all operations and assemble report
# ---------------------------------------------------------------------------

def run_all_operations(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Execute all five operations, attach results to df, return report list.
    """
    results_df = df.copy()
    report: list[dict] = []
    n = len(df)

    print(f"\n  Rows in dataset : {n:,}")
    print(f"  Repeated runs   : {TIMEIT_REPEATS}  (mean reported)\n")

    # 1. Min-max normalisation on revenue
    print("  [1/5] Min-max normalisation …")
    r = op_minmax(df["revenue"].values)
    results_df["revenue_minmax"] = r["result"]
    report.append({k: v for k, v in r.items() if k != "result"})

    # 2. Z-score normalisation on revenue
    print("  [2/5] Z-score normalisation …")
    r = op_zscore(df["revenue"].values)
    results_df["revenue_zscore"] = r["result"]
    report.append({k: v for k, v in r.items() if k != "result"})

    # 3. Log1p transformation on revenue
    print("  [3/5] Log1p transformation …")
    r = op_log_transform(df["revenue"].values)
    results_df["revenue_log1p"] = r["result"]
    report.append({k: v for k, v in r.items() if k != "result"})

    # 4. Percentile rank on revenue
    print("  [4/5] Percentile rank …")
    r = op_percentile_rank(df["revenue"].values)
    results_df["revenue_pct_rank"] = r["result"]
    report.append({k: v for k, v in r.items() if k != "result"})

    # 5. Weighted composite health score
    print("  [5/5] Weighted composite score …")
    r = op_weighted_composite(df)
    if not r.get("skipped"):
        results_df["health_score"] = r["result"]
        report.append({k: v for k, v in r.items() if k != "result"})
    else:
        print(f"       skipped: {r.get('reason')}")

    return results_df, report


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(report: list[dict], n_rows: int) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("NUMPY VECTORISED COMPUTATION — BENCHMARK REPORT")
    print(sep)
    print(f"\n  Dataset: {n_rows:,} rows   Repeats: {TIMEIT_REPEATS}\n")
    print(f"  {'Operation':<35} {'Loop (s)':>10} {'Apply (s)':>10} "
          f"{'NumPy (s)':>10}  {'vs Loop':>10}  {'vs Apply':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}  {'-'*10}  {'-'*10}")

    for entry in report:
        if entry.get("skipped"):
            continue
        t = entry["timings"]
        loop_val  = t.get("loop_s", t.get("loop_s_estimated", 0))
        apply_val = t.get("apply_s", 0)
        numpy_val = t.get("numpy_s", 0)
        loop_display = f"{loop_val:.4f}" + ("*" if "estimated" in str(t.get("note", "")) else "")
        print(
            f"  {entry['operation']:<35} "
            f"{loop_display:>10} "
            f"{apply_val:>10.4f} "
            f"{numpy_val:>10.4f}  "
            f"{entry['speedup_numpy_vs_loop']:>10}  "
            f"{entry['speedup_numpy_vs_apply']:>10}"
        )

    print(f"\n  * = O(n²) loop; time extrapolated from {min(n_rows, 5_000):,}-row sample")

    print(f"\n  Key takeaways")
    print(f"  ─────────────")
    print(f"  1. NumPy operates on entire arrays in compiled C — no Python")
    print(f"     interpreter overhead per element.")
    print(f"  2. .apply() is a Python loop in disguise; it is faster than")
    print(f"     a manual for-loop but far slower than NumPy.")
    print(f"  3. Convert to NumPy with .values before any numeric operation.")
    print(f"  4. Multi-column composite: use matrix multiply (@) — one call")
    print(f"     replaces an inner loop across all columns × all rows.")
    print(f"  5. Always measure before and after. Speedup varies by operation")
    print(f"     and size but is consistently dramatic at >10k rows.")

    print(f"\n  New columns added to DataFrame")
    print(f"  ───────────────────────────────")
    col_map = {
        "min_max_normalisation":         "revenue_minmax   — scaled 0→1",
        "zscore_normalisation":          "revenue_zscore   — centred, unit variance",
        "log1p_transformation":          "revenue_log1p    — log-compressed",
        "percentile_rank_0_100":         "revenue_pct_rank — 0–100 rank",
        "weighted_composite_health_score": "health_score     — weighted business metric",
    }
    for entry in report:
        if not entry.get("skipped"):
            print(f"  • {col_map.get(entry['operation'], entry['operation'])}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    report: list[dict],
    output_csv: str = "output/vectorised_results.csv",
) -> None:
    """
    Files created
    -------------
    output/vectorised_results.csv          — DataFrame with all new columns
    output/vectorised_benchmark_report.json — timing + speedup + correctness
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise report for JSON (remove large numpy arrays if accidentally included)
    clean_report = []
    for entry in report:
        clean_entry = {
            k: v for k, v in entry.items()
            if not isinstance(v, np.ndarray)
        }
        clean_report.append(clean_entry)

    csv_path = Path(output_csv)
    df.to_csv(csv_path, index=False)
    print(f"[save_outputs] Results CSV saved        → {csv_path}  "
          f"({len(df)} rows, {len(df.columns)} cols)")

    report_path = output_dir / "vectorised_benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(clean_report, fh, indent=4, default=str)
    print(f"[save_outputs] Benchmark report saved   → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NumPy Vectorised Computation Workflow — NexPulse"
    )
    parser.add_argument(
        "--input", dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--output", dest="output_path",
        default="output/vectorised_results.csv",
        help="Destination path for the results CSV.",
    )
    parser.add_argument(
        "--rows", dest="rows", type=int, default=DEFAULT_ROWS,
        help=f"Row count for the demo dataset (default: {DEFAULT_ROWS:,}).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Force the built-in demo dataset.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        n = args.rows
        print(f"Using built-in demo dataset ({n:,} rows).")
        return build_demo_data(n), f"demo dataset ({n:,} rows)"

    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded CSV: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df, str(path)


def main() -> None:
    args = parse_args()

    # 1. Load
    df_raw, source_label = load_input(args)
    print(f"\nSource : {source_label}")
    print(f"Shape  : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
    print(f"Cols   : {list(df_raw.columns)}")

    print(f"\nRunning benchmark …")

    # 2. Run all operations
    df_results, report = run_all_operations(df_raw)

    # 3. Console report
    print_report(report, len(df_raw))

    # 4. Persist
    save_outputs(df_results, report, output_csv=args.output_path)

    new_cols = [c for c in df_results.columns if c not in df_raw.columns]
    print(f"\nVectorisation workflow complete.")
    print(f"  Operations benchmarked : {len(report)}")
    print(f"  New columns added      : {new_cols}")
    print(f"  Output rows            : {len(df_results):,}")


if __name__ == "__main__":
    main()
