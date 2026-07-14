"""
Compares KPI CSVs from three embedding model runs (3B / 4B / 8B).
Produces a console summary and writes comparison tables to:
  - summary_stats.csv   (per-model aggregate stats)
  - per_report.csv      (all three models side-by-side per report)
"""

import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

FILES = {
    "3B": HERE / "3B_kpi_20260612.csv",
    "4B": HERE / "4B_kpi_20260612.csv",
    "8B": HERE / "8B_kpi_20260612.csv",
}

NUMERIC_COLS = ["elapsed_s", "s_per_page", "peak_vram_gb", "peak_ram_gb", "file_size_mb"]

# ── Load ──────────────────────────────────────────────────────────────────────
frames = {}
for model, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Remove whitespaces there for easier reading the raw files
    df["model"] = model
    frames[model] = df
    
    

# ── Per-model summary stats ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY STATISTICS PER MODEL")
print("=" * 72)

agg_rows = []
for model, df in frames.items():
    n_reports = len(df)
    total_pages = df["pages"].sum()
    total_time_s = df["elapsed_s"].sum()
    embed_dim = df["embed_dim"].iloc[0]

    row = {
        "model":           model,
        "embed_dim":       embed_dim,
        "n_reports":       n_reports,
        "total_pages":     total_pages,
        "total_time_min":  round(total_time_s / 60, 2),
        "mean_s_per_page": round(df["s_per_page"].mean(), 4),
        "max_peak_vram_gb":    round(df["peak_vram_gb"].max(), 2),
        "max_peak_ram_gb":     round(df["peak_ram_gb"].max(), 2),
        "total_file_size_gb":  round(df["file_size_mb"].sum() / 1024, 3),
        "mean_file_size_mb":   round(df["file_size_mb"].mean(), 1),
    }
    agg_rows.append(row)

summary_df = pd.DataFrame(agg_rows).set_index("model")

# Transposed for readability in console
print(summary_df.T.to_string())
summary_df.to_csv(HERE / "summary_stats.csv")
print(f"\n  -> Saved: summary_stats.csv")




# ── Side-by-side per report ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  PER-REPORT COMPARISON  (s_per_page | peak_vram_gb | file_size_mb)")
print("=" * 72)

# Merge all models on report
merged = None
for model, df in frames.items():
    sub = df[["report", "pages", "s_per_page", "peak_vram_gb", "file_size_mb"]].copy()
    sub = sub.rename(columns={
        "s_per_page":    f"s/page_{model}",
        "peak_vram_gb":  f"vram_gb_{model}",
        "file_size_mb":  f"size_mb_{model}",
    })
    merged = sub if merged is None else merged.merge(sub.drop(columns="pages"), on="report")

# Reorder: group by metric, not by model
models = list(FILES.keys())
col_order = (
    ["report", "pages"]
    + [f"s/page_{m}"  for m in models]
    + [f"vram_gb_{m}" for m in models]
    + [f"size_mb_{m}" for m in models]
)
merged = merged[col_order]

merged.to_csv(HERE / "per_report.csv", index=False)
print(f"  -> Saved: per_report.csv  ({len(merged)} reports)\n")

###############
### OUTPUT:
# ========================================================================
#   SUMMARY STATISTICS PER MODEL
# ========================================================================
# model                      3B         4B         8B
# embed_dim           3072.0000  2560.0000  4096.0000
# n_reports             53.0000    53.0000    53.0000
# total_pages         4549.0000  4549.0000  4549.0000
# total_time_min        25.4000    19.8700    20.8800
# mean_s_per_page        0.3972     0.3288     0.3405
# max_peak_vram_gb     110.6100    65.2900    82.2800
# max_peak_ram_gb       21.8300    24.7800    26.3400
# total_file_size_gb    97.9260    35.0890    56.1410
# mean_file_size_mb   1892.0000   677.9000  1084.7000

#   -> Saved: summary_stats.csv

# ========================================================================
#   PER-REPORT COMPARISON  (s_per_page | peak_vram_gb | file_size_mb)
# ========================================================================
#   -> Saved: per_report.csv  (2 reports)
###############