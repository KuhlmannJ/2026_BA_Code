#########
#
# "8B_kpi_20260612.csv"     ran with BATCH_SIZE=1000 (lit. infinity)
# "all_8B_kpi_20260613.csv" ran with BATCH_SIZE=8
### Smaller batches reduce GPU memory pressure (cache efficiency) and improve CPU-GPU interleaving
### CPU cased an 20s idletime on GPU. With smaller batch the CPU preprocesses one batch while GPU computes the previous one.
#########

import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

FILES = {
    "8B": HERE / "8B_kpi_20260612.csv",
    "all_8B": HERE / "all_8B_kpi_20260613.csv",
}

NUMERIC_COLS = ["elapsed_s", "s_per_page", "peak_vram_gb", "peak_ram_gb", "file_size_mb"]

frames = {}
for model, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip() # Remove whitespaces there for easier reading the raw files
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
summary_df.to_csv(HERE / "ALLvs8B_summary_stats.csv")
print(f"\n  -> Saved: ALLvs8B_summary_stats.csv")