import json, glob, pandas as pd

# ================== PIPELINE A — cost per condition ==================
# Sums the per-report usage logs A-04-fromClaude.py writes to
#   PipelineA/PipelineA-Answers/{condition}/logs/{report}.log
# and re-applies the same price constants.
IN_PRICE, OUT_PRICE = 2.97, 14.88          # $ per 1M tokens (as in A-04)

print("=== Pipeline A cost ===")
for cond in ["bare", "thinking", "thinking_system"]:
    logs = glob.glob(f"PipelineA/PipelineA-Answers/{cond}/logs/*.log")
    tin = tout = 0
    for f in logs:
        d = json.load(open(f))
        tin  += d["input_tokens"]
        tout += d["output_tokens"]
    n    = len(logs)
    cost = (tin / 1_000_000) * IN_PRICE + (tout / 1_000_000) * OUT_PRICE
    print(f"{cond:16s} n={n}  in={tin/1e6:.2f}M  out={tout/1e3:.1f}k  "
          f"total=${cost:.2f}  per_report=${cost/n:.3f}")


# ================== PIPELINE B — latency per model ==================
# mean / median duration and mean s/page from the per-run ***results.csv
# (columns: model, maxToken, report, duration, pages, t_inf/page) that
# B-03-HPC.py writes. Two runs (1st_/2nd_) give the reported range.
runs = {
    "32B-Instruct":       ["1st_Qwen3-VL-32B-Instruct",      "2nd_Qwen3-VL-32B-Instruct"],
    "30B-A3B (MoE)":      ["1st_Qwen3-VL-30B-A3B-Thinking",  "2nd_Qwen3-VL-30B-A3B-Thinking"],
    "32B-Thinking":       ["1st_Qwen3-VL-32B-Thinking",      "2nd_Qwen3-VL-32B-Thinking"],
    "32B-Thinking +GEPA": ["GEPA_Qwen3-VL-32B-Thinking"],
}

def rng(xs):                                # min-max across the runs, rounded
    xs = [round(x) for x in xs]
    return f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}-{max(xs)}"

print("\n=== Pipeline B latency (s per report) ===")
for label, dirs in runs.items():
    means, medians, spp = [], [], []
    for d in dirs:
        df = pd.read_csv(f"PipelineB/PipelineB-Answers/{d}/***results.csv")
        means.append(df["duration"].mean())
        medians.append(df["duration"].median())
        spp.append(df["t_inf/page"].mean())
    print(f"{label:20s} mean={rng(means):>8s}  median={rng(medians):>8s}  s/page={rng(spp):>6s}")
    
    
###################
### OUTPUT:
# === Pipeline A cost ===
# bare             n=52  in=1.08M  out=40.1k  total=$3.79  per_report=$0.073
# thinking         n=54  in=1.15M  out=40.1k  total=$4.01  per_report=$0.074
# thinking_system  n=54  in=1.56M  out=39.8k  total=$5.23  per_report=$0.097

# === Pipeline B latency (s per report) ===
# 32B-Instruct         mean=   40-46  median=   34-36  s/page=   5-6
# 30B-A3B (MoE)        mean= 254-270  median= 224-233  s/page= 35-37
# 32B-Thinking         mean= 337-347  median= 293-322  s/page= 46-47
# 32B-Thinking +GEPA   mean=     352  median=     301  s/page=    47
##################