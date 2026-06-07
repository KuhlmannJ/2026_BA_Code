import json
import re
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def banner(text: str) -> None:
    width = 72
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


#### GLOBAL VARIABLES ####

CHECKLIST_PATH = Path("checklist.csv")
RAW_DIR        = Path("baselines/baseline_a_frontier_model/raw")
RESULTS_DIR    = Path("baselines/baseline_a_frontier_model/eval")

SCOPE_MAP: dict[str, str] = {
    "scope_1":                "1",
    "scope_2_market_based":   "2mb",
    "scope_2_location_based": "2lb",
    "scope_3":                "3",
}

# Multipliers: unit string → t CO2e.
# "Mt / MT" in sustainability reports = metric ton (= tonne), NOT megaton.
# "m t" (with space) = million tonnes = 1e6 t.
UNIT_TO_T_CO2E: dict[str, float] = {
    # straight tonnes
    "t CO2e": 1.0,   "t CO2": 1.0,   "t": 1.0,
    "tCO2e":  1.0,   "tCO2":  1.0,
    "CO2e in t": 1.0, "CO2e t": 1.0, "CO2e in tons": 1.0,
    "CO2e emissions (tonnes)": 1.0,
    "metric tonnes of CO2 equivalent": 1.0,
    "tonnes of CO2 equivalent": 1.0,
    "emissions": 1.0,
    # metric-ton abbreviations (= 1 t)
    "Mt CO2e": 1.0,  "MT CO2e": 1.0, "MTCO2e": 1.0,
    "MtCO2e":  1.0,  "mt CO2e": 1.0,
    # kilotonnes (1e3 t)
    "kt CO2e": 1e3,  "kt CO2": 1e3,  "kt": 1e3,
    "ktons CO2": 1e3,
    "CO2 emissions from energy consumption (in 1,000 t)": 1e3,
    "CO2 emissions (1,000 tonnes)": 1e3,
    # million tonnes (1e6 t)
    "m t CO2e": 1e6, "m t CO2": 1e6,
    "million t": 1e6, "million t CO2": 1e6,
    "MM MT CO2e": 1e6,
    # kilograms (1e-3 t)
    "kg CO2e": 1e-3,
    # US short ton ≈ 0.9072 t
    "CO2e (US Tons)": 0.907185, "CO2e (US tons)": 0.907185,
    "ton CO2": 0.907185,        "ton CO2e": 0.907185,
}

UNIT_TO_T_CO2E.update({
    # additional variants seen in the wild
    "tonnes CO2e": 1.0,  "tonnes CO2": 1.0,  "tonnes": 1.0,
    "Tonnes CO2e": 1.0,  "Tonne CO2e": 1.0,
    "metric tons CO2e": 1.0, "metric tons CO2": 1.0, "metric tons": 1.0,
    "Metric tons CO2e": 1.0,
    "metric ton CO2e": 1.0,
    "tonne CO2e": 1.0,   "tonne CO2": 1.0,   "tonne": 1.0,
    "CO2 equivalent tonnes": 1.0,
    "CO2e": 1.0,
})

MATCH_TOL = 0.01  # |pred_t - gold_t| / gold_t ≤ 1 % → match


def norm_name(s: str) -> str:
    """Normalize a report name for fuzzy gold-standard lookup.
    Lowercases and collapses any run of non-alphanumeric chars to a single '_'.
    """
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def to_t_co2e(value: float, unit: str) -> float | None:
    """Return value converted to t CO2e, or None if unit is unknown."""
    factor = UNIT_TO_T_CO2E.get(unit)
    return value * factor if factor is not None else None


def evaluate_json(json_path: Path, gold: pd.DataFrame,
                  gold_norm_index: dict[str, str]) -> dict:
    """Evaluate one extracted JSON file against gold rows for that report."""
    # Always use the filename stem as the canonical name — the internal
    # report_name field set by Claude can be mangled (e.g. "_pdf" suffix).
    report_name = json_path.stem

    try:
        with open(json_path) as fh:
            raw = fh.read().strip()
        if not raw:
            return {
                "report":  report_name,
                "error":   "Empty JSON file",
                "results": [],
                "summary": {},
            }
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "report":  report_name,
            "error":   f"JSON parse error: {exc}",
            "results": [],
            "summary": {},
        }

    # Fuzzy name lookup: normalize stem → find matching gold report_name
    stem_norm   = norm_name(report_name)
    gold_pdf    = gold_norm_index.get(stem_norm)
    if gold_pdf is None:
        return {
            "report":  report_name,
            "error":   f"No gold rows found (normalized key: '{stem_norm}')",
            "results": [],
            "summary": {},
        }
    gold_report = gold[gold["report_name"] == gold_pdf].copy()

    emissions = data.get("emissions", {})
    results: list[dict] = []

    for scope_key, scope_code in SCOPE_MAP.items():
        scope_data = emissions.get(scope_key, {})
        for year_str, entries in scope_data.items():
            try:
                year = int(year_str)
            except ValueError:
                continue
            if not entries:
                continue

            # Claude may return multiple items per cell; take first
            entry     = entries[0]
            ext_value = entry.get("value")
            ext_unit  = entry.get("unit", "")
            ext_label = entry.get("label", "")
            if ext_value is None:
                continue

            ext_t = to_t_co2e(float(ext_value), ext_unit)

            # Match against every gold row for this (year, scope)
            mask      = (gold_report["year"] == year) & (gold_report["scope"] == scope_code)
            gold_rows = gold_report[mask]

            gold_entries: list[dict] = []
            best_rel_err: float | None = None

            for _, row in gold_rows.iterrows():
                gold_t  = to_t_co2e(float(row["value"]), str(row["unit_normalized"]))
                rel_err: float | None = None
                if gold_t and ext_t is not None:
                    rel_err = abs(ext_t - gold_t) / abs(gold_t)

                gold_entries.append({
                    "value":        row["value"],
                    "unit":         row["unit"],
                    "unit_norm":    row["unit_normalized"],
                    "value_t_co2e": gold_t,
                    "metric_name":  row["metric_name"],
                    "display_type": row["display_type"],
                    "page":         row["page"],
                    "rel_error":    round(rel_err, 6) if rel_err is not None else None,
                })
                if rel_err is not None:
                    if best_rel_err is None or rel_err < best_rel_err:
                        best_rel_err = rel_err

            is_match = best_rel_err is not None and best_rel_err <= MATCH_TOL
            in_gold  = len(gold_entries) > 0

            results.append({
                "year":             year,
                "scope":            scope_code,
                "extracted_value":  ext_value,
                "extracted_unit":   ext_unit,
                "extracted_label":  ext_label,
                "extracted_t_co2e": round(ext_t, 4) if ext_t is not None else None,
                "unit_known":       ext_t is not None,
                "in_gold":          in_gold,
                "gold_entries":     gold_entries,
                "best_rel_error":   round(best_rel_err, 6) if best_rel_err is not None else None,
                "is_match":         is_match,
            })

    # Gold pairs Claude missed entirely
    extracted_keys = {(r["year"], r["scope"]) for r in results}
    gold_keys      = set(zip(gold_report["year"].astype(int), gold_report["scope"]))
    missing        = sorted(gold_keys - extracted_keys)

    total_extracted = len(results)
    match_count     = sum(1 for r in results if r["is_match"])
    matched_errors  = [r["best_rel_error"] for r in results
                       if r["is_match"] and r["best_rel_error"] is not None]
    mean_err = sum(matched_errors) / len(matched_errors) if matched_errors else None

    summary = {
        "total_extracted":        total_extracted,
        "in_gold":                sum(1 for r in results if r["in_gold"]),
        "not_in_gold":            sum(1 for r in results if not r["in_gold"]),
        "matched":                match_count,
        "precision":              round(match_count / total_extracted, 4) if total_extracted else None,
        "recall":                 round(match_count / len(gold_keys),   4) if gold_keys else None,
        "mean_rel_error_matched": round(mean_err, 6) if mean_err is not None else None,
        "missing_in_extraction":  [{"year": y, "scope": s} for y, s in missing],
    }

    return {
        "report":  report_name,
        "results": sorted(results, key=lambda r: (r["scope"], r["year"])),
        "summary": summary,
    }


def print_summary(eval_result: dict) -> None:
    report = eval_result.get("report", "?")
    error  = eval_result.get("error")
    if error:
        print(f"  ERROR [{report}]: {error}")
        return

    s = eval_result["summary"]
    print(f"\n  Report                   : {report}")
    print(f"  Extracted (year×scope)   : {s['total_extracted']}")
    print(f"  In gold standard         : {s['in_gold']}")
    print(f"  Not in gold              : {s['not_in_gold']}")
    print(f"  Matches (≤1 % rel. err)  : {s['matched']}")
    prec = f"{s['precision']:.2%}" if s["precision"] is not None else "n/a"
    rec  = f"{s['recall']:.2%}"    if s["recall"]    is not None else "n/a"
    print(f"  Precision                : {prec}")
    print(f"  Recall                   : {rec}")
    if s["mean_rel_error_matched"] is not None:
        print(f"  Mean rel. err (matched)  : {s['mean_rel_error_matched']:.4%}")
    if s["missing_in_extraction"]:
        pairs = ", ".join(f"{m['year']}/{m['scope']}" for m in s["missing_in_extraction"])
        print(f"  Not extracted (yr/scope) : {pairs}")

    print()
    print(f"  {'Year':<6} {'Scope':<5} {'Extracted':>15} {'Unit':<12} {'t CO2e':>15} {'OK':>4} {'Rel.Err':>10}")
    print(f"  {'-'*6} {'-'*5} {'-'*15} {'-'*12} {'-'*15} {'-'*4} {'-'*10}")
    for r in eval_result["results"]:
        t_str = f"{r['extracted_t_co2e']:>15,.1f}" if r["extracted_t_co2e"] is not None else f"{'?':>15}"
        icon  = "✓" if r["is_match"] else ("–" if not r["in_gold"] else "✗")
        err_s = f"{r['best_rel_error']:.4%}" if r["best_rel_error"] is not None else "n/a"
        print(f"  {r['year']:<6} {r['scope']:<5} {r['extracted_value']:>15} {r['extracted_unit']:<12} {t_str} {icon:>4} {err_s:>10}")


def main() -> None:
    banner("Baseline A — Evaluation against Beck et al. Gold Standard")

    print(f"  Loading gold standard: {CHECKLIST_PATH}")
    gold = pd.read_csv(CHECKLIST_PATH)
    print(f"  Gold rows loaded     : {len(gold)}")

    # Build normalized name → original PDF name lookup
    gold_norm_index: dict[str, str] = {}
    for pdf_name in gold["report_name"].unique():
        key = norm_name(pdf_name.removesuffix(".pdf"))
        gold_norm_index[key] = pdf_name

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        print(f"\n  ERROR: No JSON files in {RAW_DIR}")
        return
    print(f"  JSON files found     : {len(json_files)} (in {RAW_DIR})\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for json_path in json_files:
        print(f"  Evaluating: {json_path.name}")
        result = evaluate_json(json_path, gold, gold_norm_index)
        print_summary(result)
        all_results.append(result)

        out_path = RESULTS_DIR / f"{result['report']}_eval.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"  → Saved: {out_path}\n")

    # Aggregate summary
    banner("Aggregate Summary")
    agg_extracted = sum(r["summary"].get("total_extracted", 0) for r in all_results if "summary" in r)
    agg_matched   = sum(r["summary"].get("matched", 0)         for r in all_results if "summary" in r)
    agg_gold_keys = sum(
        r["summary"].get("in_gold", 0) + len(r["summary"].get("missing_in_extraction", []))
        for r in all_results if "summary" in r
    )
    print(f"  Reports evaluated  : {len(all_results)}")
    print(f"  Total extracted    : {agg_extracted}")
    print(f"  Total matched      : {agg_matched}")
    if agg_extracted:
        print(f"  Overall precision  : {agg_matched / agg_extracted:.2%}")
    if agg_gold_keys:
        print(f"  Overall recall     : {agg_matched / agg_gold_keys:.2%}")

    agg_path = RESULTS_DIR / "aggregate_eval.json"
    with open(agg_path, "w") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\n  → Aggregate saved: {agg_path}")


if __name__ == "__main__":
    main()