import pandas as pd
from pathlib import Path

### Listing all downloadable reports
reports_downloaded = {p.name for p in Path("localdata/all_esg_reports").glob("*.pdf")}


### Loading Gold_Standard in full
gs = pd.read_csv("gold_standard.csv")

# Finding wrong field name in gs and mapping it
print("Mismatches:", reports_downloaded - set(gs["report_name"]))
gs["report_name"] = gs["report_name"].replace("viacomcbs_2020_report.pdf", "ViacomCBS_ESG Report_2020-2021_vFINAL.pdf")


# Defining Status Column for each report, beginning with "notavail"
gs["status"] = gs["report_name"].apply(lambda r: None if r in reports_downloaded else "notavail")


# Stipping unnecessary columns
toKeep = ["report_name", "year", "scope", "page", "value", "unit", "unit_normalized", "status"]
gs_slim = gs[toKeep]

# Sorting rows like VSC does it with JSON
gs_slim["_sort"] = gs_slim["report_name"].str.lower() # need to circumvent ASCII case-sensitive sorting
gs_slim = gs_slim.sort_values(["_sort", "scope", "year"], ignore_index=True).drop(columns="_sort")

# Categorizing reports by scope coverage
scope_counts = (
    gs_slim.groupby(["report_name", "scope"])["value"]
    .apply(lambda x: x.notna().sum())
    .unstack(fill_value=0)
)

def categorize(row):
    present = [s for s in ["1", "2lb", "2mb", "3"] if row.get(s, 0) > 0]
    if not present: return "useless", None
    return ("complete" if len(present) == 4 else "partial"), "+".join(present)

scope_counts[["status", "scopes_present"]] = [
    categorize(row) for row in scope_counts.to_dict("records")
]

gs_slim = gs_slim.drop(columns="status").merge(scope_counts[["status", "scopes_present"]], on="report_name")
gs_slim.to_csv("gs_slim.csv", index=False)

print("="*60)
print(gs_slim.drop_duplicates("report_name")["status"].value_counts())
print("="*60)
print(gs_slim.drop_duplicates("report_name")["scopes_present"].value_counts(dropna=False))
print("="*60)