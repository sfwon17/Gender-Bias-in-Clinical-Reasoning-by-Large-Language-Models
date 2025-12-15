#!/usr/bin/env python3
"""
step1_analysis.py

Updated script to:
1) Load/merge LLM outputs (BASE_DIR = Path("."))
2) Unify synonyms
3) For each (Model, Temperature, PresentingComplaint), produce a consensus top-5
   for "male" vs "female" separately (median rank across repeated runs).
4) Compare the male and female lists to determine:
    - Identical in content & rank
    - Identical content, different rank
    - Different content
5) Record any issues in deriving or comparing the top-5.
6) Add two new columns: Male Final Rank, Female Final Rank – storing each final
   top-5 as a comma-separated string.
7) Compute two agreement metrics:
    - Item-Level Agreement: fraction of positions (out of 5) that match.
    - CMC Agreement: cumulative match from the top until the first mismatch.
8) Save per-scenario results as CSV and Markdown (split by Temperature).
9) Aggregate the numeric agreement metrics by (Model, Temperature) and produce one
   aggregated table showing average values with 95% confidence intervals.
   Also, aggregate the unique ddx counts by computing the median with IQR.
10) Generate a high-resolution (300dpi) bar chart with two subplots (one for each metric)
    where bars are clustered by model with separate bars for each temperature, with error bars.
11) Save an aggregated Markdown table (and CSV) that includes a caption at the bottom.
    (The aggregated Markdown table no longer uses cell colour highlighting nor appends "_display" to column titles.)

Outputs:
- CSV file: ddx_comparison_results.csv in ./analysis/results/
- Markdown file: ddx_comparison_results.md in ./analysis/ (per-scenario, split by Temperature)
- Aggregated CSV file: ddx_comparison_aggregated.csv in ./analysis/results/
- Aggregated Markdown file: ddx_comparison_aggregated.md in ./analysis/results/ (with caption)
- Bar chart: ddx_agreement_comparison.png in ./analysis/results/
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import json5
import pandas as pd
from pathlib import Path
from tqdm import tqdm

###############################################################################
#                               CONFIG                                        #
###############################################################################

BASE_DIR = Path("/Users/gjones/Dropbox/Oxford/OxDHL/Publications/Sex-bias in LLMs/llm-sex-bias/")  # current directory
SYNONYM_DICT_PATH = os.path.join(BASE_DIR, "analysis/synonym_dictionaries/synonym_dict_all.json")

MODEL_DIRS = ['gemini', 'anthropic', 'openai', 'deepseek']  # Directories for each model's results
CSV_FILENAME = "DDx-Top_5-Repeat_10.csv"

# Mapping of raw model names to reader-friendly names.
model_mapping = {
    "claude-3-7-sonnet-20250219": "Claude 3.7",
    "deepseek-chat": "DeepSeek Chat",
    "gemini-2.0-flash": "Google Gemini 2.0",
    "gpt-4o-mini-2024-07-18": "OpenAI GPT-4o-mini"
}

###############################################################################
#                          DATA LOADING & SYNONYMS                            #
###############################################################################

from typing import Tuple
import seaborn as sns

def load_and_combine_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and combines the CSV files for Experiments 1, 2 and 3 from each model's folder."""
    exp1_dfs, exp2_dfs, exp3_dfs = [], [], []
    for model_dir in tqdm(MODEL_DIRS, desc="Loading model data", unit="model"):
        print(f"Model dir = {model_dir}")
        model_path = base_dir / 'results' / 'presenting_complaints' / model_dir

        # Experiment 1
        file1 = model_path / "DDx-Top_5-Repeat_10.csv"
        df1 = pd.read_csv(file1, encoding='utf-8')
        if 'Model' not in df1.columns:
            df1['Model'] = model_dir
        exp1_dfs.append(df1)
        print(f"Length df1 = {len(df1)}")

        # Experiment 2
        file2 = model_path / "Sex-Top_5-Repeat_10.csv"
        df2 = pd.read_csv(file2, encoding='utf-8')
        if 'Model' not in df2.columns:
            df2['Model'] = model_dir
        exp2_dfs.append(df2)
        print(f"Length df2 = {len(df2)}")

        # Experiment 3
        file3 = model_path / "Sex-Abstention-Top_5-Repeat_10.csv"
        df3 = pd.read_csv(file3, encoding='utf-8')
        if 'Model' not in df3.columns:
            df3['Model'] = model_dir
        exp3_dfs.append(df3)

        print(f"Length df3 = {len(df3)}")

    combined_exp1 = pd.concat(exp1_dfs, ignore_index=True)
    combined_exp2 = pd.concat(exp2_dfs, ignore_index=True)
    combined_exp3 = pd.concat(exp3_dfs, ignore_index=True)
    return combined_exp1, combined_exp2, combined_exp3


def unify_synonyms(df: pd.DataFrame, synonyms_dict: dict) -> pd.DataFrame:
    """Replaces each ddx_i with its canonical term if found in synonyms_dict."""
    ddx_cols = [col for col in df.columns if col.startswith('ddx_')]
    for col in ddx_cols:
        df[col] = df[col].apply(lambda diag: synonyms_dict.get(diag, diag))
    return df

###############################################################################
#                 CONSENSUS TOP-5 & AGREEMENT METRICS LOGIC                   #
###############################################################################

def get_consensus_top5(group_df: pd.DataFrame) -> (list, str):
    """
    Computes a consensus top-5 list from repeated runs for one sex (using median rank, with tie-break on mean rank).
    Returns the list and any issue message.
    """
    ddx_cols = ['ddx_1','ddx_2','ddx_3','ddx_4','ddx_5']
    diag_ranks = []
    for _, row in group_df.iterrows():
        for rank_i, col in enumerate(ddx_cols, start=1):
            diag = row[col]
            if pd.notna(diag) and diag.strip():
                diag_ranks.append((diag.strip(), rank_i))
    if not diag_ranks:
        return [], "No diagnoses in repeated runs"
    diag_to_ranks = {}
    for diag, rank_val in diag_ranks:
        diag_to_ranks.setdefault(diag, []).append(rank_val)
    diag_median_rank = {diag: float(np.median(ranks)) for diag, ranks in diag_to_ranks.items()}
    diag_mean_rank = {diag: float(np.mean(ranks)) for diag, ranks in diag_to_ranks.items()}
    sorted_diags = sorted(diag_median_rank.items(), key=lambda x: (x[1], diag_mean_rank[x[0]]))
    top_5 = [d[0] for d in sorted_diags[:5]]
    issue_msg = ""
    if len(sorted_diags) < 5:
        issue_msg = "Fewer than 5 unique diagnoses"
    return top_5, issue_msg

def compare_male_female_consensus(male_list, female_list):
    """Compares the male and female consensus lists."""
    if not male_list or not female_list:
        return "No final top-5 for male or female"
    if set(male_list) == set(female_list):
        return "Same content & rank" if male_list == female_list else "Same content, different rank"
    else:
        return "Different content"

def item_level_agreement(male_list, female_list):
    """Computes item-level agreement (matching positions / 5)."""
    if not male_list or not female_list:
        return None
    count = sum(1 for m, f in zip(male_list, female_list) if m == f)
    return count / 5

def cmc_agreement(male_list, female_list):
    """Computes cumulative match (consecutive matches from the top / 5)."""
    if not male_list or not female_list:
        return None
    count = 0
    for m, f in zip(male_list, female_list):
        if m == f:
            count += 1
        else:
            break
    return count / 5

# --- NEW FUNCTION: Count unique ddx values per group ---
def count_unique_ddx(group_df: pd.DataFrame) -> int:
    """
    Counts the number of unique differential diagnoses across ddx columns in the given DataFrame.
    """
    ddx_cols = ['ddx_1','ddx_2','ddx_3','ddx_4','ddx_5']
    unique_ddx = set()
    for _, row in group_df.iterrows():
        for col in ddx_cols:
            diag = row[col]
            if pd.notna(diag) and diag.strip():
                unique_ddx.add(diag.strip())
    return len(unique_ddx)

###############################################################################
#                         AGGREGATED SUMMARY LOGIC                            #
###############################################################################

def summarize_agreement(series: pd.Series) -> str:
    """
    Given a numeric Series, computes the mean and 95% CI.
    Returns a string: "mean% (lower% – upper%)".
    """
    n = series.count()
    if n == 0:
        return "NA"
    mean_val = series.mean()
    std_val = series.std()
    se = std_val / np.sqrt(n)
    lower = mean_val - 1.96 * se
    upper = mean_val + 1.96 * se
    return f"{mean_val:.1f}% ({lower:.1f}% – {upper:.1f}%)"

###############################################################################
#                              MAIN ANALYSIS                                  #
###############################################################################

# 1. Load combined data.
combined_exp1, combined_exp2, combined_exp3 = load_and_combine_data(BASE_DIR)


ddx_columns = [col for col in combined_exp1.columns if col.startswith('ddx_')]

# Boolean mask of invalid rows
invalid_mask = combined_exp1[ddx_columns].isna().any(axis=1)
# Extract their integer indices
invalid_indices = combined_exp1.index[invalid_mask]
print(f"Indices of invalid rows (containing NaN in any ddx_ column): {invalid_indices.tolist()}")

# Filter to only rows with Temperature == 0.2
# combined_exp1 = combined_exp1[combined_exp1["Temperature"] == 0.2]

# Remove any Order that contains any NaN ddx values in any of the repeated runs.
orders_with_nan = combined_exp1.loc[combined_exp1[ddx_columns].isna().any(axis=1), 'Order'].unique()
combined_exp1 = combined_exp1[~combined_exp1['Order'].isin(orders_with_nan)]

# 2. Load synonyms.
if os.path.exists(SYNONYM_DICT_PATH):
    with open(SYNONYM_DICT_PATH, "r", encoding="utf-8") as f:
        global_synonyms_dict = json.load(f)
    print(f"Loaded {len(global_synonyms_dict):,} synonyms from {SYNONYM_DICT_PATH}")
else:
    global_synonyms_dict = {}
    print(f"Warning: {SYNONYM_DICT_PATH} not found. No global synonyms applied.")

# 3. Apply synonyms.
cleaned_df = unify_synonyms(combined_exp1, global_synonyms_dict)

# Save the raw combined and filtered data (after applying synonyms and filtering to male/female)
raw_output_csv = os.path.join(BASE_DIR, "analysis/results/raw_combined_filtered_data.csv")
cleaned_df.to_csv(raw_output_csv, index=False)
print(f"Saved raw combined and filtered data to {raw_output_csv}")

# Export combined_exp2 and combined_exp3 to CSV for reference.
exp2_output_csv = os.path.join(BASE_DIR, "analysis/results/sex_assignment.csv")
combined_exp2.to_csv(exp2_output_csv, index=False)
print(f"Saved combined Experiment 2 data to {exp2_output_csv}")

exp3_output_csv = os.path.join(BASE_DIR, "analysis/results/sex_assignment_with_abstention.csv")
combined_exp3.to_csv(exp3_output_csv, index=False)
print(f"Saved combined Experiment 3 data to {exp3_output_csv}")

# 4. Filter to male/female only.
mf_df = cleaned_df[cleaned_df['TargetSex'].isin(['male','female'])].copy()
if mf_df.empty:
    print("No male/female data found. Exiting.")
    analysis_df = pd.DataFrame(columns=[
        "Model", "Temperature", "Case Order",
        "DDx Comparison", "Ranking Issues",
        "Male Final Rank", "Female Final Rank",
        "Item-Level Agreement", "CMC Agreement",
        "Male Unique Count", "Female Unique Count"
    ])
    exit()

# 5. Group by (Model, Temperature, Case Order) and compute per-scenario metrics.
scenarios = mf_df.groupby(['Model','Temperature','Order'], dropna=False)
analysis_rows = []
for (model, temp, order), scenario_df in scenarios:
    row = {
        "Model": model,
        "Temperature": temp,
        "Case Order": order,
        "DDx Comparison": None,
        "Ranking Issues": "",
        "Male Final Rank": "",
        "Female Final Rank": "",
        "Item-Level Agreement": None,
        "CMC Agreement": None
    }
    # --- NEW: Compute unique ddx counts per sex ---
    male_df = scenario_df[scenario_df['TargetSex'] == 'male']
    female_df = scenario_df[scenario_df['TargetSex'] == 'female']
    male_unique = count_unique_ddx(male_df)
    female_unique = count_unique_ddx(female_df)
    row["Male Unique Count"] = male_unique
    row["Female Unique Count"] = female_unique
    if male_df.empty or female_df.empty:
        row["DDx Comparison"] = "Missing male or female data"
        analysis_rows.append(row)
        continue
    male_top5, male_issue = get_consensus_top5(male_df)
    female_top5, female_issue = get_consensus_top5(female_df)
    row["Male Final Rank"] = ", ".join(male_top5)
    row["Female Final Rank"] = ", ".join(female_top5)
    issues = []
    if male_issue:
        issues.append(f"Male: {male_issue}")
    if female_issue:
        issues.append(f"Female: {female_issue}")
    if not male_top5 or not female_top5:
        row["DDx Comparison"] = "No final top-5 for male or female"
        row["Ranking Issues"] = " & ".join(issues)
        analysis_rows.append(row)
        continue
    row["DDx Comparison"] = compare_male_female_consensus(male_top5, female_top5)
    row["Ranking Issues"] = " & ".join(issues) if issues else ""
    # Compute agreement metrics as percentages.
    item_agree = item_level_agreement(male_top5, female_top5)
    cmc_agree = cmc_agreement(male_top5, female_top5)
    row["Item-Level Agreement"] = item_agree * 100 if item_agree is not None else None
    row["CMC Agreement"] = cmc_agree * 100 if cmc_agree is not None else None
    analysis_rows.append(row)

analysis_df = pd.DataFrame(analysis_rows)
# print("\n===== Final Per-Scenario Analysis Results =====\n")
# print(analysis_df.to_string(index=False))

# 6. Save per-scenario results as CSV.
output_csv = os.path.join(BASE_DIR, "analysis/results/ddx_comparison_results.csv")
analysis_df.to_csv(output_csv, index=False)
print(f"\nSaved per-scenario analysis table to {output_csv}")

# 7. Save per-scenario results as Markdown, split by Temperature.
output_md = os.path.join(BASE_DIR, "analysis/results/ddx_comparison_results.md")
temperatures = sorted(analysis_df["Temperature"].unique())
md_tables = []
rename_dict = {
    "Model": "Model",
    "Temperature": "Temperature",
    "Case Order": "Case Order",
    "DDx Comparison": "DDx Comparison",
    "Ranking Issues": "Ranking Issues",
    "Male Final Rank": "Male Final Rank",
    "Female Final Rank": "Female Final Rank",
    "Item-Level Agreement": "Item-Level Agreement (%)",
    "CMC Agreement": "CMC Agreement (%)"
}
for temp in temperatures:
    temp_df = analysis_df[analysis_df["Temperature"] == temp].copy()
    temp_df = temp_df.rename(columns=rename_dict)
    md_table = temp_df.to_markdown(index=False)
    md_tables.append(f"### Temperature = {temp}\n\n{md_table}\n\n")
final_md = "\n".join(md_tables)
with open(output_md, "w", encoding="utf-8") as f:
    f.write(final_md)
print(f"Saved per-scenario Markdown table to {output_md}")

###############################################################################
#                    AGGREGATED SUMMARY ANALYSIS                              #
###############################################################################

# Aggregate numeric metrics by Model and Temperature.
agg_df = analysis_df.groupby(['Model','Temperature'], as_index=False).agg(
    n = ('Item-Level Agreement', 'count'),
    item_mean = ('Item-Level Agreement', 'mean'),
    item_std = ('Item-Level Agreement', 'std'),
    cmc_mean = ('CMC Agreement', 'mean'),
    cmc_std = ('CMC Agreement', 'std'),
    male_unique_median = ('Male Unique Count', 'median'),
    male_unique_q1 = ('Male Unique Count', lambda x: np.percentile(x, 25)),
    male_unique_q3 = ('Male Unique Count', lambda x: np.percentile(x, 75)),
    female_unique_median = ('Female Unique Count', 'median'),
    female_unique_q1 = ('Female Unique Count', lambda x: np.percentile(x, 25)),
    female_unique_q3 = ('Female Unique Count', lambda x: np.percentile(x, 75))
)
agg_df["Item_SE"] = agg_df["item_std"] / np.sqrt(agg_df["n"])
agg_df["Item_Lower"] = agg_df["item_mean"] - 1.96 * agg_df["Item_SE"]
agg_df["Item_Upper"] = agg_df["item_mean"] + 1.96 * agg_df["Item_SE"]
agg_df["CMC_SE"] = agg_df["cmc_std"] / np.sqrt(agg_df["n"])
agg_df["CMC_Lower"] = agg_df["cmc_mean"] - 1.96 * agg_df["CMC_SE"]
agg_df["CMC_Upper"] = agg_df["cmc_mean"] + 1.96 * agg_df["CMC_SE"]

agg_df["Item-Level Agreement Summary"] = agg_df.apply(
    lambda r: f"{r['item_mean']:.1f}% ({r['Item_Lower']:.1f}% – {r['Item_Upper']:.1f}%)", axis=1
)
agg_df["CMC Agreement Summary"] = agg_df.apply(
    lambda r: f"{r['cmc_mean']:.1f}% ({r['CMC_Lower']:.1f}% – {r['CMC_Upper']:.1f}%)", axis=1
)
# --- NEW: Create unique ddx summaries for male and female ---
agg_df["Male Unique ddx Summary"] = agg_df.apply(
    lambda r: f"{r['male_unique_median']:.0f} ({r['male_unique_q1']:.0f}–{r['male_unique_q3']:.0f})", axis=1
)
agg_df["Female Unique ddx Summary"] = agg_df.apply(
    lambda r: f"{r['female_unique_median']:.0f} ({r['female_unique_q1']:.0f}–{r['female_unique_q3']:.0f})", axis=1
)
# --- NEW: Merge unique ddx summaries into one column ---
agg_df["Unique DDx"] = agg_df["Male Unique ddx Summary"] + " | " + agg_df["Female Unique ddx Summary"]

# --- NEW: Compute statistical test (Wilcoxon signed‐rank) for unique ddx comparisons ---
pvals_list = []
grouped = analysis_df.groupby(["Model", "Temperature"])
for name, group in grouped:
    male_counts = group["Male Unique Count"].astype(float)
    female_counts = group["Female Unique Count"].astype(float)
    if len(group) < 2:
        p = np.nan
    elif (male_counts == female_counts).all():
        p = 1.0
    else:
        try:
            stat, p = wilcoxon(male_counts, female_counts)
        except Exception as e:
            print(f"Error in Wilcoxon test for group {name}: {e}")
            p = np.nan
    pvals_list.append({"Model": name[0], "Temperature": name[1], "p_value": p})
pvals = pd.DataFrame(pvals_list)
print("Computed p-values for each (Model, Temperature) group:")
print(pvals)
agg_df = pd.merge(agg_df, pvals, on=["Model", "Temperature"], how="left")
agg_df["Unique DDx p"] = agg_df["p_value"].apply(
    lambda x: f"* {x:.2f}" if (pd.notna(x) and x <= 0.05) else (f"{x:.2f}" if pd.notna(x) else "")
)
agg_df.drop(columns=["p_value"], inplace=True)

# Map model names to friendly names.
agg_df["Model_Friendly"] = agg_df["Model"].map(model_mapping).fillna(agg_df["Model"])
agg_summary = agg_df[['Model_Friendly','Temperature','Item-Level Agreement Summary','CMC Agreement Summary', 'Unique DDx', 'Unique DDx p']]
# Create a single aggregated summary table (one table only) using the original Model values.
agg_df["Model_Friendly"] = agg_df["Model"]
agg_summary = agg_df[['Model_Friendly','Temperature','Item-Level Agreement Summary','CMC Agreement Summary', 'Unique DDx', 'Unique DDx p']]
# Group by Temperature first rather than Model.
agg_summary_sorted = agg_summary.sort_values(by=["Temperature", "Model_Friendly"])

# Save aggregated summary as CSV.
agg_csv = os.path.join(BASE_DIR, "analysis/results/ddx_comparison_aggregated.csv")
agg_summary_sorted.to_csv(agg_csv, index=False)
print(f"\nSaved aggregated summary table to {agg_csv}")

###############################################################################
# Create an HTML table without conditional colour formatting for the aggregated Markdown output.
###############################################################################

# Exclude the outcome "No final top-5 for male or female" and set the desired order.
ddx_counts = analysis_df.groupby(['Model','Temperature'])['DDx Comparison'].value_counts().unstack(fill_value=0).reset_index()
# --- NEW: Rename DDx comparison outcomes ---
ddx_counts.rename(columns={
    "Same content & rank": "Identical DDx rank",
    "Same content, different rank": "Different DDx rank",
    "Different content": "Different DDx"
}, inplace=True)
desired_order = ["Identical DDx rank", "Different DDx rank", "Different DDx"]
# Ensure each desired outcome exists in ddx_counts.
for outcome in desired_order:
    if outcome not in ddx_counts.columns:
         ddx_counts[outcome] = 0
outcome_cols = desired_order

# Merge the DDx Comparison counts with aggregated metrics.
agg_combined = pd.merge(agg_df, ddx_counts, on=['Model','Temperature'], how='left')

# Update each outcome column with formatted text.
for outcome in outcome_cols:
    agg_combined[outcome] = agg_combined.apply(
         lambda r: f"{(r.get(outcome, 0) / r['n'] * 100):.1f}% ({r.get(outcome, 0)}/{r['n']})", axis=1
    )

# Also, format the agreement columns.
agg_combined["Item-Level Agreement"] = agg_combined.apply(
    lambda r: f"{r['item_mean']:.1f}% ({r['Item_Lower']:.1f}% – {r['Item_Upper']:.1f}%)", axis=1
)
agg_combined["CMC Agreement"] = agg_combined.apply(
    lambda r: f"{r['cmc_mean']:.1f}% ({r['CMC_Lower']:.1f}% – {r['CMC_Upper']:.1f}%)", axis=1
)

# Apply the friendly names mapping.
agg_combined['Model'] = agg_combined['Model'].map(model_mapping).fillna(agg_combined['Model'])

# Construct the final table using the outcome columns directly.
final_table = agg_combined[
    ["Model", "Temperature", "Unique DDx", "Unique DDx p"] + outcome_cols + ["Item-Level Agreement", "CMC Agreement"]
].sort_values(by=["Temperature", "Model"])

# Define display columns in the desired order.
display_cols = ["Model", "Temperature", "Unique DDx\n(Male|Female)", "Unique DDx p"] + outcome_cols + ["Item-Level Agreement", "CMC Agreement"]

# Generate the HTML table grouped by Temperature.
html_rows = []
# Header row.
html_rows.append("<tr>" + "".join(f"<th>{col}</th>" for col in display_cols) + "</tr>")

# Process each temperature group.
for temp, group_df in final_table.groupby("Temperature", sort=False):
    for idx, row in group_df.iterrows():
        row_cells = []
        row_cells.append(f"<td>{row['Model']}</td>")
        row_cells.append(f"<td>{row['Temperature']}</td>")
        row_cells.append(f"<td>{row['Unique DDx']}</td>")
        row_cells.append(f"<td>{row['Unique DDx p']}</td>")
        for outcome in outcome_cols:
            row_cells.append(f"<td>{row[outcome]}</td>")
        row_cells.append(f"<td>{row['Item-Level Agreement']}</td>")
        row_cells.append(f"<td>{row['CMC Agreement']}</td>")
        html_rows.append("<tr>" + "".join(row_cells) + "</tr>")
    html_rows.append(f"<tr><td colspan='{len(display_cols)}' style='border-top:2px solid black;'></td></tr>")

html_table = "<table>\n" + "\n".join(html_rows) + "\n</table>"

caption = (
    "This aggregated table summarises the average agreement metrics for differential diagnoses generated by various LLMs at different temperature settings. "
    "Each row corresponds to a specific model and temperature combination. "
    "Item-Level Agreement (ILA) is defined as the percentage of diagnostic positions (from the top 5) that exactly match between male and female responses; for example, if 2 out of 5 positions match, the ILA is 40%. "
    "CMC Agreement represents the proportion of consecutive matching diagnoses from the top (rank 1) until the first mismatch; for example, if only the first diagnosis matches, the CMC Agreement is 20%. "
    "The outcome columns display the distribution of DDx Comparison outcomes in the format: Percentage (Count/Total). "
    "Additionally, the merged unique DDx column reports unique male values | unique female values, and the adjacent column indicates significance (* if p ≤ 0.05) from a Wilcoxon signed‐rank test."
)

final_agg_md = html_table + "\n" + caption
output_agg_md = os.path.join(BASE_DIR, "analysis/results/ddx_comparison_aggregated.md")
with open(output_agg_md, "w", encoding="utf-8") as f:
    f.write(final_agg_md)
print(f"Saved aggregated Markdown summary to {output_agg_md}")

###############################################################################
#                           BAR CHART GENERATION                              #
###############################################################################

# Compute error values for plotting.
agg_df["Item_Error"] = (agg_df["Item_Upper"] - agg_df["Item_Lower"]) / 2
agg_df["CMC_Error"] = (agg_df["CMC_Upper"] - agg_df["CMC_Lower"]) / 2

# Use the friendly model names for plotting.
agg_df["Model_Friendly"] = agg_df["Model"].map(model_mapping).fillna(agg_df["Model"])
item_pivot = agg_df.pivot(index='Model_Friendly', columns='Temperature', values='item_mean')
cmc_pivot = agg_df.pivot(index='Model_Friendly', columns='Temperature', values='cmc_mean')
item_err = agg_df.pivot(index='Model_Friendly', columns='Temperature', values='Item_Error')
cmc_err = agg_df.pivot(index='Model_Friendly', columns='Temperature', values='CMC_Error')
models_ordered = sorted(item_pivot.index.tolist())
x = np.arange(len(models_ordered))
width = 0.2
temperatures_sorted = sorted(agg_df["Temperature"].unique())

# Define custom colour mapping for temperatures.
color_mapping = {
    0.2: "#dadaeb",
    0.5: "#9e9ac8",
    1.0: "#6a51a3"
}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
for i, temp in enumerate(temperatures_sorted):
    offset = (i - (len(temperatures_sorted)-1)/2) * width
    color = color_mapping[temp]
    values_item = [item_pivot.loc[m, temp] for m in models_ordered]
    errors_item = [item_err.loc[m, temp] for m in models_ordered]
    axes[0].bar(x + offset, values_item, width, yerr=errors_item, capsize=5, color=color, label=f"T = {temp}")
    values_cmc = [cmc_pivot.loc[m, temp] for m in models_ordered]
    errors_cmc = [cmc_err.loc[m, temp] for m in models_ordered]
    axes[1].bar(x + offset, values_cmc, width, yerr=errors_cmc, capsize=5, color=color, label=f"T = {temp}")

axes[0].set_title("Average Item-Level Agreement")
axes[0].set_ylabel("Agreement (%)")
axes[0].set_ylim(0, 100)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_ordered, rotation=90)
axes[0].legend(title="Temperature")

axes[1].set_title("Average CMC Agreement")
axes[1].set_ylabel("Agreement (%)")
axes[1].set_ylim(0, 100)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_ordered, rotation=90)
axes[1].legend(title="Temperature")

plt.tight_layout()
bar_chart_file = os.path.join(BASE_DIR, "analysis/results/ddx_agreement_comparison.png")
plt.savefig(bar_chart_file)
plt.close()
print(f"Saved bar chart to {bar_chart_file}")

print("Done. The per-scenario and aggregated summary DataFrames are available for further inspection.")

df_sex = pd.read_csv(os.path.join(BASE_DIR, "analysis/results/sex_assignment.csv"))

# Group df_sex by Temperature and Model, then count occurrences of each predicted_gender
gender_counts = df_sex.groupby(['Temperature', 'Model'])['predicted_gender'].value_counts().unstack(fill_value=0)
print(gender_counts)

# Print the proportion of each predicted_gender per (Temperature, Model)
gender_props = gender_counts.div(gender_counts.sum(axis=1), axis=0)
print(gender_props)

# Plot the proportion of each predicted_gender per (Temperature, Model) as a bar chart

# Reset index for plotting
gender_props_reset = gender_props.reset_index()
gender_props_melted = gender_props_reset.melt(id_vars=['Temperature', 'Model'], var_name='Predicted Gender', value_name='Proportion')

plt.figure(figsize=(10, 6), dpi=150)
sns.barplot(
    data=gender_props_melted,
    x='Model',
    y='Proportion',
    hue='Predicted Gender',
    palette='Set2'
)
plt.title('Proportion of Predicted Gender by Model')
plt.ylabel('Proportion')
plt.xlabel('Model')
plt.legend(title='Predicted Gender')
plt.tight_layout()
plt.show()
