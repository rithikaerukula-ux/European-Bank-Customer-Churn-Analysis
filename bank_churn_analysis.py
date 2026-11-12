"""
European Banking Customer Churn Analysis
Run:
    pip install pandas numpy matplotlib openpyxl
    python bank_churn_analysis.py

Input:
    European_Bank (1)(1).csv
Outputs:
    Bank_Churn_Analysis_Results.xlsx
    Chapter_4_Copy_Paste_Report.txt
    charts/ (PNG charts)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

INPUT_FILE = "European_Bank (1)(1).csv"
OUTPUT_EXCEL = "Bank_Churn_Analysis_Results.xlsx"
OUTPUT_REPORT = "Chapter_4_Copy_Paste_Report.txt"
CHART_DIR = Path("charts")

def safe_pct(num, den):
    return (num / den * 100) if den else 0.0

def clean_col_name(name):
    return str(name).strip()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        # Also allow a user to pass the CSV path as a command-line argument.
        import sys
        if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
            input_path = Path(sys.argv[1])
        else:
            raise FileNotFoundError(
                f"Cannot find {INPUT_FILE}. Keep this script in the same folder as the CSV "
                "or run: python bank_churn_analysis.py 'full/path/to/your.csv'"
            )

    df = pd.read_csv(input_path)
    df.columns = [clean_col_name(c) for c in df.columns]
    original_rows = len(df)
    original_cols = len(df.columns)

    required = ["Exited", "Geography", "Age", "Gender", "CreditScore", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}")

    # Convert expected numerical columns where present.
    numeric_cols = [c for c in [
        "Year", "CustomerId", "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quality checks before cleaning.
    missing_by_col = df.isna().sum().rename("Missing values").to_frame()
    duplicate_rows = int(df.duplicated().sum())
    invalid_exited = int((~df["Exited"].isin([0, 1]) & df["Exited"].notna()).sum())
    binary_checks = {}
    for c in ["Exited", "HasCrCard", "IsActiveMember"]:
        if c in df.columns:
            binary_checks[c] = sorted(df[c].dropna().unique().tolist(), key=lambda x: str(x))

    # Clean only exact duplicate rows; drop rows with missing target or invalid target.
    df = df.drop_duplicates().copy()
    df = df[df["Exited"].isin([0, 1])].copy()
    # For core analysis, keep rows with non-missing values only where a specific analysis needs that variable.
    cleaned_rows = len(df)

    # Labels for segmentation. These are analysis choices, not official bank categories.
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[-np.inf, 29, 39, 49, np.inf],
        labels=["Below 30", "30–39", "40–49", "50 and above"], right=True
    )
    df["TenureGroup"] = pd.cut(
        df["Tenure"], bins=[-np.inf, 2, 5, np.inf],
        labels=["Short tenure (0–2)", "Medium tenure (3–5)", "Long tenure (6+)"], right=True
    )
    # Data-driven credit score and balance/salary groups using quartiles where possible.
    def qcut_safe(series, labels):
        valid = series.dropna()
        if valid.nunique() < 2:
            return pd.Series(["Single group" if pd.notna(x) else np.nan for x in series], index=series.index)
        try:
            out = pd.qcut(series, q=4, labels=labels, duplicates="drop")
            return out.astype("object")
        except Exception:
            return pd.Series(["Unclassified" if pd.notna(x) else np.nan for x in series], index=series.index)
    df["CreditScoreGroup"] = qcut_safe(df["CreditScore"], ["Lowest quartile", "Lower-middle", "Upper-middle", "Highest quartile"])
    df["BalanceGroup"] = np.where(df["Balance"].isna(), np.nan,
                            np.where(df["Balance"] == 0, "Zero balance",
                            qcut_safe(df.loc[df["Balance"] > 0, "Balance"],
                                      ["Low positive", "Medium-low positive", "Medium-high positive", "High positive"]).reindex(df.index)))
    # Fill missing labels only for the balance group where the original balance was zero.
    df.loc[df["Balance"].eq(0), "BalanceGroup"] = "Zero balance"
    df["SalaryGroup"] = qcut_safe(df["EstimatedSalary"], ["Lowest quartile", "Lower-middle", "Upper-middle", "Highest quartile"])

    # Common segment table with counts, churn rates and segment contribution.
    total_n = len(df)
    total_exited = int((df["Exited"] == 1).sum())
    total_retained = int((df["Exited"] == 0).sum())
    overall_churn = safe_pct(total_exited, total_n)
    overall_retention = safe_pct(total_retained, total_n)

    def segment_table(column, label=None):
        d = df[df[column].notna()].copy()
        if d.empty:
            return pd.DataFrame(columns=["Segment", "Total customers", "Exited customers", "Retained customers", "Churn rate (%)", "Customer share (%)", "Share of all exits (%)"])
        out = d.groupby(column, observed=False)["Exited"].agg(
            **{"Total customers": "count", "Exited customers": "sum"}
        ).reset_index().rename(columns={column: "Segment"})
        out["Retained customers"] = out["Total customers"] - out["Exited customers"]
        out["Churn rate (%)"] = (out["Exited customers"] / out["Total customers"] * 100).round(2)
        out["Customer share (%)"] = (out["Total customers"] / len(df) * 100).round(2)
        out["Share of all exits (%)"] = np.where(total_exited > 0, out["Exited customers"] / total_exited * 100, 0).round(2)
        return out

    # Overall and data quality tables
    overall = pd.DataFrame([
        ["Total records in original file", original_rows],
        ["Total columns in original file", original_cols],
        ["Exact duplicate rows identified", duplicate_rows],
        ["Invalid non-missing Exited values identified", invalid_exited],
        ["Records after removing exact duplicates and invalid/missing target", cleaned_rows],
        ["Exited customers", total_exited],
        ["Retained customers", total_retained],
        ["Overall churn rate (%)", round(overall_churn, 2)],
        ["Overall retention rate (%)", round(overall_retention, 2)],
    ], columns=["Measure", "Result"])

    quality = missing_by_col.reset_index().rename(columns={"index": "Variable"})
    quality["Missing percentage (%)"] = (quality["Missing values"] / original_rows * 100).round(2)
    quality = quality[["Variable", "Missing values", "Missing percentage (%)"]]

    # Numeric profile comparison
    profile_vars = [c for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"] if c in df.columns]
    profile_rows = []
    for var in profile_vars:
        for status, code in [("Exited customers", 1), ("Retained customers", 0)]:
            s = df.loc[df["Exited"] == code, var].dropna()
            profile_rows.append([var, status, len(s), s.mean() if len(s) else np.nan, s.median() if len(s) else np.nan,
                                 s.min() if len(s) else np.nan, s.max() if len(s) else np.nan])
    profile = pd.DataFrame(profile_rows, columns=["Variable", "Customer status", "Count", "Mean", "Median", "Minimum", "Maximum"])
    for c in ["Mean", "Median", "Minimum", "Maximum"]:
        profile[c] = profile[c].round(2)

    # Binary/category comparisons
    categorical_tables = {}
    segment_cols = {
        "Geography": "Geography",
        "Gender": "Gender",
        "Age group": "AgeGroup",
        "Credit score group": "CreditScoreGroup",
        "Tenure group": "TenureGroup",
        "Balance group": "BalanceGroup",
        "Number of products": "NumOfProducts",
        "Credit card ownership": "HasCrCard",
        "Activity status": "IsActiveMember",
        "Estimated salary group": "SalaryGroup",
    }
    for sheet, col in segment_cols.items():
        if col in df.columns:
            t = segment_table(col)
            categorical_tables[sheet] = t

    # Friendly labels for binary tables.
    for name, col in [("Credit card ownership", "HasCrCard"), ("Activity status", "IsActiveMember")]:
        if name in categorical_tables:
            t = categorical_tables[name].copy()
            mapping = {"HasCrCard": {0: "No credit card", 1: "Has credit card"},
                       "IsActiveMember": {0: "Inactive", 1: "Active"}}
            t["Segment"] = t["Segment"].map(mapping[col]).fillna(t["Segment"].astype(str))
            categorical_tables[name] = t

    # High-balance group: use positive balance 75th percentile as a transparent proxy.
    positive_balances = df.loc[df["Balance"] > 0, "Balance"].dropna()
    high_balance_threshold = float(positive_balances.quantile(0.75)) if len(positive_balances) else np.nan
    highbal = df[df["Balance"].notna() & (df["Balance"] >= high_balance_threshold)].copy() if pd.notna(high_balance_threshold) else df.iloc[0:0].copy()
    highbal_exited = int((highbal["Exited"] == 1).sum())
    highbal_total = len(highbal)
    high_balance_summary = pd.DataFrame([
        ["Positive-balance 75th percentile threshold", round(high_balance_threshold, 2) if pd.notna(high_balance_threshold) else np.nan],
        ["High-balance customers", highbal_total],
        ["High-balance customers who exited", highbal_exited],
        ["High-balance customer churn rate (%)", round(safe_pct(highbal_exited, highbal_total), 2)],
        ["Overall churn rate (%)", round(overall_churn, 2)],
        ["High-balance churn rate minus overall rate (percentage points)", round(safe_pct(highbal_exited, highbal_total)-overall_churn, 2)],
        ["Total recorded balance among high-balance customers", round(highbal["Balance"].sum(), 2)],
    ], columns=["Measure", "Result"])

    # Cross-segmentation tables
    cross_tables = {}
    for name, cols in {
        "Geography x Age": ["Geography", "AgeGroup"],
        "Geography x Balance": ["Geography", "BalanceGroup"],
        "Age x Activity": ["AgeGroup", "IsActiveMember"],
        "Tenure x Products": ["TenureGroup", "NumOfProducts"],
    }.items():
        d = df.dropna(subset=cols + ["Exited"]).copy()
        if len(d):
            t = d.groupby(cols, observed=False)["Exited"].agg(
                **{"Total customers": "count", "Exited customers": "sum"}
            ).reset_index()
            t["Retained customers"] = t["Total customers"] - t["Exited customers"]
            t["Churn rate (%)"] = (t["Exited customers"] / t["Total customers"] * 100).round(2)
            cross_tables[name] = t

    # Charts (matplotlib default palette; no forced colors)
    CHART_DIR.mkdir(exist_ok=True)
    charts_created = []
    def save_bar(x, y, title, xlabel, ylabel, filename, rotate=0):
        plt.figure(figsize=(8, 5))
        plt.bar([str(v) for v in x], y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if rotate:
            plt.xticks(rotation=rotate, ha="right")
        plt.tight_layout()
        out = CHART_DIR / filename
        plt.savefig(out, dpi=160)
        plt.close()
        charts_created.append(str(out))

    status_counts = df["Exited"].value_counts().reindex([1, 0], fill_value=0)
    save_bar(["Exited", "Retained"], [int(status_counts.loc[1]), int(status_counts.loc[0])],
             "Exited vs Retained Customers", "Customer status", "Number of customers", "overall_churn.png")
    for chart_name, table_name in [
        ("Churn Rate by Geography", "Geography"),
        ("Churn Rate by Age Group", "Age group"),
        ("Churn Rate by Gender", "Gender"),
        ("Churn Rate by Credit Score Group", "Credit score group"),
        ("Churn Rate by Tenure Group", "Tenure group"),
        ("Churn Rate by Balance Group", "Balance group"),
        ("Churn Rate by Number of Products", "Number of products"),
        ("Churn Rate by Activity Status", "Activity status"),
        ("Churn Rate by Salary Group", "Estimated salary group"),
    ]:
        if table_name in categorical_tables and not categorical_tables[table_name].empty:
            t = categorical_tables[table_name]
            save_bar(t["Segment"], t["Churn rate (%)"], chart_name, "Segment", "Churn rate (%)",
                     re.sub(r"[^a-z0-9]+", "_", chart_name.lower()).strip("_")+".png", rotate=25)

    # Text report with actual values and interpretations.
    lines = []
    lines += [
        "CHAPTER 4: DATA ANALYSIS, RESULTS AND INTERPRETATION",
        "=" * 60, "",
        "4.1 Introduction",
        "This chapter analyzes customer churn using the supplied bank customer dataset. The analysis compares exited and retained customers and examines geographic, demographic, financial, product-use, tenure, and activity segments. The results describe associations in the dataset and do not by themselves prove causation.", "",
        "4.2 Dataset Overview and Descriptive Statistics",
        f"The original dataset contains {original_rows:,} records and {original_cols} columns. After removing {duplicate_rows:,} exact duplicate rows and excluding records with missing or invalid Exited values, {cleaned_rows:,} records remained for churn analysis. The dataset includes the following variables: {', '.join(df.columns)}.", "",
        f"4.4 Overall Customer Churn Analysis",
        f"The analysis included {total_n:,} customers. Of these, {total_exited:,} customers were recorded as exited and {total_retained:,} were retained. The overall churn rate was {overall_churn:.2f}%, while the retention rate was {overall_retention:.2f}%. These values provide the baseline for comparing individual customer segments.", "",
    ]
    # Helpers for writing table-backed interpretations
    def add_segment_interpretation(section, label, key):
        t = categorical_tables.get(key)
        lines.append(section)
        if t is None or t.empty:
            lines.append(f"No usable data were available for {label}.")
        else:
            valid = t.dropna(subset=["Churn rate (%)"])
            if len(valid):
                hi = valid.loc[valid["Churn rate (%)"].idxmax()]
                lo = valid.loc[valid["Churn rate (%)"].idxmin()]
                lines.append(f"Churn varied across {label}. The highest observed rate was {hi['Churn rate (%)']:.2f}% for {hi['Segment']} (n={int(hi['Total customers'])}), while the lowest observed rate was {lo['Churn rate (%)']:.2f}% for {lo['Segment']} (n={int(lo['Total customers'])}). Compare these rates with the overall churn rate of {overall_churn:.2f}% and consider segment sizes before interpreting the differences.")
            else:
                lines.append(f"No valid churn rates were available for {label}.")
        lines.append("")

    add_segment_interpretation("4.5 Geographic Customer Segmentation", "geographic groups", "Geography")
    add_segment_interpretation("4.6 Age-Based Customer Segmentation", "age groups", "Age group")
    add_segment_interpretation("4.7 Gender-Based Customer Segmentation", "gender groups", "Gender")
    add_segment_interpretation("4.8 Credit Score Analysis", "credit score groups", "Credit score group")
    add_segment_interpretation("4.9 Tenure-Based Customer Segmentation", "tenure groups", "Tenure group")
    add_segment_interpretation("4.10 Account Balance Analysis", "balance groups", "Balance group")
    add_segment_interpretation("4.11 Product Usage and Credit Card Ownership", "number-of-product groups", "Number of products")
    add_segment_interpretation("4.12 Customer Engagement Analysis", "activity groups", "Activity status")
    add_segment_interpretation("4.13 Estimated Salary and Financial Profile", "estimated salary groups", "Estimated salary group")

    lines += [
        "4.14 High-Value and High-Balance Customer Analysis",
        f"High-balance customers were defined as customers with a positive recorded balance at or above the 75th percentile of positive balances (threshold = {high_balance_threshold:,.2f}). This group contained {highbal_total:,} customers, of whom {highbal_exited:,} exited. The churn rate for this group was {safe_pct(highbal_exited, highbal_total):.2f}%, compared with the overall rate of {overall_churn:.2f}%. The sum of balances in this group was {highbal['Balance'].sum():,.2f}. This balance total is not the same as revenue or profit lost by the bank.", "",
        "4.15 Comparison of Exited and Retained Customers",
    ]
    for var in profile_vars:
        e = df.loc[df["Exited"] == 1, var].dropna()
        r = df.loc[df["Exited"] == 0, var].dropna()
        if len(e) and len(r):
            lines.append(f"{var}: exited customers had a mean of {e.mean():.2f} and median of {e.median():.2f}; retained customers had a mean of {r.mean():.2f} and median of {r.median():.2f}.")
    for col, label in [("IsActiveMember", "active membership"), ("HasCrCard", "credit card ownership")]:
        if col in df:
            e = df.loc[df["Exited"] == 1, col].mean() * 100
            r = df.loc[df["Exited"] == 0, col].mean() * 100
            lines.append(f"The share of exited customers with {label} coded as 1 was {e:.2f}%, compared with {r:.2f}% among retained customers.")
    lines += ["", "4.16 Cross-Segmentation Analysis"]
    for name, t in cross_tables.items():
        if not t.empty:
            ix = t["Churn rate (%)"].idxmax()
            row = t.loc[ix]
            seg_cols = [c for c in t.columns if c not in ["Total customers", "Exited customers", "Retained customers", "Churn rate (%)"]]
            combo = ", ".join(f"{c}={row[c]}" for c in seg_cols)
            lines.append(f"For {name}, the highest observed churn rate was {row['Churn rate (%)']:.2f}% in the combination {combo} (n={int(row['Total customers'])}). Check the group size before drawing conclusions.")
    lines += [
        "", "4.17 Summary of Data Analysis",
        f"The dataset's overall churn rate was {overall_churn:.2f}%. Segment tables and charts in the accompanying Excel workbook show how churn varied by geography, age, gender, credit score, tenure, balance, product count, activity, and estimated salary. These are descriptive findings. They should be interpreted alongside segment sizes and the limitations of the available data; they do not establish causal relationships.", "",
        "DATA QUALITY NOTE",
        f"Missing values by variable are listed in the Excel workbook. Original dataset missing values: {int(pd.read_csv(input_path).isna().sum().sum())}. Exact duplicates identified: {duplicate_rows}. Invalid non-missing Exited values identified: {invalid_exited}.",
        "",
        "COPYING RESULTS INTO WORD",
        "1. Open Bank_Churn_Analysis_Results.xlsx.",
        "2. Copy each relevant sheet/table into Chapter 4 in Word.",
        "3. Copy the corresponding interpretation paragraphs from this text file.",
        "4. Insert charts from the charts folder where indicated.",
        "5. Confirm that the segment definitions in Chapter 3 match the groups shown in the tables.",
        "6. Cite the actual dataset source in your report.",
    ]

    # Write Excel workbook with all tables
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Overall Summary", index=False)
        quality.to_excel(writer, sheet_name="Data Quality", index=False)
        profile.to_excel(writer, sheet_name="Exit vs Retained", index=False)
        for name, table in categorical_tables.items():
            sheet = re.sub(r"[\[\]\:\*\?\/\\]", "-", name)[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
        high_balance_summary.to_excel(writer, sheet_name="High Balance Summary", index=False)
        for name, table in cross_tables.items():
            sheet = re.sub(r"[\[\]\:\*\?\/\\]", "-", name)[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
        # Basic descriptive stats
        df.describe(include="all").transpose().reset_index().rename(columns={"index":"Variable"}).to_excel(
            writer, sheet_name="Descriptive Statistics", index=False
        )
        # Apply readable column widths and freeze headers.
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                letter = column_cells[0].column_letter
                max_len = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 38)

    Path(OUTPUT_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print("ANALYSIS COMPLETED")
    print(f"Input file: {input_path.resolve()}")
    print(f"Records analyzed: {total_n:,}")
    print(f"Exited: {total_exited:,} ({overall_churn:.2f}%)")
    print(f"Retained: {total_retained:,} ({overall_retention:.2f}%)")
    print(f"Excel results: {Path(OUTPUT_EXCEL).resolve()}")
    print(f"Copy-paste report: {Path(OUTPUT_REPORT).resolve()}")
    print(f"Charts folder: {CHART_DIR.resolve()}")
    print("\nCharts created:")
    for p in charts_created:
        print(" -", p)

if __name__ == "__main__":
    main()
