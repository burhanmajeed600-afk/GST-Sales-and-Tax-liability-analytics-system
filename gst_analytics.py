#!/usr/bin/env python3
"""
GST Sales & Tax Liability Analytics System
==========================================
Processes an invoice dataset and produces:
  1. CGST / SGST / IGST per invoice line (intra- vs inter-state logic)
  2. Tax summary by product x tax slab
  3. Month-wise GST liability summary (and month x slab)
  4. Invoice error report (data-quality + GST compliance checks)
  5. Top products by revenue and by tax contribution
  6. Excel report + PNG charts

Usage
-----
  python gst_analytics.py --demo                       # generate sample data & run
  python gst_analytics.py --input invoices.csv         # run on your own file
  python gst_analytics.py --input invoices.xlsx --seller-state Haryana --top 15

Expected input columns (names are case-insensitive; extra columns are kept):
  invoice_no, invoice_date, customer_name, customer_gstin (blank for B2C),
  place_of_supply (buyer state), product_name, hsn_code, quantity,
  unit_price, discount, gst_rate
Optional:
  tax_charged   -> total GST actually charged on the invoice line; used to detect
                   under/over-charged tax.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VALID_SLABS = [0, 0.1, 0.25, 3, 5, 12, 18, 28]  # GST rate slabs in %
DEFAULT_SELLER_STATE = "Haryana"
TOLERANCE = 1.0  # Rs. tolerance when comparing charged vs computed tax

REQUIRED_COLS = [
    "invoice_no", "invoice_date", "customer_name", "customer_gstin",
    "place_of_supply", "product_name", "hsn_code", "quantity",
    "unit_price", "discount", "gst_rate",
]

STATE_CODES = {
    "Jammu and Kashmir": "01", "Himachal Pradesh": "02", "Punjab": "03",
    "Chandigarh": "04", "Uttarakhand": "05", "Haryana": "06", "Delhi": "07",
    "Rajasthan": "08", "Uttar Pradesh": "09", "Bihar": "10", "Sikkim": "11",
    "Arunachal Pradesh": "12", "Nagaland": "13", "Manipur": "14",
    "Mizoram": "15", "Tripura": "16", "Meghalaya": "17", "Assam": "18",
    "West Bengal": "19", "Jharkhand": "20", "Odisha": "21",
    "Chhattisgarh": "22", "Madhya Pradesh": "23", "Gujarat": "24",
    "Maharashtra": "27", "Karnataka": "29", "Goa": "30", "Kerala": "32",
    "Tamil Nadu": "33", "Puducherry": "34", "Telangana": "36",
    "Andhra Pradesh": "37", "Ladakh": "38",
}

GSTIN_REGEX = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------- #
# GSTIN helpers
# --------------------------------------------------------------------------- #
def gstin_checksum_char(first14: str) -> str:
    """Compute the GSTIN check character (mod-36 algorithm)."""
    total = 0
    for i, ch in enumerate(first14):
        prod = _B36.index(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 36 + prod % 36
    return _B36[(36 - total % 36) % 36]


def is_valid_gstin(g: str) -> bool:
    g = str(g).strip().upper()
    return bool(GSTIN_REGEX.match(g)) and gstin_checksum_char(g[:14]) == g[14]


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Input file not found: {path}")
    # keep identifier columns as text so HSN codes / GSTINs are never turned into numbers
    text_cols = {"hsn_code": str, "HSN_Code": str, "invoice_no": str, "customer_gstin": str}
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=text_cols)
    else:
        df = pd.read_csv(p, dtype=text_cols)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    df = df.copy()
    df["source_row"] = df.index + 2  # row number as seen in the CSV/Excel file
    for c in ("invoice_no", "customer_name", "customer_gstin",
              "place_of_supply", "product_name", "hsn_code"):
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    df["customer_gstin"] = df["customer_gstin"].str.upper()
    df["raw_date"] = df["invoice_date"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce", dayfirst=True)
    for c in ("quantity", "unit_price", "discount", "gst_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["discount"] = df["discount"].fillna(0)
    if "tax_charged" in df.columns:
        df["tax_charged"] = pd.to_numeric(df["tax_charged"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, seller_state: str) -> pd.DataFrame:
    """Return a long-form error table: one row per (invoice line, error)."""
    errors = []

    def flag(mask, etype, severity, detail):
        mask = mask.fillna(False)
        if not mask.any():
            return
        sub = df.loc[mask, ["source_row", "invoice_no", "product_name"]].copy()
        sub["error_type"] = etype
        sub["severity"] = severity  # Critical rows are excluded from liability
        sub["detail"] = detail(df.loc[mask]) if callable(detail) else detail
        errors.append(sub)

    # --- Critical: data cannot be trusted / cannot be assigned to a period ---
    flag(df["invoice_no"].isna(), "Missing invoice number", "Critical",
         "Invoice number is blank")
    flag(df["invoice_date"].isna(), "Invalid / missing date", "Critical",
         lambda d: "Unparseable date: " + d["raw_date"].astype(str))
    flag(~df["gst_rate"].isin(VALID_SLABS), "Invalid GST rate", "Critical",
         lambda d: "Rate " + d["gst_rate"].astype(str) + "% is not a valid slab")
    flag(df["quantity"].isna() | (df["quantity"] <= 0), "Invalid quantity",
         "Critical", lambda d: "Quantity = " + d["quantity"].astype(str))
    flag(df["unit_price"].isna() | (df["unit_price"] < 0), "Invalid unit price",
         "Critical", lambda d: "Unit price = " + d["unit_price"].astype(str))
    gross = df["quantity"] * df["unit_price"]
    flag((df["discount"] > gross) & (gross >= 0), "Discount exceeds gross value", "Critical",
         "Discount is greater than quantity x unit price")
    flag(df["discount"] < 0, "Negative discount", "Critical", "Discount < 0")
    flag(df["place_of_supply"].isna() | ~df["place_of_supply"].isin(STATE_CODES),
         "Invalid place of supply", "Critical",
         lambda d: "State: " + d["place_of_supply"].astype(str))

    dup_cols = ["invoice_no", "invoice_date", "product_name", "quantity", "unit_price"]
    dup = df.duplicated(subset=dup_cols, keep="first") & df["invoice_no"].notna()
    flag(dup, "Duplicate invoice line", "Critical",
         "Identical line already exists for this invoice (counted once)")

    # --- Warnings: compliance issues that need review but still carry tax ---
    has_gstin = df["customer_gstin"].notna()
    bad_gstin = has_gstin & ~df["customer_gstin"].apply(
        lambda g: is_valid_gstin(g) if pd.notna(g) else True)
    flag(bad_gstin, "Invalid GSTIN", "Warning",
         lambda d: "GSTIN fails format/checksum: " + d["customer_gstin"].astype(str))

    pos_code = df["place_of_supply"].map(STATE_CODES)
    mismatch = has_gstin & pos_code.notna() & (df["customer_gstin"].str[:2] != pos_code)
    flag(mismatch, "GSTIN state != place of supply", "Warning",
         lambda d: "GSTIN state code " + d["customer_gstin"].str[:2].astype(str)
         + " vs place of supply " + d["place_of_supply"].astype(str))

    flag(df["hsn_code"].isna(), "Missing HSN", "Warning", "HSN/SAC code is blank")
    hsn_bad = df["hsn_code"].notna() & ~df["hsn_code"].fillna("").str.fullmatch(r"\d{4}|\d{6}|\d{8}")
    flag(hsn_bad, "Invalid HSN format", "Warning",
         lambda d: "HSN must be 4/6/8 digits: " + d["hsn_code"].astype(str))

    # Same invoice number carrying different dates / customers
    for col, label in (("invoice_date", "dates"), ("customer_name", "customers")):
        n = df.groupby("invoice_no")[col].transform("nunique")
        flag(n > 1, f"Invoice has multiple {label}", "Warning",
             f"Same invoice number appears with different {label}")

    # Charged tax vs computed tax
    if "tax_charged" in df.columns:
        computed = (gross - df["discount"]) * df["gst_rate"] / 100
        diff = df["tax_charged"] - computed
        flag((diff.abs() > TOLERANCE) & df["gst_rate"].isin(VALID_SLABS), "Tax mismatch", "Warning",
             lambda d: "Charged " + d["tax_charged"].round(2).astype(str)
             + " vs expected " + computed.loc[d.index].round(2).astype(str))

    if not errors:
        return pd.DataFrame(columns=["source_row", "invoice_no", "product_name",
                                     "error_type", "severity", "detail"])
    out = pd.concat(errors, ignore_index=True)
    out["invoice_no"] = out["invoice_no"].astype(object).where(out["invoice_no"].notna(), "(blank)")
    return out.sort_values(["severity", "source_row"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tax calculation
# --------------------------------------------------------------------------- #
def calculate_tax(df: pd.DataFrame, errors: pd.DataFrame, seller_state: str) -> pd.DataFrame:
    """Compute taxable value and CGST/SGST/IGST for every clean line."""
    bad_rows = set(errors.loc[errors["severity"] == "Critical", "source_row"])
    d = df[~df["source_row"].isin(bad_rows)].copy()

    d["gross_value"] = d["quantity"] * d["unit_price"]
    d["taxable_value"] = (d["gross_value"] - d["discount"]).round(2)
    d["intra_state"] = d["place_of_supply"].str.lower() == seller_state.lower()
    rate = d["gst_rate"]
    d["cgst_rate"] = np.where(d["intra_state"], rate / 2, 0.0)
    d["sgst_rate"] = np.where(d["intra_state"], rate / 2, 0.0)
    d["igst_rate"] = np.where(d["intra_state"], 0.0, rate)
    d["cgst"] = (d["taxable_value"] * d["cgst_rate"] / 100).round(2)
    d["sgst"] = (d["taxable_value"] * d["sgst_rate"] / 100).round(2)
    d["igst"] = (d["taxable_value"] * d["igst_rate"] / 100).round(2)
    d["total_tax"] = d["cgst"] + d["sgst"] + d["igst"]
    d["invoice_value"] = d["taxable_value"] + d["total_tax"]
    d["month"] = d["invoice_date"].dt.to_period("M").astype(str)
    d["tax_slab"] = d["gst_rate"].map(lambda r: f"{r:g}%")
    d["supply_type"] = np.where(d["intra_state"], "Intra-state (CGST+SGST)",
                                "Inter-state (IGST)")
    return d.sort_values(["invoice_date", "invoice_no"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
TAX_COLS = ["taxable_value", "cgst", "sgst", "igst", "total_tax", "invoice_value"]


def product_slab_summary(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.groupby(["product_name", "hsn_code", "gst_rate"], dropna=False)
          .agg(units_sold=("quantity", "sum"), lines=("invoice_no", "size"),
               **{c: (c, "sum") for c in TAX_COLS})
          .reset_index().rename(columns={"gst_rate": "slab_%"}))
    return g.sort_values(["slab_%", "taxable_value"], ascending=[True, False]).round(2)


def monthly_summary(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.groupby("month")
          .agg(invoices=("invoice_no", "nunique"),
               **{c: (c, "sum") for c in TAX_COLS}).reset_index())
    g["mom_tax_change_%"] = (g["total_tax"].pct_change() * 100).round(1)
    total = g[["invoices"] + TAX_COLS].sum()
    total["invoices"] = d["invoice_no"].nunique()
    g.loc[len(g)] = ["TOTAL", *total.values, np.nan]
    g["invoices"] = g["invoices"].astype(int)
    return g.round(2)


def monthly_slab_summary(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.groupby(["month", "tax_slab"])
          .agg(**{c: (c, "sum") for c in TAX_COLS}).reset_index())
    g["_s"] = g["tax_slab"].str.rstrip("%").astype(float)
    return g.sort_values(["month", "_s"]).drop(columns="_s").round(2)


def top_products(d: pd.DataFrame, n: int):
    g = (d.groupby("product_name")
          .agg(taxable_value=("taxable_value", "sum"), total_tax=("total_tax", "sum"),
               units_sold=("quantity", "sum")).reset_index())
    rev = g.sort_values("taxable_value", ascending=False).head(n).copy()
    rev["revenue_share_%"] = (rev["taxable_value"] / g["taxable_value"].sum() * 100).round(2)
    rev["cumulative_%"] = rev["revenue_share_%"].cumsum().round(2)
    tax = g.sort_values("total_tax", ascending=False).head(n).copy()
    tax["tax_share_%"] = (tax["total_tax"] / g["total_tax"].sum() * 100).round(2)
    tax["cumulative_%"] = tax["tax_share_%"].cumsum().round(2)
    return rev.round(2), tax.round(2)


def error_summary(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["error_type", "severity", "count"])
    return (errors.groupby(["severity", "error_type"]).size().reset_index(name="count")
            .sort_values(["severity", "count"], ascending=[True, False]))


# --------------------------------------------------------------------------- #
# Output: Excel + charts
# --------------------------------------------------------------------------- #
def write_excel(path, sheets: dict):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(name="Arial", bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for ci, col in enumerate(frame.columns, 1):
                width = max([len(str(col))] + [len(str(v)) for v in frame[col].head(200)])
                ws.column_dimensions[get_column_letter(ci)].width = min(width + 3, 45)
                is_num = pd.api.types.is_float_dtype(frame[col])
                for row in ws.iter_rows(min_row=2, min_col=ci, max_col=ci):
                    c = row[0]
                    c.font = Font(name="Arial", size=10)
                    if is_num:
                        c.number_format = "#,##0.00"
            if name == "Monthly_Summary":  # bold the TOTAL row
                for c in ws[ws.max_row]:
                    c.font = Font(name="Arial", bold=True)
            if name in ("Errors", "Invoice_Details"):
                ws.auto_filter.ref = ws.dimensions


def make_charts(outdir: Path, monthly, rev, tax, d):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping charts")
        return []
    files = []
    m = monthly[monthly["month"] != "TOTAL"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(m["month"], m["cgst"], label="CGST")
    ax.bar(m["month"], m["sgst"], bottom=m["cgst"], label="SGST")
    ax.bar(m["month"], m["igst"], bottom=m["cgst"] + m["sgst"], label="IGST")
    ax.set_title("Month-wise GST Liability"); ax.set_ylabel("Rs.")
    ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
    f = outdir / "chart_monthly_liability.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(rev["product_name"][::-1], rev["taxable_value"][::-1], color="#1F4E78")
    axes[0].set_title("Top Products by Revenue (Taxable Value)")
    axes[1].barh(tax["product_name"][::-1], tax["total_tax"][::-1], color="#C55A11")
    axes[1].set_title("Top Products by GST Contribution")
    plt.tight_layout()
    f = outdir / "chart_top_products.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    slab = d.groupby("tax_slab")["total_tax"].sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(slab, labels=slab.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("GST Collected by Tax Slab"); plt.tight_layout()
    f = outdir / "chart_slab_share.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    return files


# --------------------------------------------------------------------------- #
# Demo data generator (with deliberately injected errors)
# --------------------------------------------------------------------------- #
def make_gstin(state_code: str, pan: str, entity="1") -> str:
    first14 = f"{state_code}{pan}{entity}Z"
    return first14 + gstin_checksum_char(first14)


def generate_demo(path: str, n_invoices=1500, seed=42):
    rng = np.random.default_rng(seed)
    products = [  # name, hsn, slab, price
        ("Basmati Rice 5kg", "1006", 5, 520), ("Wheat Flour 10kg", "1101", 0, 410),
        ("Cooking Oil 1L", "1512", 5, 160), ("Packaged Biscuits", "1905", 18, 35),
        ("Soap Bar", "3401", 18, 38), ("LED Bulb 9W", "8539", 12, 95),
        ("Mobile Charger", "8504", 18, 450), ("Bluetooth Speaker", "8518", 18, 1800),
        ("Cotton T-Shirt", "6109", 5, 399), ("Branded Jeans", "6203", 12, 1499),
        ("Steel Cookware Set", "7323", 12, 2200), ("Air Conditioner 1.5T", "8415", 28, 38500),
        ("Refrigerator 250L", "8418", 28, 24500), ("Aerated Drink 2L", "2202", 28, 95),
        ("Notebook (pack of 6)", "4820", 12, 240), ("Paracetamol Strip", "3004", 12, 28),
    ]
    customers = []
    for i, (st, code) in enumerate([("Haryana", "06"), ("Delhi", "07"), ("Punjab", "03"),
                                    ("Rajasthan", "08"), ("Uttar Pradesh", "09"),
                                    ("Maharashtra", "27"), ("Karnataka", "29")]):
        for j in range(4):
            pan = "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 5)) + \
                  "".join(rng.choice(list("0123456789"), 4)) + \
                  str(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ")))
            customers.append((f"{st} Traders {j+1}", make_gstin(code, pan), st))

    rows = []
    start = pd.Timestamp("2025-04-01")
    for inv in range(1, n_invoices + 1):
        cust = customers[rng.integers(len(customers))]
        date = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        for _ in range(int(rng.integers(1, 5))):
            p = products[rng.integers(len(products))]
            qty = int(rng.integers(1, 40 if p[3] < 1000 else 6))
            price = round(p[3] * rng.uniform(0.95, 1.05), 2)
            disc = round(qty * price * rng.choice([0, 0, 0.02, 0.05]), 2)
            taxable = qty * price - disc
            rows.append(dict(
                invoice_no=f"INV-{inv:05d}", invoice_date=date.strftime("%d-%m-%Y"),
                customer_name=cust[0], customer_gstin=cust[1], place_of_supply=cust[2],
                product_name=p[0], hsn_code=p[1], quantity=qty, unit_price=price,
                discount=disc, gst_rate=p[2], tax_charged=round(taxable * p[2] / 100, 2)))
    df = pd.DataFrame(rows)

    # ---- inject realistic errors ----
    idx = rng.choice(len(df), 90, replace=False)
    df.loc[idx[0:10], "invoice_no"] = np.nan
    df.loc[idx[10:20], "invoice_date"] = "31-02-2025"
    df.loc[idx[20:30], "gst_rate"] = 15
    df.loc[idx[30:38], "quantity"] = -2
    df.loc[idx[38:48], "customer_gstin"] = "06ABCDE1234F1Z0"            # bad checksum
    df.loc[idx[48:56], "hsn_code"] = np.nan
    df.loc[idx[56:64], "tax_charged"] = df.loc[idx[56:64], "tax_charged"] * 0.5  # under-charged
    df.loc[idx[64:70], "place_of_supply"] = "Atlantis"
    df.loc[idx[70:76], "discount"] = df.loc[idx[70:76], "quantity"] * df.loc[idx[70:76], "unit_price"] * 1.5
    df.loc[idx[76:82], "customer_gstin"] = df["customer_gstin"].iloc[0]  # state mismatch
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)  # duplicates
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Demo data written: {path} ({len(df)} lines)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="GST Sales & Tax Liability Analytics")
    ap.add_argument("--input", help="Invoice CSV/XLSX file")
    ap.add_argument("--demo", action="store_true", help="Generate sample data and run on it")
    ap.add_argument("--seller-state", default=DEFAULT_SELLER_STATE,
                    help="Your registered state (decides CGST+SGST vs IGST)")
    ap.add_argument("--top", type=int, default=10, help="Top-N products to list")
    ap.add_argument("--outdir", default="gst_output", help="Output folder")
    a = ap.parse_args()

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if a.demo:
        a.input = str(outdir / "sample_invoices.csv")
        generate_demo(a.input)
    if not a.input:
        ap.error("provide --input FILE or --demo")
    if a.seller_state not in STATE_CODES:
        ap.error(f"Unknown seller state '{a.seller_state}'. Choose from: {', '.join(STATE_CODES)}")

    df = load_data(a.input)
    errors = validate(df, a.seller_state)
    d = calculate_tax(df, errors, a.seller_state)

    prod_slab = product_slab_summary(d)
    monthly = monthly_summary(d)
    month_slab = monthly_slab_summary(d)
    rev, tax = top_products(d, a.top)
    err_sum = error_summary(errors)

    detail_cols = ["invoice_no", "invoice_date", "customer_name", "customer_gstin",
                   "place_of_supply", "supply_type", "product_name", "hsn_code",
                   "quantity", "unit_price", "discount", "taxable_value", "gst_rate",
                   "cgst", "sgst", "igst", "total_tax", "invoice_value", "month"]
    dd = d[detail_cols].copy()
    dd["invoice_date"] = dd["invoice_date"].dt.strftime("%Y-%m-%d")

    xlsx = outdir / "GST_Liability_Report.xlsx"
    write_excel(xlsx, {
        "Monthly_Summary": monthly, "Monthly_by_Slab": month_slab,
        "Product_by_Slab": prod_slab, "Top_by_Revenue": rev, "Top_by_Tax": tax,
        "Error_Summary": err_sum, "Errors": errors, "Invoice_Details": dd,
    })
    errors.to_csv(outdir / "invoice_errors.csv", index=False)
    charts = make_charts(outdir, monthly, rev, tax, d)

    # ---- console summary ----
    t = monthly.iloc[-1]
    print("\n" + "=" * 62)
    print(" GST LIABILITY SUMMARY")
    print("=" * 62)
    print(f" Lines read            : {len(df):,}")
    print(f" Lines processed       : {len(d):,}")
    print(f" Lines excluded (crit) : {len(df) - len(d):,}")
    print(f" Taxable value         : Rs. {t['taxable_value']:,.2f}")
    print(f" CGST / SGST / IGST    : {t['cgst']:,.2f} / {t['sgst']:,.2f} / {t['igst']:,.2f}")
    print(f" Total GST liability   : Rs. {t['total_tax']:,.2f}")
    print("\n Month-wise liability:")
    print(monthly[["month", "invoices", "taxable_value", "cgst", "sgst", "igst", "total_tax"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n Top products by revenue:")
    print(rev[["product_name", "taxable_value", "revenue_share_%"]].head(5)
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n Top products by GST contribution:")
    print(tax[["product_name", "total_tax", "tax_share_%"]].head(5)
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n Errors found:")
    print(err_sum.to_string(index=False) if not err_sum.empty else "  none")
    print(f"\n Report : {xlsx}")
    for c in charts:
        print(f" Chart  : {c}")


if __name__ == "__main__":
    main()
