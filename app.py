"""
GST Sales & Tax Liability Analytics - Streamlit mini app
Run:  streamlit run app.py
Needs gst_analytics.py in the same folder.
"""
import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import gst_analytics as g

st.set_page_config(page_title="GST Analytics", page_icon="🧾", layout="wide")


# --------------------------------------------------------------------------- #
# Cached pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Processing invoices...")
def run_pipeline(file_bytes: bytes, suffix: str, seller_state: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        raw = g.load_data(path)
    except SystemExit as e:  # load_data exits on bad files; show it in the UI instead
        return None, None, None, str(e)
    finally:
        Path(path).unlink(missing_ok=True)
    errors = g.validate(raw, seller_state)
    clean = g.calculate_tax(raw, errors, seller_state)
    return raw, errors, clean, None


@st.cache_data
def demo_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.csv"
        g.generate_demo(str(p))
        return p.read_bytes()


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    g.write_excel(buf, sheets)
    return buf.getvalue()


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🧾 GST Analytics")
upload = st.sidebar.file_uploader("Upload invoice file", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Use demo data", value=upload is None)
seller_state = st.sidebar.selectbox(
    "Your registered state", sorted(g.STATE_CODES),
    index=sorted(g.STATE_CODES).index(g.DEFAULT_SELLER_STATE),
    help="Sales within this state = CGST + SGST. Other states = IGST.")
top_n = st.sidebar.slider("Top-N products", 5, 25, 10)

if upload is not None:
    file_bytes, suffix = upload.getvalue(), Path(upload.name).suffix
elif use_demo:
    file_bytes, suffix = demo_bytes(), ".csv"
else:
    st.title("GST Sales & Tax Liability Analytics")
    st.info("Upload an invoice CSV/Excel file in the sidebar, or tick **Use demo data**.")
    st.markdown("**Required columns:** `" + "`, `".join(g.REQUIRED_COLS) + "`  \n"
                "**Optional:** `tax_charged` (enables tax mismatch detection)")
    st.stop()

raw, errors, clean, err_msg = run_pipeline(file_bytes, suffix, seller_state)
if err_msg:
    st.error(err_msg)
    st.stop()
if clean.empty:
    st.warning("No valid invoice lines remain after validation. Check the Errors tab.")

# --------------------------------------------------------------------------- #
# Filters (apply to the clean data)
# --------------------------------------------------------------------------- #
months = sorted(clean["month"].unique())
slabs = sorted(clean["tax_slab"].unique(), key=lambda s: float(s.rstrip("%")))
sel_months = st.sidebar.multiselect("Months", months, default=months)
sel_slabs = st.sidebar.multiselect("Tax slabs", slabs, default=slabs)
supply = st.sidebar.radio("Supply type", ["All", "Intra-state (CGST+SGST)", "Inter-state (IGST)"])

d = clean[clean["month"].isin(sel_months) & clean["tax_slab"].isin(sel_slabs)]
if supply != "All":
    d = d[d["supply_type"] == supply]

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("GST Sales & Tax Liability Analytics")
st.caption(f"Seller state: **{seller_state}**  |  {len(raw):,} lines read, "
           f"{len(clean):,} processed, {len(raw) - len(clean):,} excluded for critical errors")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Taxable value", inr(d["taxable_value"].sum()))
c2.metric("CGST", inr(d["cgst"].sum()))
c3.metric("SGST", inr(d["sgst"].sum()))
c4.metric("IGST", inr(d["igst"].sum()))
c5.metric("Total GST liability", inr(d["total_tax"].sum()))

n_crit = int((errors["severity"] == "Critical").sum()) if not errors.empty else 0
n_warn = int((errors["severity"] == "Warning").sum()) if not errors.empty else 0
if n_crit:
    st.error(f"{n_crit} critical issue(s) found; those lines are excluded from the liability above.")
if n_warn:
    st.warning(f"{n_warn} warning(s) need review before filing.")

if d.empty:
    st.info("No data for the selected filters.")
    st.stop()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_over, tab_month, tab_prod, tab_top, tab_err, tab_det = st.tabs(
    ["📊 Overview", "📅 Monthly", "📦 Product × Slab", "🏆 Top Products",
     "⚠️ Errors", "📄 Invoices"])

monthly = g.monthly_summary(d)
month_slab = g.monthly_slab_summary(d)
prod_slab = g.product_slab_summary(d)
rev, tax = g.top_products(d, top_n)
err_sum = g.error_summary(errors)

with tab_over:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Month-wise GST liability")
        m = monthly[monthly["month"] != "TOTAL"].set_index("month")
        st.bar_chart(m[["cgst", "sgst", "igst"]])
    with right:
        st.subheader("Tax by slab")
        st.bar_chart(d.groupby("tax_slab")["total_tax"].sum()
                     .reindex(slabs).dropna())
    st.subheader("Taxable value trend")
    st.line_chart(m[["taxable_value", "total_tax"]])

with tab_month:
    st.subheader("Monthly liability summary")
    st.dataframe(monthly, use_container_width=True, hide_index=True)
    st.subheader("Month × tax slab")
    pivot = month_slab.pivot_table(index="month", columns="tax_slab",
                                   values="total_tax", aggfunc="sum", fill_value=0)
    pivot = pivot[sorted(pivot.columns, key=lambda s: float(s.rstrip("%")))]
    st.dataframe(pivot.style.format("{:,.2f}"), use_container_width=True)
    with st.expander("Full month × slab detail"):
        st.dataframe(month_slab, use_container_width=True, hide_index=True)

with tab_prod:
    st.subheader("Tax summary by product and slab")
    q = st.text_input("Search product")
    view = prod_slab[prod_slab["product_name"].str.contains(q, case=False, na=False)] if q else prod_slab
    st.dataframe(view, use_container_width=True, hide_index=True)

with tab_top:
    a, b = st.columns(2)
    with a:
        st.subheader(f"Top {top_n} by revenue")
        st.bar_chart(rev.set_index("product_name")["taxable_value"], horizontal=True)
        st.dataframe(rev, use_container_width=True, hide_index=True)
    with b:
        st.subheader(f"Top {top_n} by GST contribution")
        st.bar_chart(tax.set_index("product_name")["total_tax"], horizontal=True)
        st.dataframe(tax, use_container_width=True, hide_index=True)

with tab_err:
    if errors.empty:
        st.success("No errors found. 🎉")
    else:
        st.subheader("Error summary")
        st.dataframe(err_sum, use_container_width=True, hide_index=True)
        f1, f2 = st.columns(2)
        sev = f1.multiselect("Severity", ["Critical", "Warning"], default=["Critical", "Warning"])
        types = f2.multiselect("Error type", sorted(errors["error_type"].unique()))
        ev = errors[errors["severity"].isin(sev)]
        if types:
            ev = ev[ev["error_type"].isin(types)]
        st.caption(f"{len(ev):,} issue(s). 'source_row' is the row number in your file.")
        st.dataframe(ev, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download errors (CSV)", ev.to_csv(index=False).encode(),
                           "invoice_errors.csv", "text/csv")

with tab_det:
    st.subheader("Invoice-level tax calculation")
    cols = ["invoice_no", "invoice_date", "customer_name", "place_of_supply", "supply_type",
            "product_name", "hsn_code", "quantity", "unit_price", "discount", "taxable_value",
            "gst_rate", "cgst", "sgst", "igst", "total_tax", "invoice_value"]
    detail = d[cols].copy()
    detail["invoice_date"] = detail["invoice_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(detail, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Full Excel export
# --------------------------------------------------------------------------- #
st.sidebar.divider()
detail_export = d[["invoice_no", "invoice_date", "customer_name", "customer_gstin",
                   "place_of_supply", "supply_type", "product_name", "hsn_code", "quantity",
                   "unit_price", "discount", "taxable_value", "gst_rate", "cgst", "sgst",
                   "igst", "total_tax", "invoice_value", "month"]].copy()
detail_export["invoice_date"] = detail_export["invoice_date"].dt.strftime("%Y-%m-%d")
st.sidebar.download_button(
    "⬇️ Download Excel report",
    to_excel_bytes({
        "Monthly_Summary": monthly, "Monthly_by_Slab": month_slab,
        "Product_by_Slab": prod_slab, "Top_by_Revenue": rev, "Top_by_Tax": tax,
        "Error_Summary": err_sum, "Errors": errors, "Invoice_Details": detail_export,
    }),
    "GST_Liability_Report.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
