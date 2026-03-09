import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# ----------------------------
# Page setup (nice defaults)
# ----------------------------
st.set_page_config(
    page_title="Fed Funds & Bond Prices Dashboard",
    page_icon="📈",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

st.title("📈 Interest Rate Risk: Fed Funds Changes & Bond Prices/Returns")
st.caption(
    "Dataset: WRDS Bond Returns (Beta) + WRDS rates_monthly. "
    "Goal: visualize how rate changes relate to bond returns/prices, especially by duration."
)

# ----------------------------
# Sidebar: data inputs
# ----------------------------
st.sidebar.header("Data inputs")

bond_file = st.sidebar.text_input("Path to WRDS bond returns CSV", value="data/wrds_bond_returns.csv")
rates_file = st.sidebar.text_input("Path to rates_monthly CSV", value="data/rates_monthly.csv")
tips_file = st.sidebar.text_input("Path to TIPS treasury CSV", value="data/tips.csv")
use_tips = st.sidebar.checkbox("Include TIPS dataset", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

max_tmt = st.sidebar.slider("Max Time-to-Maturity (years)", 1, 60, 40)
max_duration = st.sidebar.slider("Max Duration", 1, 40, 25)

sample_n = st.sidebar.slider("Max points for scatter (performance)", 5_000, 150_000, 40_000, step=5_000)

# ----------------------------
# Helpers
# ----------------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    """Try multiple parsing strategies robustly."""
    return pd.to_datetime(s, errors="coerce").dt.date

def month_floor(d: pd.Series) -> pd.Series:
    # convert to datetime64 -> floor to month -> keep as datetime64
    dt = pd.to_datetime(d, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

@st.cache_data(show_spinner=True)
def load_data(bond_path: str, rates_path: str, tips_path: str | None):
    bonds_raw = pd.read_csv(bond_path)
    rates_raw = pd.read_csv(rates_path)

    bonds_raw.columns = [c.strip().lower() for c in bonds_raw.columns]
    rates_raw.columns = [c.strip().lower() for c in rates_raw.columns]

    tips_raw = None
    if tips_path:
        tips_raw = pd.read_csv(tips_path)
        tips_raw.columns = [c.strip().lower() for c in tips_raw.columns]

    return bonds_raw, rates_raw, tips_raw

def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
              .str.replace("%", "", regex=False)
              .str.replace(",", "", regex=False)
              .str.strip(),
        errors="coerce"
    )

def preprocess(bonds_raw: pd.DataFrame, rates_raw: pd.DataFrame, tips_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    # ----------------------------
    # 1) Rates core (DEFINE rates_core HERE)
    # ----------------------------
    rates = rates_raw.copy()

    # Expect rates_monthly style: date + fedfunds (+ maybe gs10, etc.)
    if "date" not in rates.columns:
        raise ValueError("rates file must contain a 'date' column (rates_monthly export).")

    rates["date"] = pd.to_datetime(rates["date"], errors="coerce")
    rates["month"] = month_floor(rates["date"])

    for col in ["fedfunds", "gs1", "gs5", "gs10"]:
        if col in rates.columns:
            rates[col] = to_numeric_clean(rates[col])

    rates = rates.sort_values("month")
    if "fedfunds" in rates.columns:
        rates["d_ffr"] = rates["fedfunds"].diff()
    else:
        raise ValueError("rates_monthly export must include 'fedfunds'.")

    # Keep only the stuff you need for merges/plots
    keep = ["month", "fedfunds", "d_ffr"]
    for c in ["gs1", "gs5", "gs10"]:
        if c in rates.columns:
            keep.append(c)

    rates_core = rates[keep].copy()

    # ----------------------------
    # 2) Bonds core -> df (DEFINE df HERE)
    # ----------------------------
    bonds = bonds_raw.copy()

    # dates
    bonds["date"] = pd.to_datetime(bonds.get("date"), errors="coerce")
    bonds["t_date"] = pd.to_datetime(bonds.get("t_date"), errors="coerce")
    bonds["month"] = month_floor(bonds["date"].fillna(bonds["t_date"]))

    # numeric conversions
    for col in ["duration", "tmt", "yield", "price_eom", "ret_eom", "t_spread", "t_volume", "t_dvolume"]:
        if col in bonds.columns:
            bonds[col] = to_numeric_clean(bonds[col])

    # keep columns (minimal)
    bonds_core = bonds[[c for c in [
        "month", "issue_id", "cusip", "bond_type", "rating_class",
        "duration", "tmt", "yield", "price_eom", "ret_eom",
        "t_spread", "t_volume", "t_dvolume"
    ] if c in bonds.columns]].copy()

    # merge rates onto bonds
    df = bonds_core.merge(rates_core, on="month", how="left")

    df["regime"] = np.select(
        [df["d_ffr"] > 0, df["d_ffr"] < 0],
        ["Tightening (ΔFFR>0)", "Easing (ΔFFR<0)"],
        default="No change"
    )

    # (optional) If returns are in percent, convert to decimal
    # Uncomment if you see huge values like 500% in charts:
    # df["ret_eom"] = df["ret_eom"] / 100

    # ----------------------------
    # 3) Add TIPS (NOW rates_core and df exist)
    # ----------------------------
    if tips_raw is not None:
        tips = tips_raw.copy()

        # Dates
        tips["tdatdt"] = pd.to_datetime(tips.get("tdatdt"), errors="coerce")
        tips["month"] = month_floor(tips["tdatdt"])

        # Numerics
        for col in ["tmduratn", "tmyld", "tmretnua", "tmretnxs", "tmnomprc"]:
            if col in tips.columns:
                tips[col] = to_numeric_clean(tips[col])

        # choose return series
        ret_col = "tmretnua" if "tmretnua" in tips.columns else ("tmretnxs" if "tmretnxs" in tips.columns else None)
        if ret_col is None:
            raise ValueError("TIPS dataset needs 'tmretnua' or 'tmretnxs' for returns.")

        tips_std = pd.DataFrame({
            "month": tips["month"],
            "issue_id": tips.get("kycrspid", tips.get("kytreasno")),
            "cusip": tips.get("tcusip"),
            "bond_type": "TIPS",
            "rating_class": np.nan,
            "duration": tips.get("tmduratn"),
            "tmt": np.nan,
            "yield": tips.get("tmyld"),
            "ret_eom": tips[ret_col],
            "price_eom": tips.get("tmnomprc"),
        })

        tips_std = tips_std.merge(rates_core, on="month", how="left")
        df = pd.concat([df, tips_std], ignore_index=True)
        df["regime"] = np.select(
            [df["d_ffr"] > 0, df["d_ffr"] < 0],
            ["Tightening (ΔFFR>0)", "Easing (ΔFFR<0)"],
            default="No change"
        )

    # ----------------------------
    # 4) Filters + buckets (safe)
    # ----------------------------
    df = df.dropna(subset=["month"])

    if "price_eom" in df.columns:
        df = df[df["price_eom"].isna() | (df["price_eom"] > 0)]

    if "duration" in df.columns:
        df["duration"] = df["duration"].where(df["duration"] > 0)
    if "tmt" in df.columns:
        df["tmt"] = df["tmt"].where(df["tmt"] > 0)

    df["dur_bucket"] = pd.cut(
        df["duration"],
        bins=[-np.inf, 2, 5, 8, 12, np.inf],
        labels=["<=2", "2–5", "5–8", "8–12", "12+"],
    )

    df["tmt_bucket"] = pd.cut(
        df["tmt"],
        bins=[-np.inf, 1, 3, 7, 15, np.inf],
        labels=["<=1y", "1–3y", "3–7y", "7–15y", "15y+"],
    )

    return df

# ----------------------------
# Load + preprocess
# ----------------------------
try:
    with st.spinner("Loading and preprocessing data..."):
        bonds_raw, rates_raw, tips_raw = load_data(bond_file, rates_file, tips_file if use_tips else None)
        df = preprocess(bonds_raw, rates_raw, tips_raw)

    st.success(f"Loaded: {len(df):,} bond-month rows")

except Exception as e:
    st.error(f"Failed to load/preprocess: {e}")
    st.stop()

# ----------------------------
# Filter UI based on data
# ----------------------------
colA, colB, colC = st.columns([1.2, 1.2, 1.2])

with colA:
    bond_types = sorted([x for x in df.get("bond_type", pd.Series(dtype=str)).dropna().unique().tolist()])
    selected_bond_types = st.multiselect("Bond types", options=bond_types, default=bond_types)

with colB:
    rating_classes = sorted([x for x in df.get("rating_class", pd.Series(dtype=float)).dropna().unique().tolist()])
    selected_rating_classes = st.multiselect("Rating class (0=IG,1=HY)", options=rating_classes, default=rating_classes)

with colC:
    regime_options = ["Tightening (ΔFFR>0)", "Easing (ΔFFR<0)", "No change"]
    selected_regimes = st.multiselect("FFR regime", options=regime_options, default=regime_options)

f = df.copy()
if "bond_type" in f.columns and selected_bond_types:
    f = f[f["bond_type"].isin(selected_bond_types)]
if "rating_class" in f.columns and selected_rating_classes:
    f = f[f["rating_class"].isin(selected_rating_classes)]
if selected_regimes:
    f = f[f["regime"].isin(selected_regimes)]

f = f[(f["tmt"].isna()) | (f["tmt"] <= max_tmt)]
f = f[(f["duration"].isna()) | (f["duration"] <= max_duration)]
max_duration = st.sidebar.slider("Max Duration", 1, 100, 60)

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Rows (filtered)", f"{len(f):,}")
with k2:
    st.metric("Unique issues", f"{f['ISSUE_ID'].nunique() if 'ISSUE_ID' in f.columns else np.nan:,}")
with k3:
    st.metric("Mean duration", f"{np.nanmean(f['duration']):.2f}" if "duration" in f.columns else "NA")
with k4:
    st.metric("Mean ΔFFR (pp)", f"{np.nanmean(f['d_ffr']):.3f}" if "d_ffr" in f.columns else "NA")

st.markdown("---")

# ============================================================
# Visualization 1: Scatter (Duration vs Return) + interactive
# ============================================================
st.subheader("1) Duration vs Monthly Return (interactive scatter)")

y_choice = st.selectbox(
    "Y-axis for scatter",
    options=[c for c in ["ret_eom", "price_eom", "yield", "t_yld_pt"] if c in f.columns],
    index=0
)
scatter_df = f.dropna(subset=["duration", y_choice]).copy()

st.write("Filtered rows:", len(f))
st.write("Columns present:", list(f.columns)[:25])

st.write("duration non-null:", int(f["duration"].notna().sum()) if "duration" in f.columns else "missing")
st.write("ret_eom non-null:", int(f["ret_eom"].notna().sum()) if "ret_eom" in f.columns else "missing")

if "duration" in f.columns:
    st.write("duration sample:", f["duration"].dropna().head(10).tolist())
if "ret_eom" in f.columns:
    st.write("ret_eom sample:", f["ret_eom"].dropna().head(10).tolist())

if scatter_df.empty:
    st.warning("Scatter is empty. Try increasing Max Duration / Max TMT filters or check if duration/ret_eom are missing.")
    st.stop()

# sample for speed
if len(scatter_df) > sample_n:
    scatter_df = scatter_df.sample(sample_n, random_state=42)

base = alt.Chart(scatter_df).encode(
    x=alt.X("duration:Q", title="Duration"),
    y=alt.Y("ret_eom:Q", title="Monthly return (ret_eom)", axis=alt.Axis(format="%")),
    tooltip=[
        alt.Tooltip("month:T", title="Month"),
        alt.Tooltip("duration:Q", title="Duration", format=".2f"),
        alt.Tooltip("tmt:Q", title="Time to maturity", format=".2f"),
        alt.Tooltip("yield:Q", title="Yield", format=".2f"),
        alt.Tooltip("price_eom:Q", title="Price EOM", format=".2f"),
        alt.Tooltip("bond_type:N", title="Bond type"),
        alt.Tooltip("rating_class:N", title="Rating class"),
        alt.Tooltip("d_ffr:Q", title="ΔFFR", format=".3f"),
    ],
)

color_enc = alt.Color("bond_type:N", legend=alt.Legend(title="Bond type")) if "bond_type" in scatter_df.columns else alt.value("steelblue")

scatter = (
    base.mark_circle(opacity=0.35)
    .encode(color=color_enc)
    .properties(height=420)
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

# ============================================================
# Visualization 2: Time series (avg returns) by duration bucket
# ============================================================
st.subheader("2) Over time: Average Return by Duration Bucket")

ts_temp = f.dropna(subset=["month", "ret_eom", "dur_bucket"]).copy()
ts_temp = ts_temp[ts_temp["dur_bucket"].astype(str).str.lower() != "nan"]


ts_df = (
    ts_temp
    .groupby(["month", "dur_bucket"])
    .agg(mean_ret=("ret_eom", "mean"), n=("ret_eom", "size"))
    .reset_index()
)

line = (
    alt.Chart(ts_df)
    .mark_line()
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("mean_ret:Q", title="Mean monthly return", axis=alt.Axis(format="%")),
        color=alt.Color("dur_bucket:N", title="Duration bucket"),
        tooltip=[
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("dur_bucket:N", title="Bucket"),
            alt.Tooltip("mean_ret:Q", title="Mean return", format=".3%")
        ],
    )
    .properties(height=380)
)

st.altair_chart(line, use_container_width=True)

# ============================================================
# Visualization 3: Heatmap (Tightening sensitivity proxy)
#   mean returns under tightening by (duration bucket x rating)
# ============================================================
st.subheader("3) Tightening map: Mean Returns when ΔFFR > 0 (Duration × Rating)")

tight = f[(f["d_ffr"] > 0) & f["dur_bucket"].notna() & f["ret_eom"].notna()].copy()

if "rating_class" not in tight.columns:
    st.info("No rating_class column found — skipping heatmap.")
else:
    heat_df = (
        tight.groupby(["dur_bucket", "rating_class"], as_index=False)
        .agg(mean_ret=("ret_eom", "mean"), n=("ret_eom", "size"))
    )

    heat = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("dur_bucket:N", title="Duration bucket"),
            y=alt.Y("rating_class:N", title="Rating class (0=IG,1=HY)"),
            color=alt.Color("mean_ret:Q", title="Mean return (tightening)", legend=alt.Legend(format="%")),
            tooltip=[
                alt.Tooltip("dur_bucket:N", title="Duration bucket"),
                alt.Tooltip("rating_class:N", title="Rating class"),
                alt.Tooltip("mean_ret:Q", title="Mean return", format=".3%"),
                alt.Tooltip("n:Q", title="N"),
            ],
        )
        .properties(height=240)
    )

    st.altair_chart(heat, use_container_width=True)

# ============================================================
# Visualization 4: Fed Funds changes vs average bond returns
#   (monthly) colored by maturity bucket + smooth trend
# ============================================================
st.subheader("4) Monthly ΔFFR vs Avg Bond Return (by maturity bucket)")

bubble_temp = f.dropna(
    subset=["month", "d_ffr", "ret_eom", "tmt_bucket"]
).copy()

# remove nan bucket explicitly
bubble_temp = bubble_temp[bubble_temp["tmt_bucket"].notna()]

bubble_base_df = (
    bubble_temp
    .groupby(["month", "tmt_bucket"])
    .agg(
        avg_ret=("ret_eom", "mean"),
        avg_dffr=("d_ffr", "mean"),
        n=("ret_eom", "size")
    )
    .reset_index()
)

points = (
    alt.Chart(bubble_base_df)
    .mark_circle(opacity=0.55)
    .encode(
        x=alt.X("avg_dffr:Q", title="Monthly ΔFFR (percentage points)"),
        y=alt.Y("avg_ret:Q", title="Avg monthly return", axis=alt.Axis(format="%")),
        size=alt.Size("n:Q", title="Number of bonds", scale=alt.Scale(range=[20, 800])),
        color=alt.Color("tmt_bucket:N", title="Maturity bucket"),
        tooltip=[
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("tmt_bucket:N", title="Maturity bucket"),
            alt.Tooltip("avg_dffr:Q", title="Avg ΔFFR", format=".3f"),
            alt.Tooltip("avg_ret:Q", title="Avg return", format=".3%"),
            alt.Tooltip("n:Q", title="N"),
        ],
    )
)

trend = (
    alt.Chart(bubble_base_df)
    .transform_loess("avg_dffr", "avg_ret", groupby=["tmt_bucket"])
    .mark_line()
    .encode(
        x="avg_dffr:Q",
        y="avg_ret:Q",
        color="tmt_bucket:N",
    )
)

st.altair_chart((points + trend).properties(height=420), use_container_width=True)

# ----------------------------
# Data preview + download
# ----------------------------
st.markdown("---")
with st.expander("Preview filtered data"):
    st.dataframe(f.head(50), use_container_width=True)

with st.expander("Download filtered dataset (CSV)"):
    out_csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", out_csv, file_name="filtered_bond_rates.csv", mime="text/csv")

st.caption("Tip: If your bond file is huge, export a smaller date range from WRDS first, or increase sampling in the sidebar.")
