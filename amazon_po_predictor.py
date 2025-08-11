"""
Amazon Replenishment Forecast — Predict Weekly Amazon POs (Sell-In)

Current behavior:
- Amazon Sell-Out file is AUTHORITATIVE for Forecast_Units when present.
- CSV & Excel supported; Excel requires openpyxl.
- NEW (this rev): PO logic uses rolling WOH method:
    PO_t = max(0, sum_{k=1..WOH} Forecast_{t+k} - (OnHand_t_after_arrivals - Forecast_t))
"""

import os
import re
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ Optional libs ------------------------
PLOTLY_INSTALLED = False
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    pass

PROPHET_INSTALLED = False
try:
    from prophet import Prophet  # noqa: F401
    PROPHET_INSTALLED = True
except ImportError:
    try:
        from fbprophet import Prophet  # noqa: F401
        PROPHET_INSTALLED = True
    except ImportError:
        pass

ARIMA_INSTALLED = False
try:
    from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
    ARIMA_INSTALLED = True
except ImportError:
    pass

# Excel engine check
OPENPYXL_INSTALLED = False
try:
    import openpyxl  # noqa: F401
    OPENPYXL_INSTALLED = True
except Exception:
    OPENPYXL_INSTALLED = False

# ------------------------ Branding ------------------------
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# ------------------------ Page ------------------------
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon="🛒", layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>",
    unsafe_allow_html=True,
)

# ------------------------ Sidebar ------------------------
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history (CSV/Excel)", type=["csv", "xlsx", "xls"])
fcst_file  = st.sidebar.file_uploader("Amazon Sell-Out Forecast (CSV/Excel)", type=["csv", "xlsx", "xls"])
inv_file   = st.sidebar.file_uploader("Amazon Inventory (optional, CSV/Excel)", type=["csv", "xlsx", "xls"])

projection_type = st.sidebar.selectbox("Projection Type (primary Y-axis)", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Fallback Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if not model_opts:    model_opts.append("Naive (last value)")
model_choice = st.sidebar.selectbox("Forecast Model (used ONLY if Amazon file missing)", model_opts)

woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = int(st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12))
lead_time_weeks = int(st.sidebar.number_input("Lead Time (weeks) — PO arrival delay", min_value=0, max_value=26, value=2))

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the forecast and predicted POs.")
    st.stop()

# ------------------------ Defaults ------------------------
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_up    = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
sales_path = sales_file if sales_file is not None else (default_sales if os.path.exists(default_sales) else None)
up_path    = fcst_file  if fcst_file  is not None else (default_up    if os.path.exists(default_up)    else None)
inv_path   = inv_file   if inv_file   is not None else None

if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# ------------------------ Helpers ------------------------
def _maybe_seek_start(obj) -> None:
    if hasattr(obj, "seek"):
        try: obj.seek(0)
        except Exception: pass

def get_ext(obj) -> str:
    name = getattr(obj, "name", None) or str(obj)
    return os.path.splitext(name)[-1].lower()

def load_any_table(obj, *, sep=None, skiprows=None, header="infer", sheet_name=0):
    _maybe_seek_start(obj)
    ext = get_ext(obj)
    if ext in [".xlsx", ".xls"]:
        if not OPENPYXL_INSTALLED:
            st.error("Excel detected but 'openpyxl' is not installed. Add **openpyxl>=3.1.2** or upload CSV.")
            st.stop()
        return pd.read_excel(obj, sheet_name=sheet_name, header=header)
    return pd.read_csv(obj, sep=sep, engine="python", skiprows=skiprows, header=header)

def try_parse_date_string(s: str, prefer_year: Optional[int] = None) -> Optional[pd.Timestamp]:
    s = str(s).strip()
    if prefer_year and re.search(r"\b\d{4}\b", s) is None and re.search(r"\b\d{2}\b", s) is None:
        s_forced = f"{s} {prefer_year}"
    else:
        s_forced = s
    for fmt in ("%d %b %Y", "%d %B %Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d-%m-%Y"):
        dt = pd.to_datetime(s_forced, format=fmt, errors="coerce")
        if pd.notna(dt):
            return pd.to_datetime(dt)
    dt = pd.to_datetime(s_forced, errors="coerce")
    if pd.notna(dt):
        return pd.to_datetime(dt)
    return None

def extract_weekstart_from_header(col: str, prefer_year: Optional[int] = None) -> Optional[pd.Timestamp]:
    s = str(col)
    m = re.search(r"\(([^)]*)\)", s)
    candidate = m.group(1) if m else s
    wk = re.search(r"Week of\s*(.*)$", s, flags=re.I)
    if wk:
        candidate = wk.group(1).strip()
    parts = re.split(r"\s*-\s*", candidate.strip())
    first_part = parts[0] if parts else candidate.strip()
    first_part = re.sub(r"^Week\s*\d+\s*", "", first_part, flags=re.I).strip()
    dt = try_parse_date_string(first_part, prefer_year=prefer_year)
    if dt is None:
        m2 = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", candidate)
        if m2:
            dt = try_parse_date_string(m2.group(1), prefer_year=prefer_year)
        else:
            m3 = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9})\b", candidate)
            if m3:
                dt = try_parse_date_string(m3.group(1), prefer_year=prefer_year)
    if dt is None:
        return None
    return pd.to_datetime(dt).to_period("W-MON").start_time

def future_weeks_after(start_date, periods: int) -> pd.DatetimeIndex:
    start_date = pd.to_datetime(start_date)
    next_mon = (start_date + pd.offsets.Week(weekday=0))
    if next_mon <= start_date:
        next_mon = next_mon + pd.offsets.Week(weekday=0)
    return pd.date_range(start=next_mon, periods=periods, freq="W-MON")

# ------------------------ SALES LOADER ------------------------
def read_sales_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        if get_ext(src) in [".xlsx", ".xls"]:
            df = load_any_table(src, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            week_col = df.columns[0]
            units_col = "Ordered Units" if "Ordered Units" in df.columns else df.columns[2]
            week_start = pd.to_datetime(df[week_col].astype(str).str.split(" - ").str[0], errors="coerce")
            units = pd.to_numeric(df[units_col].astype(str).str.replace(r"[^0-9\-]", "", regex=True), errors="coerce").fillna(0)
        else:
            df = load_any_table(src, sep=";", skiprows=1, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            week_col = df.columns[0]
            units_col = "Ordered Units" if "Ordered Units" in df.columns else df.columns[2]
            week_start = pd.to_datetime(df[week_col].astype(str).str.split(" - ").str[0], errors="coerce")
            units = pd.to_numeric(df[units_col].astype(str).str.replace(r"[^0-9\-]", "", regex=True), errors="coerce").fillna(0)

        out = pd.DataFrame({"Week_Start": week_start.dt.to_period("W-MON").dt.start_time, "y": units})
        out = out.dropna(subset=["Week_Start"]).groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
        meta = {"mode": "long", "format": "excel" if get_ext(src) in [".xlsx", ".xls"] else "semicolon",
                "rows": int(len(out)), "sum_y": float(out["y"].sum())}
        return out, meta
    except Exception:
        pass

    df = load_any_table(src, header=0)
    prefer_year = datetime.now().year
    week_cols, week_map = [], {}
    for c in df.columns:
        wk = extract_weekstart_from_header(c, prefer_year)
        if wk is not None:
            week_cols.append(c); week_map[c] = wk
    if not week_cols:
        st.error("Sales file: No usable 'Week' column or week headers found.")
        st.stop()
    id_cols = [c for c in df.columns if c not in week_cols]
    long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
    long["Week_Start"] = long["col"].map(week_map)
    long.dropna(subset=["Week_Start"], inplace=True)
    long["y"] = pd.to_numeric(long["val"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").fillna(0)
    out = long.groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
    meta = {"mode": "wide", "rows": int(len(out)), "sum_y": float(out["y"].sum())}
    return out, meta

# ------------------------ AMAZON FORECAST LOADER ------------------------
def read_amazon_forecast_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prefer_year = datetime.now().year
    ext = get_ext(src)

    # Pass 1: parse every header; keep those that parse to a week
    for skip in range(0, 6):
        try:
            if ext in [".xlsx", ".xls"]:
                df = load_any_table(src, header=0)
            else:
                df = load_any_table(src, sep=None, skiprows=skip, header=0)

            week_map = {c: extract_weekstart_from_header(c, prefer_year) for c in df.columns}
            week_cols = [c for c, wk in week_map.items() if wk is not None]
            if not week_cols:
                continue

            id_cols = [c for c in df.columns if c not in week_cols]
            long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
            long["Week_Start"] = long["col"].map(week_map)
            long.dropna(subset=["Week_Start"], inplace=True)
            long["nval"] = pd.to_numeric(long["val"].astype(str).str.replace(r"[^0-9\-.]", "", regex=True), errors="coerce")
            df_up = long.groupby("Week_Start", as_index=False)["nval"].sum().rename(columns={"nval": "Amazon_Sellout_Forecast"})
            df_up["Amazon_Sellout_Forecast"] = df_up["Amazon_Sellout_Forecast"].round().astype(int)
            df_up = df_up.sort_values("Week_Start")
            meta = {"mode": "normal", "skip": skip, "week_cols_found": len(week_cols), "rows": int(len(df_up)),
                    "sum": int(df_up["Amazon_Sellout_Forecast"].sum()) if not df_up.empty else 0}
            return df_up, meta
        except Exception:
            continue

    # Pass 2: 2nd-row header fallback
    try:
        if ext in [".xlsx", ".xls"]:
            df0 = load_any_table(src, header=None)
            if df0.shape[0] >= 3:
                header = df0.iloc[1].astype(str).tolist()
                data   = df0.iloc[2:].copy()
                data.columns = header
                df = data
            else:
                df = df0
        else:
            df = load_any_table(src, sep=None, header=None)
            if df.shape[0] >= 3:
                header = df.iloc[1].astype(str).tolist()
                data   = df.iloc[2:].copy()
                data.columns = header
                df = data

        week_map = {c: extract_weekstart_from_header(c, prefer_year) for c in df.columns}
        week_cols = [c for c, wk in week_map.items() if wk is not None]
        if week_cols:
            vals = df[week_cols].applymap(lambda x: re.sub(r"[^0-9\-.]", "", str(x)))
            vals = vals.apply(pd.to_numeric, errors="coerce")
            sums = vals.sum(axis=0, skipna=True)
            recs = []
            for col, val in sums.items():
                dt = week_map[col]
                if dt is not None and pd.notna(val):
                    recs.append({"Week_Start": pd.to_datetime(dt).to_period("W-MON").start_time,
                                 "Amazon_Sellout_Forecast": int(round(val))})
            df_up = pd.DataFrame(recs).sort_values("Week_Start")
            meta = {"mode": "2nd_row_header", "week_cols_found": len(week_cols), "rows": int(len(df_up)),
                    "sum": int(df_up["Amazon_Sellout_Forecast"].sum()) if not df_up.empty else 0}
            return df_up, meta
    except Exception:
        pass

    return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"]), {"mode": "none", "week_cols_found": 0, "rows": 0, "sum": 0}

# ------------------------ INVENTORY LOADER (optional) ------------------------
def read_inventory_onhand(src) -> Optional[int]:
    today = pd.to_datetime(datetime.now().date())
    for skip in (0, 1, 2, 3, 4, 5):
        try:
            df = load_any_table(src, sep=None, skiprows=skip, header=0)
            week_col = None
            for c in df.columns:
                if re.search(r"^\s*week\s*$", str(c), re.I):
                    week_col = c; break
            if week_col is None:
                continue
            onhand_col = None
            for c in df.columns:
                if re.search(r"on\s*hand|onhand|o/h|inventory", str(c), re.I):
                    onhand_col = c; break
            if onhand_col is None:
                continue
            wk = pd.to_datetime(df[week_col].astype(str).str.split(" - ").str[0], errors="coerce")
            oh = pd.to_numeric(df[onhand_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
            tmp = pd.DataFrame({"Week_Start": wk.dt.to_period("W-MON").dt.start_time, "OH": oh})
            tmp = tmp.dropna(subset=["Week_Start", "OH"])
            tmp = tmp[tmp["Week_Start"] <= today].sort_values("Week_Start")
            if tmp.empty:
                continue
            return int(tmp["OH"].iloc[-1])
        except Exception:
            continue
    return None

# ------------------------ Load & reshape ------------------------
df_hist, sales_meta = read_sales_to_long(sales_path)
today = pd.to_datetime(datetime.now().date())
df_hist_filtered = df_hist[df_hist["Week_Start"] <= today]
df_hist = df_hist_filtered if not df_hist_filtered.empty else df_hist

df_up = pd.DataFrame()
up_meta = {"week_cols_found": 0}
if up_path:
    df_up, up_meta = read_amazon_forecast_to_long(up_path)

init_inv_override = None
if inv_path:
    init_inv_override = read_inventory_onhand(inv_path)
start_on_hand = int(init_inv_override) if init_inv_override is not None else int(init_inv)

with st.expander("Debug: parsing summary", expanded=False):
    st.json({
        "sales_meta": sales_meta,
        "amazon_forecast_meta": up_meta,
        "start_on_hand_used": start_on_hand,
        "hist_rows": int(len(df_hist)),
        "hist_sum": float(df_hist["y"].sum())
    })

# ------------------------ Forecast (MODEL = FALLBACK ONLY) ------------------------
forecast_label = "Forecast_Units"
last_week = df_hist["Week_Start"].max() if not df_hist.empty else today
future_idx = future_weeks_after(max(last_week, today), periods)

if model_choice == "Prophet" and PROPHET_INSTALLED and not df_hist.empty and df_hist["y"].sum() > 0:
    m = Prophet(weekly_seasonality=True)  # type: ignore[name-defined]
    m.fit(df_hist.rename(columns={"Week_Start": "ds", "y": "y"}))
    fut = pd.DataFrame({"ds": future_idx})
    df_fc = m.predict(fut)[["ds", "yhat"]].rename(columns={"ds": "Week_Start"})
elif model_choice == "ARIMA" and ARIMA_INSTALLED:
    tmp = df_hist.set_index("Week_Start").asfreq("W-MON", fill_value=0)
    series = tmp["y"] if not tmp.empty else pd.Series([0.0], index=pd.date_range(today, periods=1, freq="W-MON"))
    try:
        ar = ARIMA(series, order=(1, 1, 1)).fit()  # type: ignore[name-defined]
        pr = ar.get_forecast(steps=periods)
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": pr.predicted_mean.values})
    except Exception:
        last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})
else:
    last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})

df_fc[forecast_label] = pd.Series(df_fc.get("yhat", 0)).clip(lower=0).round().astype(int)
df_fc["Week_Start"] = pd.to_datetime(df_fc["Week_Start"]).dt.to_period("W-MON").dt.start_time

# ------------------------ AMAZON = AUTHORITATIVE FORECAST ------------------------
amazon_drives = False
if not df_up.empty:
    amazon = df_up.copy()
    amazon["Week_Start"] = pd.to_datetime(amazon["Week_Start"]).dt.to_period("W-MON")
    start_from = future_weeks_after(max(last_week, today), 1)[0].to_period("W-MON")
    amazon = amazon[amazon["Week_Start"] >= start_from].sort_values("Week_Start")
    if not amazon.empty:
        amazon = amazon.head(periods)
        df_fc = amazon.rename(columns={"Amazon_Sellout_Forecast": forecast_label})[["Week_Start", forecast_label]]
        df_fc["Week_Start"] = df_fc["Week_Start"].dt.start_time
        amazon_drives = True

# ------------------------ Projected $ ------------------------
df_fc["Projected_Sales"] = (df_fc[forecast_label] * float(unit_price)).round(2)

# ------------------------ Predict Weekly POs (ROLLING WOH LOGIC) ------------------------
# PO_t = max(0, sum_{k=1..WOH} F_{t+k} - (OnHand_t_after_arrivals - F_t))
n = len(df_fc)
on_hand_begin = []
po_units = []
pipeline_receipts = [0] * (n + lead_time_weeks + 5)
on_hand = int(start_on_hand)

F = df_fc[forecast_label].astype(int).to_numpy()

for t in range(n):
    # 1) Receive POs that arrive this week (placed lead_time ago)
    arriving = pipeline_receipts[t] if t < len(pipeline_receipts) else 0
    on_hand += int(arriving)

    # Save "beginning of week after arrivals" for display
    on_hand_begin.append(int(on_hand))

    # 2) Deduct this week's forecast sellout
    demand_t = int(F[t])
    on_hand_after_sellout = max(on_hand - demand_t, 0)

    # 3) Target = sum of the next WOH weeks’ forecast
    start_idx = t + 1
    end_idx = min(t + woc_target, n - 1)
    target = int(F[start_idx:end_idx + 1].sum()) if start_idx <= end_idx else 0

    # 4) PO quantity
    order_qty = max(target - on_hand_after_sellout, 0)

    # 5) Schedule arrival after lead_time weeks
    if lead_time_weeks > 0 and (t + lead_time_weeks) < len(pipeline_receipts):
        pipeline_receipts[t + lead_time_weeks] += int(order_qty)
    elif lead_time_weeks == 0:
        on_hand_after_sellout += int(order_qty)  # arrives immediately

    po_units.append(int(order_qty))

    # Update on-hand to end-of-week position
    on_hand = on_hand_after_sellout

df_fc["On_Hand_Begin"] = on_hand_begin
df_fc["Predicted_PO_Units"] = po_units
df_fc["Predicted_SellIn_$"] = (df_fc["Predicted_PO_Units"] * float(unit_price)).round(2)

df_fc["Weeks_Of_Cover"] = np.where(
    df_fc[forecast_label] > 0,
    (df_fc["On_Hand_Begin"] / df_fc[forecast_label]).round(2),
    np.nan
)
df_fc["Date"] = pd.to_datetime(df_fc["Week_Start"]).dt.strftime("%d-%m-%Y")

# ------------------------ Plot ------------------------
st.subheader(f"{len(df_fc)}-Week Forecast & Predicted POs")
if projection_type == "Sales $":
    primary_key, primary_title = "Projected_Sales", "Sales $"
    secondary_key, secondary_title = forecast_label, "Units"
else:
    primary_key, primary_title = forecast_label, "Units"
    secondary_key, secondary_title = "Projected_Sales", "Sales $"

if PLOTLY_INSTALLED:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[primary_key],
                             name=f"{primary_title} ({'Projected' if primary_key=='Projected_Sales' else 'Sell-out Forecast'})",
                             yaxis="y", line=dict(color=AMZ_ORANGE)))
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc["Predicted_PO_Units"],
                             name="Predicted PO Units (Sell-in)",
                             yaxis="y" if primary_key == forecast_label else "y2",
                             line=dict(color=AMZ_BLUE)))
    if secondary_key != primary_key:
        fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[secondary_key],
                                 name=f"{secondary_title} ({'Projected' if secondary_key=='Projected_Sales' else 'Sell-out Forecast'})",
                                 yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(
        xaxis=dict(title="Week"),
        yaxis=dict(title=primary_title),
        yaxis2=dict(title=secondary_title, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(tickformat="%d-%m-%Y")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(df_fc.set_index("Week_Start")[[forecast_label, "Predicted_PO_Units"]])
    st.line_chart(df_fc.set_index("Week_Start")["Projected_Sales"])

# ------------------------ Summary + Detail ------------------------
st.subheader("Summary Metrics")
total_po_units = int(df_fc["Predicted_PO_Units"].sum())
total_sellin = float(df_fc["Predicted_SellIn_$"].sum())
avg_sellin = float(df_fc["Predicted_SellIn_$"].mean())
recap = pd.DataFrame({
    "Metric": ["Total Predicted PO Units", "Total Predicted Sell-In $", "Avg Weekly Sell-In $"],
    "Value": [f"{total_po_units:,}", f"${total_sellin:,.2f}", f"${avg_sellin:,.2f}"]
})
st.table(recap)

st.subheader("Detailed Plan")
st.dataframe(
    df_fc[["Date", forecast_label, "Projected_Sales", "On_Hand_Begin", "Predicted_PO_Units", "Predicted_SellIn_$", "Weeks_Of_Cover"]],
    use_container_width=True
)

st.markdown(
    f"<div style='text-align:center;color:gray;margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True,
)
