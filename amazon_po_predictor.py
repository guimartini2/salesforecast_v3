# streamlit_app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Amazon PO Replenishment Predictor", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def detect_delimiter(header_line: str) -> str:
    return ";" if header_line.count(";") > header_line.count(",") else ","

def parse_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    lines = [ln for ln in re.split(r"\r?\n", text) if ln.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    delim = detect_delimiter(lines[0])
    # Heuristic: second line is the header row with Week columns per your JS
    headers = [h.strip() for h in lines[1].split(delim)]
    # mandatory first 3 columns
    if len(headers) < 4:
        return pd.DataFrame()

    # identify weekly columns
    week_cols = [h for h in headers[3:] if re.match(r"Week\s*\d+$", h, re.I)]
    if not week_cols:
        return pd.DataFrame()

    # Build dataframe from line 2 onward as data
    data_rows = []
    for row in lines[2:]:
        if not row.strip():
            continue
        cols = [c.strip() for c in row.split(delim)]
        if len(cols) < 4:
            continue
        asin, title, brand = cols[0:3]
        # numeric clean (remove thousands commas)
        vals = []
        for c in cols[3:3+len(week_cols)]:
            c = (c or "").replace(",", "")
            try:
                vals.append(float(c))
            except ValueError:
                vals.append(0.0)
        entry = {"ASIN": asin, "Product Title": title, "Brand": brand}
        for w_name, v in zip(week_cols, vals):
            entry[w_name] = v
        data_rows.append(entry)

    df = pd.DataFrame(data_rows)
    # Ensure all week columns exist even if some rows were short
    for w in week_cols:
        if w not in df.columns:
            df[w] = 0.0
    # Keep column order: id cols + weeks
    return df[["ASIN", "Product Title", "Brand"] + week_cols]

def moving_average(values, horizon, periods=4):
    vals = list(values)
    out = []
    for _ in range(horizon):
        span = min(periods, len(vals)) or 1
        avg = float(np.mean(vals[-span:])) if vals else 0.0
        out.append(avg)
        vals.append(avg)
    return out

def exponential_smoothing(values, horizon, alpha=0.3):
    if not len(values):
        return [0.0]*horizon
    level = values[0]
    for t in range(1, len(values)):
        level = alpha*values[t] + (1-alpha)*level
    return [level]*horizon

def linear_trend(values, horizon):
    n = len(values)
    if n == 0:
        return [0.0]*horizon
    x = np.arange(1, n+1, dtype=float)
    y = np.array(values, dtype=float)
    # least squares
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    forecasts = []
    for i in range(1, horizon+1):
        f = intercept + slope*(n + i)
        forecasts.append(max(0.0, f))
    return forecasts

def seasonal_naive(values, horizon, season=12):
    if not len(values):
        return [0.0]*horizon
    s = min(season, len(values))
    hist = values[-s:]
    out = []
    for i in range(horizon):
        out.append(hist[i % s])
    return out

def compute_po_schedule(
    hist_values,                      # list of floats (weekly demand history) – not used directly here
    forecasts,                        # list of floats (weekly forecast)
    current_inventory: float,
    target_woh: float,
    lead_time_weeks: int,
    safety_stock_weeks: float,
    min_order_qty: int,
    max_order_qty: int,
    price: float,
    avg_window: int = 4
):
    """
    Lead-time aware: orders placed today arrive after `lead_time_weeks`.
    We model a pipeline list of length lead_time_weeks; each week we receive pipeline.pop(0).
    """
    weeks = len(forecasts)
    pipeline = [0]*max(0, lead_time_weeks)  # arrivals by week
    on_hand = float(current_inventory)

    rows = []
    for w in range(weeks):
        # Receive arrivals (if any)
        if lead_time_weeks > 0:
            arriving = pipeline.pop(0) if pipeline else 0
            on_hand += arriving
        else:
            arriving = 0

        # This week's demand (forecast)
        demand = float(forecasts[w])

        # Average weekly demand over a short window including this week’s forecast
        start = max(0, w - (avg_window - 1))
        window_vals = forecasts[start:w+1]
        avg_demand = float(np.mean(window_vals)) if len(window_vals) else (demand or 1.0)

        target_inventory = target_woh * avg_demand
        safety_stock = safety_stock_weeks * avg_demand
        reorder_point = lead_time_weeks * avg_demand + safety_stock

        # Inventory position BEFORE placing a new order but after arrivals, before demand
        inv_position_before_demand = on_hand

        # Consume demand
        on_hand_after_demand = on_hand - demand

        # Check reorder after consuming demand (conservative)
        order_trigger = on_hand_after_demand <= reorder_point

        po_qty = 0
        if order_trigger:
            raw_needed = target_inventory - on_hand_after_demand + lead_time_weeks*avg_demand
            po_qty = int(np.clip(np.ceil(raw_needed), min_order_qty, max_order_qty))
            if lead_time_weeks > 0:
                # schedule arrival
                if len(pipeline) < lead_time_weeks:
                    # pad if needed (shouldn't happen, but safe)
                    pipeline += [0]*(lead_time_weeks - len(pipeline))
                pipeline[-1] += po_qty  # lands after lead time
            else:
                # arrives immediately
                on_hand_after_demand += po_qty

        on_hand = max(0.0, on_hand_after_demand)
        woh = (on_hand / avg_demand) if avg_demand > 1e-9 else 0.0

        rows.append({
            "Week": w+1,
            "Demand": round(demand),
            "Arrivals": arriving,
            "Inventory": int(round(on_hand)),
            "WOH": round(woh, 1),
            "TargetInventory": int(round(target_inventory)),
            "ReorderPoint": int(round(reorder_point)),
            "POQty": int(po_qty),
            "POValue": int(round(po_qty * price)),
            "Status": "Order Required" if order_trigger else ("Low Stock" if on_hand <= reorder_point else "Normal"),
        })

    return pd.DataFrame(rows)

# -----------------------------
# UI
# -----------------------------
st.title("Amazon PO Replenishment Predictor (Python)")

st.markdown(
    "Upload a CSV with columns: **ASIN**, **Product Title**, **Brand**, then **Week 1, Week 2, ...**"
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV upload…")
    st.stop()

df = parse_csv(uploaded.read())
if df.empty:
    st.error("Could not parse the CSV. Make sure it includes ASIN / Product Title / Brand and Week N columns.")
    st.stop()

st.success(f"Loaded {df['ASIN'].nunique()} ASIN(s)")

asin = st.selectbox("Select ASIN", sorted(df["ASIN"].unique()))
row = df[df["ASIN"] == asin].iloc[0]

week_cols = [c for c in df.columns if re.match(r"Week\s*\d+$", c, re.I)]
history = row[week_cols].astype(float).tolist()
# Drop leading/trailing zeros? Keep as-is; forecasts can handle zeros.

# -----------------------------
# Controls
# -----------------------------
colA, colB, colC = st.columns(3)

with colA:
    current_inventory = st.number_input("Current Inventory", min_value=0, value=1000, step=50)
    asin_price = st.number_input("ASIN Price ($)", min_value=0.0, value=25.99, step=0.5, format="%.2f")
    target_woh = st.number_input("Target WOH (weeks)", min_value=0.0, value=4.0, step=0.5)

with colB:
    lead_time_weeks = st.number_input("Lead Time (weeks)", min_value=0, value=3, step=1)
    safety_stock_weeks = st.number_input("Safety Stock (weeks)", min_value=0.0, value=1.0, step=0.5)
    forecast_horizon = st.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=26, step=1)

with colC:
    min_order_qty = st.number_input("Min Order Qty", min_value=0, value=500, step=50)
    max_order_qty = st.number_input("Max Order Qty", min_value=0, value=10000, step=100)
    model = st.selectbox("Forecasting Method", ["moving_average", "exponential_smoothing", "linear_trend", "seasonal_naive"])

# Model parameters
params = {}
if model == "moving_average":
    params["ma_periods"] = st.slider("Moving Average Periods", 1, 12, 4)
elif model == "exponential_smoothing":
    params["alpha"] = st.slider("Alpha (0.1–0.9)", 0.1, 0.9, 0.3, step=0.1)
elif model == "seasonal_naive":
    params["season"] = st.slider("Seasonality Period", 4, 52, 12, step=1)

# -----------------------------
# Forecast
# -----------------------------
if model == "moving_average":
    forecasts = moving_average(history, forecast_horizon, periods=params["ma_periods"])
elif model == "exponential_smoothing":
    forecasts = exponential_smoothing(history, forecast_horizon, alpha=params["alpha"])
elif model == "linear_trend":
    forecasts = linear_trend(history, forecast_horizon)
elif model == "seasonal_naive":
    forecasts = seasonal_naive(history, forecast_horizon, season=params["season"])
else:
    forecasts = [float(np.mean(history)) if history else 0.0]*forecast_horizon

# -----------------------------
# PO Schedule
# -----------------------------
po_df = compute_po_schedule(
    hist_values=history,
    forecasts=forecasts,
    current_inventory=current_inventory,
    target_woh=target_woh,
    lead_time_weeks=lead_time_weeks,
    safety_stock_weeks=safety_stock_weeks,
    min_order_qty=min_order_qty,
    max_order_qty=max_order_qty,
    price=asin_price,
    avg_window=4,
)

# KPIs
total_po_value = int(po_df["POValue"].sum())
total_po_qty = int(po_df["POQty"].sum())
avg_woh = float(np.mean(po_df["WOH"])) if len(po_df) else 0.0
stockout_risk = int((po_df["Inventory"] <= 0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total PO Value ($)", f"{total_po_value:,}")
k2.metric("Total PO Quantity", f"{total_po_qty:,}")
k3.metric("Avg WOH (weeks)", f"{avg_woh:.1f}")
k4.metric("Stockout Risk (weeks <=0)", f"{stockout_risk}")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Demand Forecast & Inventory Levels")
chart_df = po_df[["Week", "Demand", "Inventory", "ReorderPoint"]].melt("Week", var_name="Series", value_name="Value")
st.line_chart(chart_df, x="Week", y="Value", color="Series", height=320)

st.subheader("Purchase Order Quantities")
bars = po_df[po_df["POQty"] > 0][["Week", "POQty"]].set_index("Week")
if not bars.empty:
    st.bar_chart(bars, height=320)
else:
    st.info("No POs triggered with the current parameters.")

# -----------------------------
# PO Table & Export
# -----------------------------
st.subheader("Purchase Order Schedule (first 12 weeks shown)")
st.dataframe(po_df.head(12), use_container_width=True)

csv_buf = io.StringIO()
po_df.to_csv(csv_buf, index=False)
st.download_button(
    "Export Full PO Schedule (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"po_schedule_{asin}.csv",
    mime="text/csv",
)
