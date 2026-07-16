import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_dataset, money, scenario_frame, train_models


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return train_models(get_data())


df = get_data()
model, metrics, _ = get_model_bundle()

st.title("Profit Prediction")
st.caption("Estimate net profit using the selected operating assumptions")

defaults = df.median(numeric_only=True)
col1, col2, col3 = st.columns(3)
revenue = col1.number_input("Monthly revenue ($)", 10_000, 300_000, int(defaults["revenue"]), 5_000)
orders = col2.number_input("Monthly orders", 100, 15_000, int(defaults["orders"]), 100)
commission = col3.slider("Aggregator commission rate", 0.10, 0.40, float(defaults["commission_rate"]), 0.01)

delivery = col1.slider("Self-delivery cost per order ($)", 1.0, 12.0, float(defaults["delivery_cost_order"]), 0.25)
food = col2.slider("Food cost ratio", 0.15, 0.50, float(defaults["food_cost_ratio"]), 0.01)
labour = col3.slider("Labour cost ratio", 0.10, 0.40, float(defaults["labour_cost_ratio"]), 0.01)

growth = col1.slider("Demand growth", -0.20, 0.35, float(defaults["demand_growth"]), 0.01)

st.subheader("Channel mix")
ue = st.slider("Uber Eats share", 0, 90, 30, 1) / 100
dd = st.slider("DoorDash share", 0, 90, 20, 1) / 100
sd = 1 - ue - dd
if sd < 0:
    st.error("Uber Eats and DoorDash shares cannot exceed 100% together.")
    st.stop()

st.caption(f"Self-delivery share: **{sd:.0%}**")

x = scenario_frame(revenue, orders, commission, delivery, ue, dd, sd, food, labour, growth)
predicted = float(model.predict(x)[0])
confidence = max(0.0, min(0.99, 0.60 + float(metrics.loc[metrics["Model"] == metrics.iloc[0]["Model"], "R²"].iloc[0]) * 0.25))

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Predicted monthly net profit", money(predicted))
metric_col2.metric("Predicted net margin", f"{predicted / revenue:.1%}")
metric_col3.metric("Confidence score", f"{confidence:.0%}")

st.subheader("Sensitivity preview")
rates = np.linspace(max(0.10, commission - 0.10), min(0.40, commission + 0.10), 12)
profits = [float(model.predict(scenario_frame(revenue, orders, rate, delivery, ue, dd, sd, food, labour, growth))[0]) for rate in rates]
chart = px.line(x=rates, y=profits, labels={"x": "Commission rate", "y": "Predicted profit ($)"}, title="Commission sensitivity")
chart.add_vline(x=commission, line_dash="dash", line_color="#fbbf24")
st.plotly_chart(chart, use_container_width=True)
