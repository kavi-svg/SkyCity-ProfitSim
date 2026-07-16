import numpy as np
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
model, _, _ = get_model_bundle()

st.title("What-if Analysis")
st.caption("Test the impact of portfolio and cost assumptions in real time")

defaults = df.median(numeric_only=True)
revenue = st.number_input("Monthly revenue ($)", 10_000, 300_000, int(defaults["revenue"]), 5_000)
orders = st.number_input("Monthly orders", 100, 15_000, int(defaults["orders"]), 100)
commission = st.slider("Aggregator commission rate", 0.10, 0.40, float(defaults["commission_rate"]), 0.01)
delivery = st.slider("Self-delivery cost per order ($)", 1.0, 12.0, float(defaults["delivery_cost_order"]), 0.25)
food = st.slider("Food cost ratio", 0.15, 0.50, float(defaults["food_cost_ratio"]), 0.01)
labour = st.slider("Labour cost ratio", 0.10, 0.40, float(defaults["labour_cost_ratio"]), 0.01)
growth = st.slider("Demand growth", -0.20, 0.35, float(defaults["demand_growth"]), 0.01)

ue = st.slider("Uber Eats share", 0, 90, 30, 1) / 100
dd = st.slider("DoorDash share", 0, 90, 20, 1) / 100
sd = 1 - ue - dd
if sd < 0:
    st.error("Uber Eats and DoorDash shares cannot exceed 100% together.")
    st.stop()

x = scenario_frame(revenue, orders, commission, delivery, ue, dd, sd, food, labour, growth)
predicted = float(model.predict(x)[0])

st.metric("Projected net profit", money(predicted))

rates = np.linspace(max(0.10, commission - 0.10), min(0.40, commission + 0.10), 15)
profits = [float(model.predict(scenario_frame(revenue, orders, rate, delivery, ue, dd, sd, food, labour, growth))[0]) for rate in rates]
fig = px.line(x=rates, y=profits, labels={"x": "Commission rate", "y": "Predicted profit ($)"}, title="Commission sensitivity")
fig.add_vline(x=commission, line_dash="dash", line_color="#fbbf24")
st.plotly_chart(fig, use_container_width=True)
