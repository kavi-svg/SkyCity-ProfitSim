import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_dataset, money, optimise_mix, scenario_frame, train_models


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return train_models(get_data())


df = get_data()
model, _, _ = get_model_bundle()

st.title("Optimization")
st.caption("Find a higher-profit channel mix and quantify the uplift")

defaults = df.median(numeric_only=True)
revenue = st.number_input("Monthly revenue ($)", 10_000, 300_000, int(defaults["revenue"]), 5_000)
orders = st.number_input("Monthly orders", 100, 15_000, int(defaults["orders"]), 100)
commission = st.slider("Aggregator commission rate", 0.10, 0.40, float(defaults["commission_rate"]), 0.01)
delivery = st.slider("Self-delivery cost per order ($)", 1.0, 12.0, float(defaults["delivery_cost_order"]), 0.25)
food = st.slider("Food cost ratio", 0.15, 0.50, float(defaults["food_cost_ratio"]), 0.01)
labour = st.slider("Labour cost ratio", 0.10, 0.40, float(defaults["labour_cost_ratio"]), 0.01)
growth = st.slider("Demand growth", -0.20, 0.35, float(defaults["demand_growth"]), 0.01)

current_mix = st.slider("Current Uber Eats share", 0, 90, 30, 1) / 100
dd_current = st.slider("Current DoorDash share", 0, 90, 20, 1) / 100
sd_current = 1 - current_mix - dd_current
if sd_current < 0:
    st.error("Uber Eats and DoorDash shares cannot exceed 100% together.")
    st.stop()

x_current = scenario_frame(revenue, orders, commission, delivery, current_mix, dd_current, sd_current, food, labour, growth)
current_profit = float(model.predict(x_current)[0])

best_ue, best_dd, best_sd, best_profit = optimise_mix(model, revenue, orders, commission, delivery, food, labour, growth)

uplift = (best_profit - current_profit) / abs(current_profit) if current_profit else 0

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Current scenario profit", money(current_profit))
metric2.metric("Optimised profit", money(best_profit), f"{uplift:.1%} uplift")
metric3.metric("Recommended self-delivery", f"{best_sd:.0%}")

st.success(f"Recommended mix: Uber Eats {best_ue:.0%}, DoorDash {best_dd:.0%}, self-delivery {best_sd:.0%}.")
st.caption("Use this as a decision-support signal, then validate it against capacity, service levels, and contract terms.")

candidates = []
for ue in np.arange(0.05, 0.76, 0.05):
    for dd in np.arange(0.05, 0.76 - ue, 0.05):
        sd = 1 - ue - dd
        x = scenario_frame(revenue, orders, commission, delivery, ue, dd, sd, food, labour, growth)
        candidates.append((ue, dd, sd, float(model.predict(x)[0])))

plot_df = pd.DataFrame(candidates, columns=["Uber Eats", "DoorDash", "Self-delivery", "Profit"])
plot_df = plot_df.assign(Score=plot_df["Profit"])
st.plotly_chart(px.scatter(plot_df, x="Uber Eats", y="DoorDash", size="Score", color="Score", hover_name="Profit", title="Profit surface by channel mix"), use_container_width=True)
