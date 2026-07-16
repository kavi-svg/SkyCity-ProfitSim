import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_dataset


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


df = get_data()
df["margin"] = df["net_profit"] / df["revenue"]

st.title("Data Analytics")
st.caption("Operational sizing, channel exposure, and profitability patterns")

trend = (
    df.assign(month=pd.to_datetime(df["month"]).dt.to_period("M"))
    .groupby("month", as_index=False)[["revenue", "net_profit"]]
    .sum()
)
trend["month"] = trend["month"].astype(str)

left, right = st.columns(2)
with left:
    st.plotly_chart(px.line(trend, x="month", y="revenue", markers=True, title="Revenue trend"), use_container_width=True)
with right:
    st.plotly_chart(px.line(trend, x="month", y="net_profit", markers=True, title="Profit trend"), use_container_width=True)

cuisine = df.groupby("cuisine", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("net_profit", "sum"))
segment = df.groupby("segment", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("net_profit", "sum"))
region = df.groupby("region", as_index=False).agg(Revenue=("revenue", "sum"), Profit=("net_profit", "sum"))

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(px.bar(cuisine.sort_values("Revenue", ascending=False), x="cuisine", y="Revenue", title="Revenue by cuisine"), use_container_width=True)
with col2:
    st.plotly_chart(px.bar(segment.sort_values("Revenue", ascending=False), x="segment", y="Revenue", title="Revenue by segment"), use_container_width=True)
with col3:
    st.plotly_chart(px.bar(region.sort_values("Revenue", ascending=False), x="region", y="Revenue", title="Revenue by region"), use_container_width=True)

st.subheader("Correlation heatmap")
correlation_columns = ["revenue", "orders", "commission_rate", "delivery_cost_order", "ue_share", "dd_share", "sd_share", "food_cost_ratio", "labour_cost_ratio", "demand_growth", "net_profit"]
correlation = df[correlation_columns].corr()
st.plotly_chart(px.imshow(correlation, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature correlation"), use_container_width=True)

st.subheader("Distribution view")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    st.plotly_chart(px.histogram(df, x="revenue", nbins=25, title="Revenue distribution"), use_container_width=True)
with plot_col2:
    st.plotly_chart(px.histogram(df, x="margin", nbins=20, title="Margin distribution"), use_container_width=True)
