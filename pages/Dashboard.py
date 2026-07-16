
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_dataset, money


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


df = get_data()

st.title("Executive Dashboard")
st.caption("High-level profitability oversight across SkyCity Auckland venues")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Revenue", money(df["revenue"].sum()))
kpi2.metric("Net Profit", money(df["net_profit"].sum()))
kpi3.metric("Average Margin", f"{(df['net_profit'].sum() / df['revenue'].sum()):.1%}")
kpi4.metric("Orders", f"{df['orders'].sum():,.0f}")

left, right = st.columns(2)
with left:
    channel_mix = pd.DataFrame(
        {
            "Channel": ["Uber Eats", "DoorDash", "Self-delivery"],
            "Share": [df["ue_share"].mean(), df["dd_share"].mean(), df["sd_share"].mean()],
        }
    )
    st.plotly_chart(
        px.pie(
            channel_mix,
            names="Channel",
            values="Share",
            hole=0.55,
            title="Average channel mix",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#22c55e"],
        ),
        use_container_width=True,
    )
with right:
    monthly = (
        df.assign(month=pd.to_datetime(df["month"]).dt.to_period("M"))
        .groupby("month", as_index=False)[["revenue", "net_profit"]]
        .sum()
    )
    monthly["month"] = monthly["month"].astype(str)
    st.plotly_chart(
        px.line(monthly, x="month", y=["revenue", "net_profit"], markers=True, title="Revenue and profit trend"),
        use_container_width=True,
    )

st.subheader("Venue performance")
venue_summary = (
    df.groupby("venue", as_index=False)
    .agg(Revenue=("revenue", "sum"), Net_Profit=("net_profit", "sum"), Orders=("orders", "sum"))
)
venue_summary["Margin"] = venue_summary["Net_Profit"] / venue_summary["Revenue"]
st.dataframe(
    venue_summary.style.format({"Revenue": "${:,.0f}", "Net_Profit": "${:,.0f}", "Margin": "{:.1%}", "Orders": "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)
