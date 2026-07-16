import pandas as pd
import plotly.express as px
import streamlit as st 

from utils import load_dataset, money

st.set_page_config(page_title="SkyCity ProfitSim", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
        .stApp {background: #07111f; color: #e5e7eb;}
        .stSidebar {background: #0f172a;}
        [data-testid='stMetric'] {background: #12213a; padding: 16px; border-radius: 12px; border: 1px solid #263a56;}
        h1, h2, h3 {color: #f8fafc;}
        .block-container {padding-top: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


df = get_data()

st.title("SkyCity ProfitSim")
st.caption("Decision-support analytics for restaurant and bar profitability in Auckland")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Total Revenue", money(df["revenue"].sum()))
summary_col2.metric("Total Net Profit", money(df["net_profit"].sum()))
summary_col3.metric("Average Margin", f"{(df['net_profit'].sum() / df['revenue'].sum()):.1%}")
summary_col4.metric("Average Orders", f"{df['orders'].mean():,.0f}")

st.subheader("Business objective")
st.write(
    "This workspace turns a pricing and channel-mix scenario into a practical forecast that helps leadership understand profitability, test commission assumptions, and identify the best delivery mix."
)

st.subheader("What you can explore")
explore_col1, explore_col2, explore_col3 = st.columns(3)
with explore_col1:
    st.info("Dashboard — executive KPIs, venue performance, and monthly trends")
with explore_col2:
    st.info("Prediction — estimate net profit for a selected operating scenario")
with explore_col3:
    st.info("Optimization — find the channel mix that maximizes expected profit")

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

st.subheader("Suggested next step")
st.write("Open the dashboard, analytics, prediction, what-if, optimization, or about pages from the sidebar to explore the full workflow.")
