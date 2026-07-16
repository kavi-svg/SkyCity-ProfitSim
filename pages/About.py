import streamlit as st

st.title("Research & Methodology")
st.caption("How the forecasting workflow was designed")

st.subheader("Dataset")
st.write("The app uses a realistic synthetic dataset that mirrors common monthly operating characteristics across Auckland restaurants and bars: revenue, order volume, commission exposure, food and labour cost ratios, and delivery channel shares.")

st.subheader("Methodology")
st.write("The workflow creates engineered features such as commission exposure, self-delivery cost burden, profit per order, and cost-to-revenue ratios before training a set of regression models and comparing RMSE, MAE, and R².")

st.subheader("Model performance")
st.write("Model selection is based on the lowest test RMSE. The best model is then used to generate profit scenarios and optimization recommendations.")

st.subheader("Business insights")
st.write("The tool helps teams understand when higher self-delivery penetration improves margin, when commission pressure hurts profitability, and how different channel mixes shift expected outcomes.")
