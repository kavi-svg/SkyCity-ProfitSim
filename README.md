# SkyCity ProfitSim

A Streamlit decision-support dashboard for modelling restaurant and bar profitability across Uber Eats, DoorDash, and self-delivery channels.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL Streamlit prints (normally `http://localhost:8501`). The first version uses deterministic synthetic data so it works immediately; use it to explore the dashboard while preparing source data.

## Included modules

- Executive KPI dashboard and venue/channel analysis
- Net-profit prediction
- Commission sensitivity what-if analysis
- Channel-mix optimisation recommendation
- Linear Regression, Random Forest, and Gradient Boosting comparison
