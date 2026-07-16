"""Data generation, modelling and scenario utilities for SkyCity ProfitSim."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "revenue", "orders", "commission_rate", "delivery_cost_order", "ue_share",
    "dd_share", "sd_share", "food_cost_ratio", "labour_cost_ratio", "demand_growth",
    "commission_ue", "delivery_sd", "profit_per_order", "cost_to_revenue",
]


def money(value: float) -> str:
    return f"${value:,.0f}"


def generate_demo_data(rows: int = 360, seed: int = 42) -> pd.DataFrame:
    """Create realistic monthly venue data so the app works without private data."""
    rng = np.random.default_rng(seed)
    venue = rng.choice(["Harbour Bar", "Sky Lounge", "The Grill", "Orbit Café", "Elliott Stables"], rows)
    cuisine = rng.choice(["Modern NZ", "Asian", "Bar", "European", "Café"], rows)
    region = rng.choice(["City Centre", "North Shore", "Central", "East Auckland"], rows)
    segment = rng.choice(["Casual Dining", "Cocktail Bar", "Quick Service", "Fine Dining"], rows)
    revenue = rng.uniform(45_000, 220_000, rows)
    orders = (revenue / rng.uniform(28, 52, rows)).round().astype(int)
    ue = rng.uniform(.10, .45, rows)
    dd = rng.uniform(.05, .30, rows)
    sd = np.maximum(.05, 1 - ue - dd)
    total = ue + dd + sd
    ue, dd, sd = ue / total, dd / total, sd / total
    commission = rng.uniform(.18, .34, rows)
    delivery = rng.uniform(2.0, 7.5, rows)
    food = rng.uniform(.24, .38, rows)
    labour = rng.uniform(.16, .29, rows)
    growth = rng.uniform(-.08, .18, rows)
    month_values = pd.date_range("2024-01-01", periods=rows, freq="MS")
    month = [month_values[int(rng.integers(0, len(month_values)))] for _ in range(rows)]
    demand = 1 + growth
    profit = (
        revenue * (1 - food - labour - commission * (ue + dd) - 0.04) * demand
        - orders * delivery * sd
        + rng.normal(0, 3_500, rows)
    )
    profit = np.clip(profit, 6_000, None)
    df = pd.DataFrame({
        "month": month,
        "venue": venue,
        "cuisine": cuisine,
        "region": region,
        "segment": segment,
        "revenue": revenue,
        "orders": orders,
        "commission_rate": commission,
        "delivery_cost_order": delivery,
        "ue_share": ue,
        "dd_share": dd,
        "sd_share": sd,
        "food_cost_ratio": food,
        "labour_cost_ratio": labour,
        "demand_growth": growth,
        "net_profit": profit,
    })
    return engineer_features(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["commission_ue"] = out["commission_rate"] * out["ue_share"]
    out["delivery_sd"] = out["delivery_cost_order"] * out["sd_share"]
    out["profit_per_order"] = out["net_profit"] / out["orders"] if "net_profit" in out else 0
    out["cost_to_revenue"] = out["food_cost_ratio"] + out["labour_cost_ratio"] + out["commission_rate"] * (out["ue_share"] + out["dd_share"])
    return out


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "SkyCity Auckland Restaurants & Bars.csv"
    data_path = Path(path)
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"])
        return engineer_features(df)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demo_data()
    df.to_csv(data_path, index=False)
    return df


def train_models(df: pd.DataFrame):
    x, y = df[FEATURES], df["net_profit"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.22, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    scores, fitted = [], {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scores.append({
            "Model": name,
            "RMSE": mean_squared_error(y_test, pred) ** .5,
            "MAE": mean_absolute_error(y_test, pred),
            "R²": r2_score(y_test, pred),
        })
        fitted[name] = model
    result = pd.DataFrame(scores).sort_values("RMSE").reset_index(drop=True)
    return fitted[result.loc[0, "Model"]], result, fitted


def scenario_frame(revenue, orders, commission, delivery_cost, ue, dd, sd, food, labour, growth):
    row = pd.DataFrame([{
        "revenue": revenue, "orders": orders, "commission_rate": commission, "delivery_cost_order": delivery_cost,
        "ue_share": ue, "dd_share": dd, "sd_share": sd, "food_cost_ratio": food,
        "labour_cost_ratio": labour, "demand_growth": growth, "net_profit": 0,
    }])
    return engineer_features(row)[FEATURES]


def optimise_mix(model, revenue, orders, commission, delivery_cost, food, labour, growth):
    candidates = []
    for ue in np.arange(.05, .76, .05):
        for dd in np.arange(.05, .76 - ue, .05):
            sd = 1 - ue - dd
            x = scenario_frame(revenue, orders, commission, delivery_cost, ue, dd, sd, food, labour, growth)
            candidates.append((ue, dd, sd, float(model.predict(x)[0])))
    return max(candidates, key=lambda item: item[3])
