"""Data loading functions for the Streamlit dashboard.
All loaders are cached so the files are read only once per session.
"""
import os
import pickle

import pandas as pd
import streamlit as st

# Project root → output/
_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


@st.cache_data
def load_dashboard() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_OUTPUT, "dashboard_dataset.csv"))


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load model metrics — Ensemble row dropped if present."""
    df = pd.read_csv(os.path.join(_OUTPUT, "model_metrics.csv"), index_col=0)
    # Drop Ensemble row if it exists
    drop_idx = [i for i in df.index if "ensemble" in str(i).lower()]
    return df.drop(index=drop_idx, errors="ignore")


@st.cache_data
def load_predictions() -> dict:
    """Return dict: model_name → DataFrame(country_code, pred_year, actual, predicted).
    Only XGBoost, ARIMA, LSTM — no Ensemble.
    """
    files = {
        "XGBoost": "xgb_test_predictions.csv",
        "ARIMA":   "arima_test_predictions.csv",
        "LSTM":    "lstm_test_predictions.csv",
    }
    out = {}
    for name, fname in files.items():
        fpath = os.path.join(_OUTPUT, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        if "y_pred" in df.columns:       # XGBoost: year = anchor year → pred_year = year+1
            df["pred_year"] = df["year"] + 1
            df["actual"]    = df["gdp_next_year"]
            df["predicted"] = df["y_pred"]
        else:                            # ARIMA / LSTM: year = pred_year
            df["pred_year"] = df["year"]
        out[name] = df[["country_code", "pred_year", "actual", "predicted"]].copy()
    return out


@st.cache_data
def load_classification_metrics() -> pd.DataFrame | None:
    """Load Phase 7 classification metrics — Ensemble row dropped if present."""
    path = os.path.join(_OUTPUT, "phase7_classification_metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    drop_idx = [i for i in df.index if "ensemble" in str(i).lower()]
    return df.drop(index=drop_idx, errors="ignore")


@st.cache_resource
def load_xgb():
    """Return (XGBoost model, LabelEncoder). Uses cache_resource for non-serialisable objects."""
    with open(os.path.join(_OUTPUT, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(_OUTPUT, "xgb_label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, le