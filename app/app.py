# Streamlit entrypoint

import os
import io
import json
from datetime import datetime
from typing import Optional, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------- Config ---------
st.set_page_config(page_title="Model Inference Dashboard", page_icon="📊", layout="wide")

# --------- Helpers ---------
DATE_COL_CANDIDATES = ["timestamp", "created_at", "inference_time", "time"]
MODEL_COL_CANDIDATES = ["model", "model_name"]
DATASET_COL_CANDIDATES = ["dataset", "dataset_name"]
STATUS_COL_CANDIDATES = ["status", "outcome"]
TARGET_COL_CANDIDATES = ["target", "label", "ground_truth"]
PRED_COL_CANDIDATES = ["prediction", "pred", "output"]
SCORE_COL_CANDIDATES = ["score", "confidence", "prob", "probability"]
DURATION_COL_CANDIDATES = ["latency_ms", "latency", "duration_ms", "inference_ms"]

@st.cache_data(show_spinner=False)
def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower.endswith(".parquet"):
        return pd.read_parquet(buffer)
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return pd.read_json(buffer, lines=True)
    if lower.endswith(".json"):
        data = json.load(io.BytesIO(file_bytes))
        return pd.json_normalize(data)
    raise ValueError("Unsupported file type. Use CSV, Parquet, JSON, or JSONL.")


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# --------- Sidebar: Data ---------
st.sidebar.header("Data")
uploader = st.sidebar.file_uploader("Upload inference results", type=["csv", "parquet", "json", "jsonl"], accept_multiple_files=False)
example_btn = st.sidebar.button("Load example data")

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.schema = {}

if uploader is not None:
    df = read_table(uploader.getvalue(), uploader.name)
    st.session_state.df = df
elif example_btn:
    # Generate a small synthetic example
    rng = np.random.default_rng(42)
    n = 500
    start = pd.Timestamp("2024-01-01")
    times = pd.date_range(start=start, periods=n, freq="H")
    models = rng.choice(["gpt-4o", "llama3-70b", "mistral-large"], size=n)
    datasets = rng.choice(["prod", "staging", "eval-set"], size=n)
    status = rng.choice(["success", "error"], size=n, p=[0.9, 0.1])
    latency = np.maximum(rng.normal(800, 200, size=n).astype(float), 50)
    score = np.clip(rng.normal(0.75, 0.1, size=n), 0, 1)
    y_true = rng.integers(0, 2, size=n)
    y_pred = (score > 0.5).astype(int)
    st.session_state.df = pd.DataFrame({
        "timestamp": times,
        "model": models,
        "dataset": datasets,
        "status": status,
        "latency_ms": latency,
        "score": score,
        "target": y_true,
        "prediction": y_pred,
    })


df = st.session_state.df
if df is None:
    st.info("Upload a dataset or load the example to begin.")
    st.stop()

# Detect schema
col_time = detect_column(df, DATE_COL_CANDIDATES)
col_model = detect_column(df, MODEL_COL_CANDIDATES)
col_dataset = detect_column(df, DATASET_COL_CANDIDATES)
col_status = detect_column(df, STATUS_COL_CANDIDATES)
col_target = detect_column(df, TARGET_COL_CANDIDATES)
col_pred = detect_column(df, PRED_COL_CANDIDATES)
col_score = detect_column(df, SCORE_COL_CANDIDATES)
col_latency = detect_column(df, DURATION_COL_CANDIDATES)

# Allow manual overrides
st.sidebar.subheader("Column mapping")
col_time = st.sidebar.selectbox("Timestamp column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_time) if col_time in df.columns else 0)
col_model = st.sidebar.selectbox("Model column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_model) if col_model in df.columns else 0)
col_dataset = st.sidebar.selectbox("Dataset column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_dataset) if col_dataset in df.columns else 0)
col_status = st.sidebar.selectbox("Status column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_status) if col_status in df.columns else 0)
col_target = st.sidebar.selectbox("Target/Label column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_target) if col_target in df.columns else 0)
col_pred = st.sidebar.selectbox("Prediction column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_pred) if col_pred in df.columns else 0)
col_score = st.sidebar.selectbox("Score/Confidence column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_score) if col_score in df.columns else 0)
col_latency = st.sidebar.selectbox("Latency (ms) column", [None] + list(df.columns), index=( [None] + list(df.columns) ).index(col_latency) if col_latency in df.columns else 0)

# Prepare df
if col_time:
    df[col_time] = coerce_datetime(df[col_time])
    df = df.sort_values(col_time)

# --------- Sidebar: Filters ---------
st.sidebar.header("Filters")

if col_time:
    min_date = pd.to_datetime(df[col_time]).min()
    max_date = pd.to_datetime(df[col_time]).max()
    date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
else:
    date_range = None

models = sorted(df[col_model].dropna().unique().tolist()) if col_model else []
datasets = sorted(df[col_dataset].dropna().unique().tolist()) if col_dataset else []
statuses = sorted(df[col_status].dropna().unique().tolist()) if col_status else []

sel_models = st.sidebar.multiselect("Model", options=models, default=models[:3] if models else [])
sel_datasets = st.sidebar.multiselect("Dataset", options=datasets, default=datasets[:3] if datasets else [])
sel_status = st.sidebar.multiselect("Status", options=statuses, default=statuses if statuses else [])

filtered = df.copy()
if col_time and date_range:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered[col_time] >= start_dt) & (filtered[col_time] < end_dt)]
if col_model and sel_models:
    filtered = filtered[filtered[col_model].isin(sel_models)]
if col_dataset and sel_datasets:
    filtered = filtered[filtered[col_dataset].isin(sel_datasets)]
if col_status and sel_status:
    filtered = filtered[filtered[col_status].isin(sel_status)]

# --------- KPIs ---------
st.title("Model Inference Dashboard")

kpi_cols = st.columns(4)

with kpi_cols[0]:
    total = len(filtered)
    st.metric("Total inferences", f"{total:,}")
with kpi_cols[1]:
    if col_status:
        success_rate = (filtered[col_status] == "success").mean() * 100
        st.metric("Success rate", f"{success_rate:.1f}