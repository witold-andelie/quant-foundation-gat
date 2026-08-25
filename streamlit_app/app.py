"""Streamlit navigation entry point."""

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(
    page_title="Quant Alpha Foundation",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("home.py", title="Start Here", icon=":material/home:", default=True)
gat_forecasting = st.Page(
    "pages/8_GAT_Forecasting.py",
    title="Key Findings",
    icon=":material/hub:",
)
performance = st.Page(
    "pages/1_Performance.py",
    title="Performance",
    icon=":material/monitoring:",
)
factor_research = st.Page(
    "pages/5_Factor_Research.py",
    title="Factor Research",
    icon=":material/science:",
)
alpha_decay = st.Page(
    "pages/2_Alpha_Decay.py",
    title="Alpha Decay",
    icon=":material/timeline:",
)
market_data = st.Page(
    "pages/3_Market_Data.py",
    title="Market Data",
    icon=":material/table_chart:",
)
overview = st.Page(
    "pages/7_Overview.py",
    title="Platform & Reproducibility",
    icon=":material/account_tree:",
)
data_pipeline = st.Page(
    "pages/4_Data_Pipeline.py",
    title="Data Pipeline",
    icon=":material/lan:",
)
live_streaming = st.Page(
    "pages/6_Live_Streaming.py",
    title="Live Streaming",
    icon=":material/stream:",
)

navigation = st.navigation(
    {
        "Start Here": [home, gat_forecasting],
        "Research Evidence": [
            performance,
            factor_research,
            alpha_decay,
            market_data,
        ],
        "Engineering Appendix": [
            overview,
            data_pipeline,
            live_streaming,
        ],
    }
)
navigation.run()
