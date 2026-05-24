import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------
# HIDE STREAMLIT TOP BAR / DEPLOY BAR
# ---------------------------------------------------

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden; height: 0px;}
        [data-testid="stDecoration"] {visibility: hidden; height: 0px;}
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
            color: #f8fafc;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -1px;
            color: #f8fafc;
        }
        .subtitle {
            font-size: 1rem;
            color: #cbd5e1;
            margin-bottom: 1.5rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(14,165,233,0.10));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .stat-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            height: 100%;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 0.25rem;
        }
        .stat-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.1;
        }
        .stat-note {
            font-size: 0.78rem;
            color: #cbd5e1;
            margin-top: 0.25rem;
        }
        .section-card {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 20px;
            padding: 1rem 1rem 0.5rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.14);
        }
        .small-badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            color: #bfdbfe;
            font-size: 0.82rem;
            border: 1px solid rgba(59,130,246,0.25);
            margin-right: 0.4rem;
        }
        .hint-box {
            background: rgba(30,41,59,0.9);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #e2e8f0;
        }
        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"

MODEL_PATH = DASHBOARD_DIR / "model.pkl"
SCALER_PATH = DASHBOARD_DIR / "scaler.pkl"
FEATURES_PATH = DASHBOARD_DIR / "feature_columns.pkl"
DATA_PATH = DASHBOARD_DIR / "dashboard_data.csv"

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------

@st.cache_resource
def load_artifacts():
    artifacts = {}

    with open(MODEL_PATH, "rb") as f:
        artifacts["model"] = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        artifacts["scaler"] = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        artifacts["feature_columns"] = pickle.load(f)

    return artifacts

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def make_risk_tier(prob):
    if prob >= 0.75:
        return "Critical Risk"
    elif prob >= 0.40:
        return "Suspicious"
    return "Clear"

def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

# ---------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------

df = load_data()
artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">🛡️ Real-Time Fraud Detection System</div>
        <div class="subtitle">
            Explainable AI fraud operations dashboard for monitoring suspicious transactions, exploring risk, and reviewing model-driven insights.
        </div>
        <span class="small-badge">Live Risk Scoring</span>
        <span class="small-badge">Explainable AI</span>
        <span class="small-badge">Fraud Operations</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("Model file not found. Save `dashboard/model.pkl` first.")
    st.stop()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Transaction Explorer", "SHAP Explainer"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

if "TransactionAmt" in df.columns:
    min_amt = float(df["TransactionAmt"].min())
    max_amt = float(df["TransactionAmt"].max())
else:
    min_amt, max_amt = 0.0, 1.0

amt_range = st.sidebar.slider(
    "Transaction Amount Range",
    min_value=min_amt,
    max_value=max_amt,
    value=(min_amt, max_amt)
)

amount_mask = pd.Series(True, index=df.index)
if "TransactionAmt" in df.columns:
    amount_mask = df["TransactionAmt"].between(amt_range[0], amt_range[1])

# ---------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------

if page == "Overview":
    total_transactions = len(df)
    total_fraud = int(df["TrueLabel"].sum()) if "TrueLabel" in df.columns else 0
    fraud_rate = (total_fraud / total_transactions * 100) if total_transactions else 0
    avg_fraud_amount = (
        df.loc[df["TrueLabel"] == 1, "TransactionAmt"].mean()
        if "TrueLabel" in df.columns and "TransactionAmt" in df.columns
        else np.nan
    )

    st.markdown("### Overview")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Transactions", f"{total_transactions:,}", "Filtered dashboard records"),
        ("Total Fraud", f"{total_fraud:,}", "Positive class count"),
        ("Fraud Rate", f"{fraud_rate:.2f}%", "Share of fraud cases"),
        ("Avg Fraud Amount", format_money(avg_fraud_amount), "Average suspicious amount"),
    ]

    for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0; color:#f8fafc;">Transaction Amount by Risk Tier</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "RiskTier" in df.columns and "TransactionAmt" in df.columns:
            fig1 = px.box(
                df,
                x="RiskTier",
                y="TransactionAmt",
                color="RiskTier",
                points=False,
                log_y=True,
                title=None,
                color_discrete_map={
                    "Clear": "#38bdf8",
                    "Suspicious": "#f59e0b",
                    "Critical Risk": "#ef4444",
                },
            )
            fig1.update_layout(
                height=480,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            fig1.update_xaxes(title="Risk Tier", gridcolor="rgba(148,163,184,0.15)")
            fig1.update_yaxes(title="Transaction Amount (Log Scale)", gridcolor="rgba(148,163,184,0.15)")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Required columns are not available for this chart.")

    with right:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0; color:#f8fafc;">Risk Distribution</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "RiskTier" in df.columns:
            tier_counts = df["RiskTier"].value_counts().reset_index()
            tier_counts.columns = ["RiskTier", "Count"]

            fig2 = px.pie(
                tier_counts,
                names="RiskTier",
                values="Count",
                hole=0.5,
                title=None,
                color="RiskTier",
                color_discrete_map={
                    "Clear": "#38bdf8",
                    "Suspicious": "#f59e0b",
                    "Critical Risk": "#ef4444",
                },
            )
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("RiskTier column not available.")

    st.markdown(
        """
        <div class="hint-box">
            <b>Quick insight:</b> Clear transactions usually dominate the dataset, while the small Critical Risk segment is the one that needs immediate review.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# TRANSACTION EXPLORER
# ---------------------------------------------------

elif page == "Transaction Explorer":
    st.markdown("### Transaction Explorer")

    cols_for_view = [
        c for c in ["TransactionID", "TransactionAmt", "FraudProbability", "RiskTier", "TrueLabel", "HourOfDay", "DeviceType"]
        if c in df.columns
    ]

    view_df = df.loc[amount_mask, cols_for_view] if cols_for_view else df.loc[amount_mask]

    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0; color:#f8fafc;">Filtered Transactions</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(view_df.head(200), use_container_width=True, height=420)

    with top_right:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0; color:#f8fafc;">Search & Focus</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "TransactionID" in df.columns:
            search_id = st.text_input("Search TransactionID")

            if search_id:
                result = df.loc[df["TransactionID"].astype(str) == str(search_id), cols_for_view]
                if result.empty:
                    st.warning("TransactionID not found.")
                else:
                    st.dataframe(result, use_container_width=True)
                    prob = float(result["FraudProbability"].iloc[0]) if "FraudProbability" in result.columns else np.nan

                    if pd.notna(prob):
                        st.metric("Fraud Probability", f"{prob:.3f}")
                        st.metric("Risk Tier", make_risk_tier(prob))

                        fig = px.bar(
                            x=["Fraud Probability"],
                            y=[prob],
                            range_y=[0, 1],
                            title="Live Risk Score",
                            color_discrete_sequence=["#38bdf8"],
                        )
                        fig.update_layout(
                            height=320,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.0)",
                            font=dict(color="#e2e8f0"),
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        if prob >= 0.75:
                            st.error("This transaction is highly suspicious and should be reviewed immediately.")
                        elif prob >= 0.40:
                            st.warning("This transaction shows mixed signals and deserves manual review.")
                        else:
                            st.success("This transaction appears low risk based on the model.")
        else:
            st.info("TransactionID is not available in this dashboard sample.")

# ---------------------------------------------------
# SHAP EXPLAINER PAGE
# ---------------------------------------------------

elif page == "SHAP Explainer":
    st.markdown("### SHAP Explainer")

    st.markdown(
        """
        <div class="section-card">
            <p style="margin:0; color:#cbd5e1;">
                This dashboard uses the SHAP outputs generated during notebook analysis. Enter a TransactionID to inspect a transaction and read its explanation summary.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "TransactionID" in df.columns and "FraudProbability" in df.columns:
        search_id = st.text_input("TransactionID for explanation")

        if search_id:
            raw_row = df.loc[df["TransactionID"].astype(str) == str(search_id)].head(1)

            if raw_row.empty:
                st.warning("TransactionID not found.")
            else:
                st.subheader("Transaction Details")
                st.dataframe(raw_row, use_container_width=True)

                prob = float(raw_row["FraudProbability"].iloc[0])
                tier = make_risk_tier(prob)

                m1, m2 = st.columns(2)
                m1.metric("Fraud Probability", f"{prob:.3f}")
                m2.metric("Risk Tier", tier)

                if tier == "Critical Risk":
                    st.error("The model sees strong fraud signals in this transaction.")
                elif tier == "Suspicious":
                    st.warning("The model sees mixed signals and suggests a closer look.")
                else:
                    st.success("The model currently sees low fraud risk.")

                st.markdown(
                    """
                    <div class="hint-box">
                        <b>Plain-English explanation:</b> The model’s decision is influenced by a mix of amount, timing, and behavior patterns. Higher-risk transactions usually show unusual combinations compared with normal customer activity.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                waterfall_candidates = [
                    DASHBOARD_DIR / f"shap_{search_id}.png",
                    DASHBOARD_DIR / "shap_waterfall.png",
                    DASHBOARD_DIR / "shap_summary.png",
                ]

                for img_path in waterfall_candidates:
                    if img_path.exists():
                        st.subheader("SHAP Waterfall Plot")
                        st.image(Image.open(img_path), use_container_width=True)
                        break
                else:
                    st.info(
                        "Save a SHAP waterfall image in the dashboard folder to display it here. The notebook phase already handles the detailed SHAP analysis."
                    )
    else:
        st.info("FraudProbability or TransactionID is not available in dashboard data.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Built for the internship final capstone.")