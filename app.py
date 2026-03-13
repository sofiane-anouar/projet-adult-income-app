import streamlit as st

st.set_page_config(
    page_title="Adult Income — Détection de Biais",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio > label { font-weight: 600; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; color: #a0aec0 !important; }

    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #38bdf8 !important; }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.6rem 0 0.6rem 0;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.3rem;
    }

    /* ---- Info box ---- */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ---- Warning box ---- */
    .warn-box {
        background: #1e1a0e;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        color: #fde68a;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ---- Fairness badge ---- */
    .badge-ok   { background:#064e3b; color:#6ee7b7; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.82rem; }
    .badge-warn { background:#451a03; color:#fcd34d; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.82rem; }
    .badge-bad  { background:#450a0a; color:#fca5a5; border-radius:6px; padding:2px 8px; font-weight:700; font-size:0.82rem; }

    /* ---- Main background ---- */
    .stApp { background-color: #0f172a; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; }
    p, li { color: #cbd5e1 !important; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #0f3460 !important; color: #38bdf8 !important; }

    /* Divider */
    hr { border-color: #1e293b !important; }

    /* DataFrame */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Adult Income")
    st.markdown("<small style='color:#64748b'>Détection de Biais — Parcours A</small>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📊 Exploration des Données", "⚠️ Détection de Biais", "🤖 Modélisation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Dataset · UCI Adult Income<br>48 842 observations · 15 variables</small>",
        unsafe_allow_html=True,
    )

# ── Route to pages ───────────────────────────────────────────────────────────
if page == "🏠 Accueil":
    from views.accueil import show
    show()
elif page == "📊 Exploration des Données":
    from views.exploration import show
    show()
elif page == "⚠️ Détection de Biais":
    from views.biais import show
    show()
elif page == "🤖 Modélisation":
    from views.modelisation import show
    show()
