import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils.data_loader import load_data
from utils.fairness import demographic_parity_difference, disparate_impact_ratio, equalized_odds_difference

DARK_TEMPLATE = "plotly_dark"
PAPER_BG = "#0f172a"
PLOT_BG = "#1e293b"


def styled_fig(fig, title=""):
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=14, color="#f1f5f9"), x=0.01),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor="#334155", linecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", linecolor="#334155")
    return fig


@st.cache_data
def train_model(model_name="Logistic Regression"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    df = load_data()
    features = ["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss",
                 "gender", "race", "workclass", "marital-status", "occupation"]
    df_model = df[features + ["income_binary", "age_group"]].dropna()

    # Encode categoricals
    cat_cols = ["gender", "race", "workclass", "marital-status", "occupation"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model = df_model.copy()
        df_model[col + "_enc"] = le.fit_transform(df_model[col])
        le_dict[col] = le

    enc_features = [c for c in features if c not in cat_cols] + [c + "_enc" for c in cat_cols]
    X = df_model[enc_features].values
    y = df_model["income_binary"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_model.index, test_size=0.3, random_state=42, stratify=y
    )
    df_test = df_model.loc[idx_test]

    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=500, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return y_test, y_pred, df_test, metrics, cm


def plot_confusion_matrix(cm, title):
    labels = ["≤50K", ">50K"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels, y=labels,
            colorscale="Blues",
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            showscale=False,
        )
    )
    fig = styled_fig(fig, title)
    fig.update_layout(
        xaxis_title="Prédiction",
        yaxis_title="Réalité",
        height=280,
    )
    return fig


def show():
    df = load_data()
    st.markdown("# 🤖 Modélisation & Analyse de Fairness")
    st.markdown(
        "<div class='info-box'>Entraînement d'un modèle de classification binaire pour prédire le revenu, "
        "suivi d'une analyse de fairness sur les <b>prédictions du modèle</b>.</div>",
        unsafe_allow_html=True,
    )

    # ── Choix du modèle ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚙️ Configuration</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        model_name = st.selectbox("Algorithme", ["Logistic Regression", "Random Forest"])

    st.markdown("<div class='warn-box'>⏳ L'entraînement peut prendre quelques secondes (Random Forest ~30s).</div>", unsafe_allow_html=True)

    with st.spinner(f"Entraînement du modèle {model_name}…"):
        y_test, y_pred, df_test, metrics, cm = train_model(model_name)

    # ── Performances globales ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Performances Globales</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("🔬 Précision", f"{metrics['precision']*100:.1f}%")
    m3.metric("📡 Rappel", f"{metrics['recall']*100:.1f}%")
    m4.metric("⚖️ F1-Score", f"{metrics['f1']*100:.1f}%")

    col_cm, col_info = st.columns([1, 1])
    with col_cm:
        fig = plot_confusion_matrix(cm, "Matrice de Confusion — Global")
        st.plotly_chart(fig, use_container_width=True)
    with col_info:
        st.markdown(
            f"""
            <div class='info-box' style='margin-top:0'>
            <b>Interprétation :</b><br>
            Le modèle atteint <b>{metrics['accuracy']*100:.1f}% d'accuracy</b> globale.
            Un F1-score de <b>{metrics['f1']*100:.1f}%</b> indique un bon équilibre précision/rappel.<br><br>
            Cependant, ces métriques globales <b>masquent des disparités entre groupes</b>.
            L'analyse de fairness ci-dessous révèle si le modèle est équitable.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Fairness sur prédictions ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚖️ Métriques de Fairness sur les Prédictions</div>", unsafe_allow_html=True)

    tab_g, tab_r, tab_a = st.tabs(["👥 Genre", "🌍 Race", "🎂 Âge"])

    def fairness_tab(attr_col, priv, unpriv, label):
        s = df_test[attr_col].astype(str).values
        dpd = demographic_parity_difference(y_test, y_pred, s)
        dir_ = disparate_impact_ratio(y_test, y_pred, s, unpriv, priv)
        eod = equalized_odds_difference(y_test, y_pred, s)

        c1, c2, c3 = st.columns(3)
        c1.metric("Diff. Parité Démographique", f"{dpd['difference']:.3f}")
        c2.metric("Ratio Impact Disproportionné", f"{dir_['ratio']:.3f}")
        c3.metric("Diff. Equal Opportunity (TPR)", f"{eod['tpr_diff']:.3f}")

        # Bar chart taux prédits par groupe
        pred_rates = pd.DataFrame([
            {"Groupe": g, "Taux prédit >50K$ (%)": r * 100, "Type": "Prédit"}
            for g, r in dpd["rates"].items()
        ])
        real_rates_dict = {
            g: np.mean(y_test[s == g]) * 100 for g in dpd["rates"]
        }
        real_rates = pd.DataFrame([
            {"Groupe": g, "Taux prédit >50K$ (%)": r, "Type": "Réel"}
            for g, r in real_rates_dict.items()
        ])
        combined = pd.concat([pred_rates, real_rates])

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                combined, x="Groupe", y="Taux prédit >50K$ (%)",
                color="Type", barmode="group",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
                text=combined["Taux prédit >50K$ (%)"].round(1).astype(str) + "%",
            )
            fig = styled_fig(fig, f"Taux Réel vs Prédit par {label}")
            fig.update_traces(textposition="outside")
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Confusion matrices par groupe (top 2)
            groups = list(dpd["rates"].keys())[:2]
            for g in groups:
                mask = s == g
                if mask.sum() < 10:
                    continue
                from sklearn.metrics import confusion_matrix as cm_fn
                cm_g = cm_fn(y_test[mask], y_pred[mask])
                fig = plot_confusion_matrix(cm_g, f"Matrice de Confusion — {g}")
                fig.update_layout(height=240)
                st.plotly_chart(fig, use_container_width=True)

        # TPR / FPR par groupe
        tpr_df = pd.DataFrame([
            {"Groupe": g, "Taux": v * 100, "Métrique": "TPR (Rappel)"}
            for g, v in eod["tpr"].items()
        ] + [
            {"Groupe": g, "Taux": v * 100, "Métrique": "FPR (Faux Positifs)"}
            for g, v in eod["fpr"].items()
        ])
        fig = px.bar(
            tpr_df, x="Groupe", y="Taux", color="Métrique", barmode="group",
            color_discrete_sequence=["#34d399", "#f87171"],
            labels={"Taux": "%"},
        )
        fig = styled_fig(fig, f"TPR & FPR par {label} (Equal Opportunity)")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with tab_g:
        fairness_tab("gender", "Male", "Female", "Genre")
    with tab_r:
        fairness_tab("race", "White", "Black", "Race")
    with tab_a:
        fairness_tab("age_group", "36-45", "18-25", "Groupe d'Âge")

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏁 Conclusion</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='warn-box'>
        <b>Synthèse :</b> Les métriques de fairness révèlent que le modèle <b>reproduit et amplifie</b>
        les biais présents dans les données d'entraînement. Les femmes, les minorités ethniques et les 
        jeunes adultes obtiennent des taux de prédiction >50K$ significativement inférieurs, 
        même à caractéristiques équivalentes.<br><br>
        <b>Recommandations :</b><br>
        • <b>Pré-traitement :</b> Reweighing, oversampling des groupes défavorisés<br>
        • <b>En cours d'entraînement :</b> Contraintes de fairness (Fairlearn, AIF360)<br>
        • <b>Post-traitement :</b> Calibration des seuils par groupe, reject option classification<br>
        • <b>Gouvernance :</b> Audit régulier, documentation des biais (Model Card), supervision humaine
        </div>
        """,
        unsafe_allow_html=True,
    )
