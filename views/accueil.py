import streamlit as st
from utils.data_loader import load_data


def show():
    df = load_data()

    # Hero banner
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
                    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
                    border: 1px solid #1e3a5f;'>
            <h1 style='color:#38bdf8 !important; font-size:2.4rem; margin:0 0 0.4rem 0;'>
                ⚖️ Détection de Biais — Adult Income
            </h1>
            <p style='color:#94a3b8 !important; font-size:1.05rem; margin:0;'>
                Analyse des inégalités de revenus aux États-Unis · Dataset UCI Adult (1994)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    total_rows = len(df)
    pct_above = (df["income_binary"].mean() * 100)
    pct_female = (df["gender"].value_counts(normalize=True).get("Female", 0) * 100)
    n_countries = df["native-country"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Observations", f"{total_rows:,}")
    c2.metric("💰 Revenus >50K$", f"{pct_above:.1f}%")
    c3.metric("♀️ Part féminine", f"{pct_female:.1f}%")
    c4.metric("🌍 Pays représentés", n_countries)

    st.markdown("<div class='section-header'>🎯 Contexte et Problématique</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            <div class='info-box'>
            <b>Origine du dataset :</b> Extrait du recensement américain de 1994 par Ronny Kohavi
            et Barry Becker (UCI Machine Learning Repository). Il contient des informations
            socio-économiques sur <b>48 842 individus</b> et est couramment utilisé pour entraîner
            des modèles de classification binaire.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='info-box'>
            <b>Problématique :</b> Peut-on prédire si un individu gagne <b>plus ou moins de 50 000 $
            par an</b> à partir de ses caractéristiques personnelles (âge, éducation, profession…) ?
            Si oui, ce modèle est-il équitable pour tous les groupes démographiques ?
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='warn-box'>
            <b>⚠️ Biais potentiels identifiés :</b> Les données reflètent la société américaine
            de 1994. Des inégalités structurelles liées au <b>genre</b>, à la <b>race</b> et à l'<b>âge</b>
            sont susceptibles d'être encodées dans le dataset et amplifiées par un modèle
            de machine learning non audité.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<div class='section-header'>📋 Variables du Dataset</div>", unsafe_allow_html=True)
        variables = {
            "age": "Âge de l'individu",
            "workclass": "Type d'employeur",
            "education": "Niveau d'études",
            "marital-status": "Situation matrimoniale",
            "occupation": "Catégorie professionnelle",
            "race": "Origine ethnique",
            "gender": "Genre",
            "hours-per-week": "Heures travaillées/semaine",
            "native-country": "Pays d'origine",
            "income": "🎯 Variable cible (≤50K / >50K)",
        }
        for var, desc in variables.items():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #1e293b;'>"
                f"<code style='color:#38bdf8; background:#0f172a; padding:1px 5px; border-radius:4px;'>{var}</code>"
                f"<span style='color:#94a3b8; font-size:0.85rem;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-header'>🗺️ Plan de l'Application</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    pages_info = [
        ("🏠", "Accueil", "Présentation du projet, contexte et KPIs globaux"),
        ("📊", "Exploration", "Visualisations de la distribution des données et des groupes"),
        ("⚠️", "Détection de Biais", "Métriques de fairness : parité démographique, impact disproportionné"),
        ("🤖", "Modélisation", "Entraînement d'un modèle ML et analyse des biais sur les prédictions"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], pages_info):
        col.markdown(
            f"""
            <div style='background:#1e293b; border-radius:12px; padding:1rem; height:140px;
                        border:1px solid #334155; text-align:center;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='color:#f1f5f9; font-weight:700; margin:0.3rem 0;'>{title}</div>
                <div style='color:#64748b; font-size:0.8rem; line-height:1.4;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
