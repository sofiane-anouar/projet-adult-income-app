import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils.data_loader import load_data
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

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


def fairness_badge(value, metric="di"):
    """Return HTML badge based on threshold."""
    if metric == "di":
        if value >= 0.8:
            return f"<span class='badge-ok'>✅ Équitable (≥0.8)</span>"
        elif value >= 0.6:
            return f"<span class='badge-warn'>⚠️ Limite (0.6–0.8)</span>"
        else:
            return f"<span class='badge-bad'>🚨 Discriminatoire (&lt;0.6)</span>"
    else:  # parity difference
        if value <= 0.1:
            return f"<span class='badge-ok'>✅ Équitable (≤0.10)</span>"
        elif value <= 0.2:
            return f"<span class='badge-warn'>⚠️ Modéré (0.10–0.20)</span>"
        else:
            return f"<span class='badge-bad'>🚨 Élevé (&gt;0.20)</span>"


def show():
    df = load_data()

    st.markdown("# ⚠️ Détection de Biais")
    st.markdown(
        "<div class='info-box'>Cette page analyse les biais présents dans le dataset en calculant des "
        "<b>métriques de fairness</b> reconnues : la Parité Démographique et le Ratio d'Impact Disproportionné.</div>",
        unsafe_allow_html=True,
    )

    # ── Sélecteur d'attribut sensible ────────────────────────────────────────
    st.markdown("<div class='section-header'>🎛️ Attribut Sensible</div>", unsafe_allow_html=True)
    attr = st.selectbox(
        "Choisir l'attribut sensible à analyser :",
        ["Genre (gender)", "Race (race)", "Âge (age_group)"],
        label_visibility="collapsed",
    )
    attr_col = {"Genre (gender)": "gender", "Race (race)": "race", "Âge (age_group)": "age_group"}[attr]

    # ── Explication du biais ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📖 Quel Biais Analysons-Nous ?</div>", unsafe_allow_html=True)

    explanations = {
        "gender": {
            "title": "Biais de Genre",
            "desc": "Le genre est un attribut protégé fondamental. Dans un contexte de prédiction de revenus, "
                    "un modèle biaisé sur le genre perpétue les inégalités salariales structurelles entre hommes "
                    "et femmes. Si le taux de prédiction >50K est significativement plus élevé pour les hommes, "
                    "le modèle pourrait défavoriser les femmes dans des décisions d'embauche ou de crédit.",
            "probleme": "Décisions d'embauche, d'octroi de crédit, de promotion automatisées défavorables aux femmes.",
            "priv": "Male", "unpriv": "Female",
        },
        "race": {
            "title": "Biais Racial",
            "desc": "La race est une caractéristique protégée dans le droit américain (Equal Credit Opportunity Act). "
                    "Des disparités dans les prédictions de revenus selon l'origine ethnique reflètent et amplifient "
                    "des discriminations systémiques. Un modèle entraîné sur ces données encodera ces biais historiques.",
            "probleme": "Discrimination algorithmique reproduisant des inégalités systémiques dans l'accès aux ressources.",
            "priv": "White", "unpriv": "Black",
        },
        "age_group": {
            "title": "Biais d'Âge",
            "desc": "L'âgisme est une forme de discrimination reconnue (Age Discrimination in Employment Act, 1967). "
                    "Des modèles désavantagent souvent les jeunes (manque d'expérience) et les seniors (biais de productivité). "
                    "Cette analyse compare les tranches d'âge pour identifier des disparités injustes.",
            "probleme": "Discrimination à l'embauche, décisions financières défavorables aux jeunes ou aux seniors.",
            "priv": "36-45", "unpriv": "18-25",
        },
    }

    info = explanations[attr_col]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='info-box'><b>{info['title']}</b><br><br>{info['desc']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div class='warn-box'><b>⚡ Impact potentiel</b><br><br>{info['probleme']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>📊 Métriques de Fairness</div>", unsafe_allow_html=True)

    y = df["income_binary"].values
    s = df[attr_col].astype(str).values

    # Métrique 1 : Parité Démographique
    dpd = demographic_parity_difference(y_true=y, y_pred=y, sensitive_attribute=s)

    # Métrique 2 : Impact Disproportionné
    dir_ = disparate_impact_ratio(
        y_true=y, y_pred=y, sensitive_attribute=s,
        unprivileged_value=info["unpriv"],
        privileged_value=info["priv"],
    )

    # Display metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📐 Différence de Parité Démographique", f"{dpd['difference']:.3f}")
        st.markdown(fairness_badge(dpd["difference"], "parity"), unsafe_allow_html=True)
        st.caption("Différence max des taux >50K entre groupes. Seuil équitable : ≤ 0.10")

    with m2:
        st.metric("⚖️ Ratio d'Impact Disproportionné", f"{dir_['ratio']:.3f}")
        st.markdown(fairness_badge(dir_["ratio"], "di"), unsafe_allow_html=True)
        st.caption(f"Taux {info['unpriv']} / Taux {info['priv']}. Règle des 4/5 : seuil ≥ 0.80")

    with m3:
        st.metric(
            f"📉 Écart de taux ({info['priv']} vs {info['unpriv']})",
            f"{dir_['rate_privileged']*100:.1f}% vs {dir_['rate_unprivileged']*100:.1f}%",
        )
        st.caption("Taux de revenu >50K pour groupe privilégié vs non-privilégié")

    # ── Visualisation des taux par groupe ────────────────────────────────────
    st.markdown("<div class='section-header'>📉 Visualisation des Disparités</div>", unsafe_allow_html=True)

    rates_df = pd.DataFrame([
        {"groupe": g, "taux": r * 100}
        for g, r in dpd["rates"].items()
    ]).sort_values("taux", ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            rates_df, x="groupe", y="taux",
            color="taux", color_continuous_scale="RdYlGn",
            text=rates_df["taux"].round(1).astype(str) + "%",
            labels={"taux": "Taux >50K$ (%)", "groupe": attr},
        )
        fig = styled_fig(fig, f"Taux de Revenu >50K$ par {attr}")
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.add_hline(y=rates_df["taux"].mean(), line_dash="dot", line_color="#94a3b8",
                      annotation_text="Moyenne", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Stacked bar: count by group and income
        grp_count = df.groupby([attr_col, "income"]).size().reset_index(name="count")
        fig = px.bar(
            grp_count, x=attr_col, y="count", color="income", barmode="stack",
            color_discrete_sequence=["#38bdf8", "#f59e0b"],
            labels={attr_col: attr, "count": "Nombre d'individus"},
        )
        fig = styled_fig(fig, f"Distribution des Revenus par {attr}")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    # ── Comparaison genre + race croisée (pour genre seulement) ──────────────
    if attr_col == "gender":
        st.markdown("<div class='section-header'>🔗 Intersectionnalité : Genre × Race</div>", unsafe_allow_html=True)
        cross = df.groupby(["race", "gender"])["income_binary"].mean().mul(100).reset_index()
        fig = px.bar(
            cross, x="race", y="income_binary", color="gender", barmode="group",
            color_discrete_sequence=["#f472b6", "#38bdf8"],
            text=cross["income_binary"].round(1).astype(str) + "%",
            labels={"income_binary": "% >50K$"},
        )
        fig = styled_fig(fig, "Taux >50K$ par Race et Genre (intersectionnalité)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif attr_col == "race":
        st.markdown("<div class='section-header'>🔗 Intersectionnalité : Race × Genre</div>", unsafe_allow_html=True)
        cross = df.groupby(["race", "gender"])["income_binary"].mean().mul(100).reset_index()
        fig = px.bar(
            cross, x="race", y="income_binary", color="gender", barmode="group",
            color_discrete_sequence=["#f472b6", "#38bdf8"],
            text=cross["income_binary"].round(1).astype(str) + "%",
            labels={"income_binary": "% >50K$"},
        )
        fig = styled_fig(fig, "Taux >50K$ par Race et Genre")
        fig.update_traces(textposition="outside")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    elif attr_col == "age_group":
        c1, c2 = st.columns(2)
        with c1:
            age_gen = df.groupby(["age_group", "gender"])["income_binary"].mean().mul(100).reset_index()
            fig = px.line(
                age_gen, x="age_group", y="income_binary", color="gender",
                markers=True, color_discrete_sequence=["#f472b6", "#38bdf8"],
                labels={"income_binary": "% >50K$", "age_group": "Groupe d'âge"},
            )
            fig = styled_fig(fig, "Évolution du Taux >50K$ par Âge et Genre")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            age_race = df.groupby(["age_group", "race"])["income_binary"].mean().mul(100).reset_index()
            fig = px.line(
                age_race, x="age_group", y="income_binary", color="race",
                markers=True,
                labels={"income_binary": "% >50K$", "age_group": "Groupe d'âge"},
            )
            fig = styled_fig(fig, "Évolution du Taux >50K$ par Âge et Race")
            st.plotly_chart(fig, use_container_width=True)

    # ── Interprétation ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💡 Interprétation</div>", unsafe_allow_html=True)

    top_group = rates_df.iloc[-1]["groupe"]
    bot_group = rates_df.iloc[0]["groupe"]
    top_rate = rates_df.iloc[-1]["taux"]
    bot_rate = rates_df.iloc[0]["taux"]
    gap = top_rate - bot_rate

    st.markdown(
        f"""
        <div class='info-box'>
        <b>🔍 Biais détecté ({info['title']}) :</b><br><br>
        Le groupe <b>"{top_group}"</b> présente un taux de revenu >50K$ de <b>{top_rate:.1f}%</b>,
        contre seulement <b>{bot_rate:.1f}%</b> pour le groupe <b>"{bot_group}"</b>,
        soit un écart de <b>{gap:.1f} points de pourcentage</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    interpretations = {
        "gender": f"""
        <b>👥 Groupe défavorisé :</b> Les femmes ({dir_['rate_unprivileged']*100:.1f}% de taux >50K$) 
        sont significativement sous-représentées dans la tranche des hauts revenus par rapport aux hommes 
        ({dir_['rate_privileged']*100:.1f}%). Le ratio DI de {dir_['ratio']:.2f} 
        {'est inférieur au seuil légal de 0.80' if dir_['ratio'] < 0.8 else 'dépasse le seuil de 0.80 mais reste préoccupant'}.<br><br>
        <b>⚡ Impact réel :</b> Un modèle ML entraîné sur ces données discriminerait les femmes dans des 
        décisions automatisées (scoring crédit, présélection de candidatures). Cette inégalité reflète 
        les écarts salariaux structurels de 1994 et ne doit pas être reproduite par des systèmes d'IA.<br><br>
        <b>✅ Recommandations :</b> Rééchantillonnage des données (oversampling du groupe minoritaire), 
        contraintes de fairness lors de l'entraînement (Fairlearn), ou post-processing des prédictions 
        (seuils différenciés par groupe).
        """,
        "race": f"""
        <b>🌍 Groupe défavorisé :</b> Les individus Noirs ({dir_['rate_unprivileged']*100:.1f}%) 
        sont fortement sous-représentés dans la catégorie >50K$ par rapport aux Blancs 
        ({dir_['rate_privileged']*100:.1f}%). Ratio DI = {dir_['ratio']:.2f} 
        {'→ discrimination au sens de la règle des 4/5' if dir_['ratio'] < 0.8 else '→ disparité notable'}.<br><br>
        <b>⚡ Impact réel :</b> Ce biais encode des inégalités structurelles liées à l'histoire 
        américaine (ségrégation, discrimination à l'embauche). Un système algorithmique les perpétuerait 
        dans des contextes critiques : crédit, logement, recrutement.<br><br>
        <b>✅ Recommandations :</b> Utiliser des algorithmes de débiaisage (Reweighing, Adversarial Debiasing), 
        auditer régulièrement les décisions par groupe, intégrer des contraintes légales (Equal Credit Opportunity Act).
        """,
        "age_group": f"""
        <b>🧑 Groupe défavorisé :</b> Les jeunes adultes (18-25 ans) présentent le taux le plus bas 
        de revenus >50K$, ce qui est en partie structurel (début de carrière). Néanmoins, 
        un ratio DI de {dir_['ratio']:.2f} indique une {'forte' if dir_['ratio'] < 0.6 else 'certaine'} 
        disparité entre jeunes et actifs de 36-45 ans.<br><br>
        <b>⚡ Impact réel :</b> Un modèle qui prédirait des revenus faibles pour les jeunes pourrait 
        biaiser des décisions de crédit ou d'assurance défavorablement pour cette tranche, 
        sans tenir compte de la trajectoire de carrière individuelle.<br><br>
        <b>✅ Recommandations :</b> Distinguer les inégalités structurelles des biais injustes, 
        utiliser des variables proxy moins corrélées à l'âge (compétences, expérience), 
        surveiller l'impact des prédictions par tranche d'âge.
        """,
    }

    st.markdown(
        f"<div class='info-box'>{interpretations[attr_col]}</div>",
        unsafe_allow_html=True,
    )

    # ── Tableau récapitulatif des métriques par groupe ────────────────────────
    st.markdown("<div class='section-header'>📋 Tableau Récapitulatif</div>", unsafe_allow_html=True)
    summary_rows = []
    for g, r in dpd["rates"].items():
        summary_rows.append({
            "Groupe": g,
            "Taux >50K$ (%)": round(r * 100, 2),
            "Écart vs moyenne (pts)": round(r * 100 - rates_df["taux"].mean(), 2),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Taux >50K$ (%)", ascending=False)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
