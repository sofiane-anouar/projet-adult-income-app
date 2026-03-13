import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_data

PALETTE = px.colors.qualitative.Set2
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


def show():
    df = load_data()

    st.markdown("# 📊 Exploration des Données")
    st.markdown("<div class='info-box'>Explorez la distribution des variables du dataset Adult Income. Utilisez les filtres pour affiner l'analyse.</div>", unsafe_allow_html=True)

    # ── Filtres ──────────────────────────────────────────────────────────────
    with st.expander("🔧 Filtres interactifs", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            genders = st.multiselect("Genre", df["gender"].unique(), default=df["gender"].unique())
        with fc2:
            races = st.multiselect("Race", df["race"].unique(), default=df["race"].unique())
        with fc3:
            age_range = st.slider("Tranche d'âge", int(df["age"].min()), int(df["age"].max()), (18, 90))

    dff = df[
        df["gender"].isin(genders)
        & df["race"].isin(races)
        & df["age"].between(age_range[0], age_range[1])
    ]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📌 Métriques Clés</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lignes filtrées", f"{len(dff):,}", f"{len(dff)-len(df):,}" if len(dff) != len(df) else None)
    k2.metric("Âge médian", f"{dff['age'].median():.0f} ans")
    k3.metric("Heures/semaine (moy)", f"{dff['hours-per-week'].mean():.1f} h")
    k4.metric("Taux >50K$", f"{dff['income_binary'].mean()*100:.1f}%")

    st.markdown("<div class='section-header'>📈 Visualisations</div>", unsafe_allow_html=True)

    # ── Viz 1 — Distribution variable cible ──────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Variable Cible", "👥 Groupes Sensibles", "🎂 Âge", "💼 Éducation & Emploi", "🌍 Autres Variables"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = dff["income"].value_counts().reset_index()
            counts.columns = ["income", "count"]
            fig = px.pie(
                counts, values="count", names="income",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
            )
            fig = styled_fig(fig, "Répartition des Revenus (≤50K vs >50K)")
            fig.update_traces(textinfo="percent+label", textfont_size=13, pull=[0, 0.05])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                dff, x="age", color="income", barmode="overlay",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
                nbins=40, opacity=0.8,
            )
            fig = styled_fig(fig, "Distribution de l'Âge par Revenu")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            gender_income = dff.groupby(["gender", "income"]).size().reset_index(name="count")
            fig = px.bar(
                gender_income, x="gender", y="count", color="income", barmode="group",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
            )
            fig = styled_fig(fig, "Revenu par Genre")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            race_income = (
                dff.groupby("race")["income_binary"].mean().mul(100).reset_index()
            )
            race_income.columns = ["race", "pct_above_50k"]
            race_income = race_income.sort_values("pct_above_50k", ascending=True)
            fig = px.bar(
                race_income, x="pct_above_50k", y="race", orientation="h",
                color="pct_above_50k", color_continuous_scale="Blues",
                text=race_income["pct_above_50k"].round(1).astype(str) + "%",
            )
            fig = styled_fig(fig, "Taux de Revenu >50K$ par Race (%)")
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap race x genre
        pivot = dff.pivot_table(values="income_binary", index="race", columns="gender", aggfunc="mean") * 100
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="Blues",
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(title="% >50K"),
            )
        )
        fig = styled_fig(fig, "Heatmap : Taux >50K$ par Race et Genre (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                dff, x="income", y="age", color="income",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
            )
            fig = styled_fig(fig, "Distribution de l'Âge par Catégorie de Revenu")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            age_rate = (
                dff.groupby("age_group")["income_binary"].mean().mul(100).reset_index()
            )
            fig = px.bar(
                age_rate, x="age_group", y="income_binary",
                color="income_binary", color_continuous_scale="Blues",
                text=age_rate["income_binary"].round(1).astype(str) + "%",
                labels={"income_binary": "% >50K$", "age_group": "Groupe d'âge"},
            )
            fig = styled_fig(fig, "Taux >50K$ par Groupe d'Âge (%)")
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter age vs hours
        fig = px.scatter(
            dff.sample(min(3000, len(dff)), random_state=42),
            x="age", y="hours-per-week", color="income",
            color_discrete_sequence=["#38bdf8", "#f59e0b"],
            opacity=0.5,
            labels={"hours-per-week": "Heures/semaine", "age": "Âge"},
        )
        fig = styled_fig(fig, "Âge vs Heures Travaillées par Semaine (échantillon 3 000)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            edu_order = (
                dff.groupby("education")["income_binary"].mean()
                .sort_values(ascending=False).index.tolist()
            )
            edu_df = (
                dff.groupby("education")["income_binary"].mean().mul(100).reset_index()
            )
            fig = px.bar(
                edu_df, x="income_binary", y="education", orientation="h",
                color="income_binary", color_continuous_scale="Teal",
                category_orders={"education": edu_order},
                labels={"income_binary": "% >50K$"},
            )
            fig = styled_fig(fig, "Taux >50K$ par Niveau d'Éducation")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            occ_df = (
                dff.groupby("occupation")["income_binary"].mean().mul(100)
                .sort_values(ascending=True).reset_index()
            )
            fig = px.bar(
                occ_df, x="income_binary", y="occupation", orientation="h",
                color="income_binary", color_continuous_scale="Purp",
                labels={"income_binary": "% >50K$"},
            )
            fig = styled_fig(fig, "Taux >50K$ par Catégorie Professionnelle")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            ms_df = dff.groupby(["marital-status", "income"]).size().reset_index(name="count")
            fig = px.bar(
                ms_df, x="marital-status", y="count", color="income", barmode="stack",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
            )
            fig = styled_fig(fig, "Revenu par Statut Marital")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            wc_df = dff["workclass"].value_counts().reset_index()
            wc_df.columns = ["workclass", "count"]
            fig = px.pie(
                wc_df, values="count", names="workclass",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig = styled_fig(fig, "Répartition par Type d'Employeur")
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    # ── Aperçu des données ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗄️ Aperçu des Données</div>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#64748b'>{len(dff):,} lignes · {dff.shape[1]} colonnes (après filtrage)</small>", unsafe_allow_html=True)
    st.dataframe(dff.head(200), use_container_width=True, height=300)

    # Download button
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger les données filtrées (CSV)",
        csv, "adult_filtered.csv", "text/csv",
    )
