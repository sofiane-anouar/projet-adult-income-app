# ⚖️ Adult Income — Détection de Biais

Application Streamlit pour l'analyse des biais dans le dataset Adult Income (UCI, 1994).

## 📋 Description

Cette application explore les **inégalités de revenus** aux États-Unis et détecte les **biais algorithmiques** liés au genre, à la race et à l'âge dans la prédiction de revenus >50K$/an.

## 🗂️ Structure

```
adult_income_app/
├── app.py                  # Point d'entrée principal
├── adult.csv               # Dataset UCI Adult Income
├── requirements.txt
├── pages/
│   ├── accueil.py          # Page 1 : Accueil & KPIs
│   ├── exploration.py      # Page 2 : Exploration des données
│   ├── biais.py            # Page 3 : Détection de biais (fairness)
│   └── modelisation.py     # Page 4 : Modèle ML & fairness sur prédictions
└── utils/
    ├── data_loader.py      # Chargement & cache des données
    └── fairness.py         # Métriques : parité démographique, DI ratio, equal opportunity
```

## 🚀 Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Pages

| Page | Contenu |
|------|---------|
| 🏠 Accueil | Présentation, KPIs globaux, plan de l'app |
| 📊 Exploration | 8+ visualisations, filtres interactifs, export CSV |
| ⚠️ Détection de Biais | Parité démographique, ratio DI, analyse intersectionnelle |
| 🤖 Modélisation | Logistic Regression / Random Forest, métriques par groupe |

## ⚙️ Dataset

- **Source :** UCI Adult Income Dataset (Kaggle)
- **Observations :** 48 842
- **Variables :** 15 (âge, éducation, genre, race, occupation, revenus…)
- **Variable cible :** income (≤50K / >50K)

## 👥 Auteurs
Sofiane Anouar ABDELHAK
