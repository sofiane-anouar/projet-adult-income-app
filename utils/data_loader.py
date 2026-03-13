import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    # Standardize whitespace in string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Binary encode income
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    return df
