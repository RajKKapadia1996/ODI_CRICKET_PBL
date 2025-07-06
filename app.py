import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ODI Cricket Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ODI Cricket Data new.csv")
    # Aggressively clean and normalize column names
    df.columns = (
        df.columns.str.replace('\u200b', '', regex=True)
                  .str.replace('\xa0', '', regex=True)
                  .str.strip()
    )
    return df

df = load_data()

# Print cleaned columns for debugging (optional)
# st.write("Cleaned columns:", df.columns.tolist())

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Section", [
    "Home",
    "Data Overview",
    "Visualizations",
    "Clustering & Segmentation",
    "Regression",
    "Classification",
    "Association Rules"
])

# --- HOME ---
if page == "Home":
    st.title("🏏 ODI Cricket Analytics Dashboard")
    st.markdown("""
