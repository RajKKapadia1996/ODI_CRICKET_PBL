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
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

df = load_data()

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
    Explore player stats, trends, and predictions in One Day International Cricket!
    ...
    """)

# --- DATA OVERVIEW ---
elif page == "Data Overview":
    st.header("Data Overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
    st.write("**Columns:**", ", ".join(df.columns))

# --- VISUALIZATIONS ---
elif page == "Visualizations":
    st.title("📊 ODI Cricket Visualizations")
    try:
        # 1. Top run scorers
        st.subheader("1️⃣ Top 10 Run Scorers")
        top_runs = df[['player', 'runs']].sort_values("runs", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=top_runs, y="player", x="runs", palette="crest", ax=ax)
        ax.set_title("Top 10 Run Scorers")
        st.pyplot(fig)

        # 2. Top wicket takers
        st.subheader("2️⃣ Top 10 Wicket Takers")
        top_wickets = df[['player', 'wickets']].sort_values("wickets", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=top_wickets, y="player", x="wickets", palette="magma", ax=ax)
        ax.set_title("Top 10 Wicket Takers")
        st.pyplot(fig)

        # 3. Runs vs. Matches (scatter)
        st.subheader("3️⃣ Runs vs. Matches")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=df, x="matches", y="runs", hue="player_type", ax=ax)
        ax.set_title("Runs vs. Matches")
        st.pyplot(fig)
        ...
    except Exception as e:
        st.error(f"Visualization error: {e}")

# --- CLUSTERING & SEGMENTATION ---
elif page == "Clustering & Segmentation":
    st.title("🧬 Player Segmentation (Clustering)")
    try:
        features = ["runs", "strike_rate", "average", "wickets", "economy", "matches"]
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        X_clustered = X.copy()
        X_clustered["cluster"] = clusters

        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=X_clustered, x="runs", y="wickets", hue="cluster", palette="Set1", ax=ax)
        ax.set_title("Player Segments by Runs & Wickets")
        st.pyplot(fig)

        st.markdown("**Cluster Counts:**")
        st.dataframe(X_clustered["cluster"].value_counts().rename("Count"))

    except Exception as e:
        st.error(f"Clustering error: {e}")

# --- REGRESSION ---
elif page == "Regression":
    st.title("📈 Regression: Predict Runs")
    try:
        features = ["balls", "strike_rate", "average", "matches"]
        target = "runs"
        data = df[features + [target]].dropna()
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.3f} | **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x=y_test, y=y_pred, color="#045FB4", ax=ax)
        ax.set_xlabel("Actual Runs")
        ax.set_ylabel("Predicted Runs")
        ax.set_title("Actual vs. Predicted Runs")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(y_test - y_pred, bins=30, color="#A9D0F5", kde=True, ax=ax)
        ax.set_title("Residuals (Actual - Predicted)")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Regression error: {e}")

# --- CLASSIFICATION ---
elif page == "Classification":
    st.title("🔮 Classification: Predict High Scorer")
    try:
        df["highscorer"] = (df["runs"] >= 5000).astype(int)
        features = ["balls", "strike_rate", "average", "matches"]
        X = df[features].dropna()
        y = df.loc[X.index, "highscorer"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2%}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        importances = pd.Series(clf.feature_importances_, index=features)
        fig, ax = plt.subplots()
        importances.sort_values().plot.barh(color="orchid", ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)
        st.write(classification_report(y_test, y_pred))
    except Exception as e:
        st.error(f"Classification error: {e}")

# --- ASSOCIATION RULES ---
elif page == "Association Rules":
    st.title("🧩 Association Rules Mining")
    try:
        assoc_df = pd.DataFrame()
        assoc_df["highscorer"] = df["runs"] > 5000
        assoc_df["aggressive"] = df["strike_rate"] > 90
        assoc_df["highwickets"] = df["wickets"] > 150
        assoc_df["goodcatcher"] = df["catches"] > 75
        assoc_df["allrounder"] = ((df["runs"] > 3000) & (df["wickets"] > 75))
        assoc_df["veteran"] = df["matches"] > 150
        assoc_df = assoc_df.astype(int)
        freq_items = apriori(assoc_df, min_support=0.1, use_colnames=True)
        rules = asso
