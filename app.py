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
    df.columns = df.columns.str.strip()  # Remove extra spaces, but keep original casing
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
    - 🧬 **Clustering:** Segment players by style & performance
    - 🤖 **Classification:** Predict high scorers or wicket-takers
    - 📈 **Regression:** Forecast runs/wickets
    - 🧩 **Association Rules:** Find player attribute patterns
    - 📊 **10+ Visualizations:** Explore top performers, correlations, and more
    Use the sidebar to navigate.
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
        # 1. Top 10 Run Scorers
        st.subheader("1️⃣ Top 10 Run Scorers")
        top_runs = df[['Player', 'Runs']].sort_values("Runs", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=top_runs, y="Player", x="Runs", palette="crest", ax=ax)
        ax.set_title("Top 10 Run Scorers")
        st.pyplot(fig)

        # 2. Top 10 Wicket Takers
        st.subheader("2️⃣ Top 10 Wicket Takers")
        top_wickets = df[['Player', 'Wkts']].sort_values("Wkts", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=top_wickets, y="Player", x="Wkts", palette="magma", ax=ax)
        ax.set_title("Top 10 Wicket Takers")
        st.pyplot(fig)

        # 3. Runs vs. Matches (scatter)
        st.subheader("3️⃣ Runs vs. Matches")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=df, x="Mat", y="Runs", hue="Player Type", ax=ax)
        ax.set_title("Runs vs. Matches")
        st.pyplot(fig)

        # 4. Average vs. Strike Rate by Player Type
        st.subheader("4️⃣ Average vs. Strike Rate by Player Type")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=df, x="Ave", y="SR", hue="Player Type", style="Player Type", s=80, ax=ax)
        ax.set_title("Batting Average vs. Strike Rate")
        st.pyplot(fig)

        # 5. Bowler Economy Distribution
        st.subheader("5️⃣ Bowler Economy Rate Distribution")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(df["Econ"].dropna(), bins=30, color="orchid", kde=True, ax=ax)
        ax.set_xlabel("Economy Rate")
        ax.set_title("Bowling Economy Distribution")
        st.pyplot(fig)

        # 6. Player Type Counts
        st.subheader("6️⃣ Player Type Distribution")
        type_counts = df["Player Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=type_counts.index, y=type_counts.values, palette="pastel", ax=ax)
        ax.set_title("Player Types")
        st.pyplot(fig)

        # 7. Country-wise Top Batsmen
        st.subheader("7️⃣ Top 5 Batsmen by Country")
        countries = df["Country"].unique()[:6]
        for country in countries:
            sub = df[df["Country"]==country].sort_values("Runs", ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(7,2))
            sns.barplot(data=sub, x="Player", y="Runs", palette="flare", ax=ax)
            ax.set_title(f"Top Batsmen - {country}")
            st.pyplot(fig)

        # 8. Correlation heatmap
        st.subheader("8️⃣ Feature Correlation Matrix")
        num_cols = ["Runs","BF","Ave","Wkts","Econ","Mat","SR","Ct"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9,7))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Numerical Feature Correlations")
        st.pyplot(fig)

        # 9. Sixes vs. Fours
        st.subheader("9️⃣ Sixes vs. Fours (Top 20 Batsmen)")
        top_bats = df.sort_values("Runs", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(7,4))
        sns.scatterplot(data=top_bats, x="4s", y="6s", hue="Player", palette="husl", s=120, ax=ax)
        ax.set_title("Sixes vs. Fours")
        st.pyplot(fig)

        # 10. Players by Country (Pie)
        st.subheader("🔟 Players by Country")
        country_counts = df["Country"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.pie(country_counts, labels=country_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set3", 10))
        ax.set_title("Players by Country")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Visualization error: {e}")

# --- CLUSTERING & SEGMENTATION ---
elif page == "Clustering & Segmentation":
    st.title("🧬 Player Segmentation (Clustering)")
    try:
        features = ["Runs", "SR", "Ave", "Wkts", "Econ", "Mat"]
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        X_clustered = X.copy()
        X_clustered["cluster"] = clusters

        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=X_clustered, x="Runs", y="Wkts", hue="cluster", palette="Set1", ax=ax)
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
        features = ["BF", "SR", "Ave", "Mat"]
        target = "Runs"
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
        df["highscorer"] = (df["Runs"] >= 5000).astype(int)
        features = ["BF", "SR", "Ave", "Mat"]
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
        assoc_df["highscorer"] = df["Runs"] > 5000
        assoc_df["aggressive"] = df["SR"] > 90
        assoc_df["highwickets"] = df["Wkts"] > 150
        assoc_df["goodcatcher"] = df["Ct"] > 75
        assoc_df["allrounder"] = ((df["Runs"] > 3000) & (df["Wkts"] > 75))
        assoc_df["veteran"] = df["Mat"] > 150
        assoc_df = assoc_df.astype(int)
        freq_items = apriori(assoc_df, min_support=0.1, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
        st.write("Example: [highscorer, aggressive] → allrounder")
    except Exception as e:
        st.error(f"Association rules error: {e}")

st.info("Built with 🐍 Streamlit, sklearn, and mlxtend. Update the code for your custom objectives or more complex models!")
