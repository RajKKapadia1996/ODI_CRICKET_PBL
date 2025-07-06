import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="ODI Cricket Analytics Dashboard", layout="wide")

# Data Load
@st.cache_data
def load_data():
    df = pd.read_csv("ODI Cricket Data new.csv")
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
    # 1. Top run scorers
    st.subheader("1️⃣ Top 10 Run Scorers")
    top_runs = df[['Player', 'Runs']].sort_values("Runs", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=top_runs, y="Player", x="Runs", palette="crest", ax=ax)
    ax.set_title("Top 10 Run Scorers")
    st.pyplot(fig)

    # 2. Top wicket takers
    st.subheader("2️⃣ Top 10 Wicket Takers")
    top_wickets = df[['Player', 'Wickets']].sort_values("Wickets", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=top_wickets, y="Player", x="Wickets", palette="magma", ax=ax)
    ax.set_title("Top 10 Wicket Takers")
    st.pyplot(fig)

    # 3. Runs vs. Matches (scatter)
    st.subheader("3️⃣ Runs vs. Matches")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x="Matches", y="Runs", hue="Player Type", ax=ax)
    ax.set_title("Runs vs. Matches")
    st.pyplot(fig)

    # 4. Average vs. Strike Rate by Player Type
    st.subheader("4️⃣ Average vs. Strike Rate by Player Type")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df, x="Average", y="Strike Rate", hue="Player Type", style="Player Type", s=80, ax=ax)
    ax.set_title("Batting Average vs. Strike Rate")
    st.pyplot(fig)

    # 5. Bowler Economy Distribution
    st.subheader("5️⃣ Bowler Economy Rate Distribution")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["Economy"].dropna(), bins=30, color="orchid", kde=True, ax=ax)
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
    num_cols = ["Runs","Balls","Fours","Sixes","Strike Rate","Average","Wickets","Overs","Economy","Bowling Average","Catches","Matches"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Numerical Feature Correlations")
    st.pyplot(fig)

    # 9. Sixes vs. Fours
    st.subheader("9️⃣ Sixes vs. Fours (Top 20 Batsmen)")
    top_bats = df.sort_values("Runs", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.scatterplot(data=top_bats, x="Fours", y="Sixes", hue="Player", palette="husl", s=120, ax=ax)
    ax.set_title("Sixes vs. Fours")
    st.pyplot(fig)

    # 10. Matches by Country (Pie)
    st.subheader("🔟 Players by Country")
    country_counts = df["Country"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(country_counts, labels=country_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set3", 10))
    ax.set_title("Players by Country")
    st.pyplot(fig)

# --- CLUSTERING & SEGMENTATION ---
elif page == "Clustering & Segmentation":
    st.title("🧬 Player Segmentation (Clustering)")
    st.markdown("Players clustered by Batting & Bowling performance.")

    # Select features for clustering
    features = ["Runs", "Strike Rate", "Average", "Wickets", "Economy", "Matches"]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    X_clustered = X.copy()
    X_clustered["Cluster"] = clusters

    # Show cluster scatterplot
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=X_clustered, x="Runs", y="Wickets", hue="Cluster", palette="Set1", ax=ax)
    ax.set_title("Player Segments by Runs & Wickets")
    st.pyplot(fig)

    st.markdown("**Cluster Counts:**")
    st.dataframe(X_clustered["Cluster"].value_counts().rename("Count"))

    st.markdown("""
    - **Cluster 0:** All-rounders / Balanced players  
    - **Cluster 1:** Batting specialists (high runs, low wickets)  
    - **Cluster 2:** Bowling specialists (high wickets, lower runs)
    """)

# --- REGRESSION ---
elif page == "Regression":
    st.title("📈 Regression: Predict Runs")
    # Features & Target
    features = ["Balls", "Strike Rate", "Average", "Matches"]
    target = "Runs"
    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.3f} | **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Scatter: Actual vs. Predicted
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=y_test, y=y_pred, color="#045FB4", ax=ax)
    ax.set_xlabel("Actual Runs")
    ax.set_ylabel("Predicted Runs")
    ax.set_title("Actual vs. Predicted Runs")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    st.pyplot(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(y_test - y_pred, bins=30, color="#A9D0F5", kde=True, ax=ax)
    ax.set_title("Residuals (Actual - Predicted)")
    st.pyplot(fig)

# --- CLASSIFICATION ---
elif page == "Classification":
    st.title("🔮 Classification: Predict High Scorer")
    st.markdown("Classifies if a player is a 'High Scorer' (Runs >= 5000)")

    # Binarize target
    df["HighScorer"] = (df["Runs"] >= 5000).astype(int)
    features = ["Balls", "Strike Rate", "Average", "Matches"]
    X = df[features].dropna()
    y = df.loc[X.index, "HighScorer"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2%}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=features)
    fig, ax = plt.subplots()
    importances.sort_values().plot.barh(color="orchid", ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    st.write(classification_report(y_test, y_pred))

# --- ASSOCIATION RULES ---
elif page == "Association Rules":
    st.title("🧩 Association Rules Mining")
    st.markdown("Finds frequent player attribute combinations.")

    # Prepare for association rules: Binarize some attributes
    assoc_df = pd.DataFrame()
    assoc_df["HighScorer"] = df["Runs"] > 5000
    assoc_df["Aggressive"] = df["Strike Rate"] > 90
    assoc_df["HighWickets"] = df["Wickets"] > 150
    assoc_df["GoodCatcher"] = df["Catches"] > 75
    assoc_df["AllRounder"] = ((df["Runs"] > 3000) & (df["Wickets"] > 75))
    assoc_df["Veteran"] = df["Matches"] > 150

    assoc_df = assoc_df.astype(int)

    # Apriori
    freq_items = apriori(assoc_df, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(10))
    st.write("Example: [HighScorer, Aggressive] → AllRounder")

st.info("Built with 🐍 Streamlit, sklearn, and mlxtend. Update the code for your custom objectives or more complex models!")
