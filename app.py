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

# ---------- THEME: DARK MODE (use config.toml for Streamlit) ----------
sns.set_style("darkgrid")
plt.rcParams['axes.facecolor'] = "#232946"
plt.rcParams['figure.facecolor'] = "#232946"
plt.rcParams['axes.labelcolor'] = "#F7F7FF"
plt.rcParams['text.color'] = "#F7F7FF"
plt.rcParams['xtick.color'] = "#F7F7FF"
plt.rcParams['ytick.color'] = "#F7F7FF"

st.set_page_config(page_title="🏏 ODI Cricket Analytics Dashboard", page_icon=":cricket_bat_and_ball:", layout="wide")

# ---------- SIDEBAR -----------
st.sidebar.title("🏏 Navigation")
page = st.sidebar.radio(
    "Go to...",
    [
        "Home",
        "Data Overview",
        "Visualizations",
        "Clustering & Segmentation",
        "Regression",
        "Classification",
        "Association Rules"
    ]
)

@st.cache_data
def load_data():
    df = pd.read_csv("ODI Cricket Data new.csv")
    df.columns = (df.columns.str.replace('\u200b', '', regex=True)
                              .str.replace('\xa0', '', regex=True)
                              .str.strip())
    num_cols = [
        "total_runs", "total_balls_faced", "total_wickets_taken",
        "total_matches_played", "matches_won", "percentage"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# ---------- BEAUTIFY TABLE ----------
def prettify_table(data):
    return data.style.background_gradient(cmap='plasma').set_properties(**{
        'font-size': '15px',
        'color': 'white',
        'background-color': '#232946',
        'border': '1.5px solid #08F7FE'
    })

# ---------- PAGES -------------
if page == "Home":
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#232946 60%,#08F7FE11 100%);padding:2rem 2.5rem 2rem 2.5rem;border-radius:20px;margin-bottom:30px;border:2px solid #08F7FE;">
            <h1 style="color:#08F7FE;font-size:2.5rem;margin-bottom:1rem;">🏏 ODI Cricket Analytics Dashboard</h1>
            <h3 style="color:#F7F7FF;font-weight:600;margin-bottom:1.2rem;">Explore International Cricket Stats with <span style="color:#08F7FE;">AI-Powered Insights</span></h3>
            <p style="font-size:1.25rem;color:#F7F7FF;margin-bottom:1.2rem;">
                <b>This dashboard lets you:</b>
                <ul style="font-size:1.1rem;line-height:2;">
                  <li>📊 <b>Visualize</b> top scorers, teams, & player stats</li>
                  <li>🧬 <b>Cluster</b> players by performance & style</li>
                  <li>🔮 <b>Classify</b> high scorers & top wicket-takers</li>
                  <li>📈 <b>Predict</b> player runs (regression)</li>
                  <li>🧩 <b>Discover</b> hidden patterns (association rules)</li>
                </ul>
                <b>Use the sidebar</b> to explore each section.<br>
                <span style="color:#08F7FE;font-size:1.1rem;">Built for students, analysts, fans, and cricket scouts!</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Data Overview":
    st.header("🗂️ Data Overview")
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp; | &nbsp; **Columns:** {df.shape[1]}")
    st.dataframe(prettify_table(df.head(25)), use_container_width=True)

elif page == "Visualizations":
    st.title("📊 Visualizations")
    try:
        # Top 10 Run Scorers
        st.subheader("1️⃣ Top 10 Run Scorers")
        top_runs = df[['player_name', 'total_runs']].sort_values("total_runs", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9,4))
        sns.barplot(data=top_runs, y="player_name", x="total_runs", palette="cool", ax=ax, edgecolor="black")
        ax.set_title("Top 10 Run Scorers", fontsize=15, color="#08F7FE")
        ax.set_xlabel("Runs")
        ax.set_ylabel("Player")
        sns.despine()
        st.pyplot(fig)

        # Top 10 Wicket Takers
        if 'total_wickets_taken' in df.columns:
            st.subheader("2️⃣ Top 10 Wicket Takers")
            top_wickets = df[['player_name', 'total_wickets_taken']].sort_values("total_wickets_taken", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(9,4))
            sns.barplot(data=top_wickets, y="player_name", x="total_wickets_taken", palette="magma", ax=ax, edgecolor="black")
            ax.set_title("Top 10 Wicket Takers", fontsize=15, color="#08F7FE")
            ax.set_xlabel("Wickets")
            ax.set_ylabel("Player")
            sns.despine()
            st.pyplot(fig)

        # Runs vs. Matches (scatter)
        st.subheader("3️⃣ Runs vs. Matches")
        if 'total_matches_played' in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            sc = sns.scatterplot(data=df, x="total_matches_played", y="total_runs", hue="role", s=110, palette="viridis", alpha=0.9, edgecolor="#232946", ax=ax)
            ax.set_title("Runs vs. Matches", fontsize=14, color="#08F7FE")
            ax.set_xlabel("Matches Played")
            ax.set_ylabel("Runs")
            ax.legend(title="Role", bbox_to_anchor=(1.01, 1), loc='upper left', labelcolor='#F7F7FF')
            st.pyplot(fig)

        # Player Role Distribution
        st.subheader("4️⃣ Player Role Distribution")
        role_counts = df["role"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=role_counts.index, y=role_counts.values, palette="Accent", ax=ax, edgecolor="black")
        ax.set_title("Player Roles", fontsize=13, color="#08F7FE")
        ax.set_xlabel("Role")
        ax.set_ylabel("Players")
        sns.despine()
        st.pyplot(fig)

        # Team-wise Top Batsmen
        st.subheader("5️⃣ Top 5 Batsmen by Team")
        teams = df["team"].unique()[:6]
        for team in teams:
            sub = df[df["team"]==team].sort_values("total_runs", ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(7,2.5))
            sns.barplot(data=sub, x="player_name", y="total_runs", palette="plasma", ax=ax, edgecolor="black")
            ax.set_title(f"Top Batsmen - {team}", fontsize=12, color="#08F7FE")
            ax.set_xlabel("Player")
            ax.set_ylabel("Runs")
            ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig)

        # Correlation heatmap
        st.subheader("6️⃣ Feature Correlation Matrix")
        num_cols = ["total_runs", "total_balls_faced", "total_wickets_taken", "total_matches_played", "matches_won", "percentage"]
        num_cols = [col for col in num_cols if col in df.columns]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", linewidths=0.5, ax=ax, cbar_kws={'shrink':.7})
        ax.set_title("Numerical Feature Correlations", fontsize=13, color="#08F7FE")
        st.pyplot(fig)

        # Top 20 by Percentage
        if 'percentage' in df.columns:
            st.subheader("7️⃣ Top 20 Players by Percentage")
            top_perc = df.sort_values("percentage", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=top_perc, y="player_name", x="percentage", hue="role", dodge=False, ax=ax, edgecolor="black", palette="Spectral")
            ax.set_title("Top 20 Players by Percentage", fontsize=13, color="#08F7FE")
            ax.set_xlabel("Percentage")
            ax.set_ylabel("Player")
            st.pyplot(fig)

        # Players by Team (Pie)
        st.subheader("8️⃣ Players by Team")
        team_counts = df["team"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6,6))
        wedges, texts, autotexts = ax.pie(team_counts, labels=team_counts.index, autopct="%1.1f%%", colors=sns.color_palette("twilight", 10),
                                          startangle=90, wedgeprops={"edgecolor":"#08F7FE", "linewidth":2}, pctdistance=0.82, textprops={'fontsize':12, 'color':'white'})
        ax.set_title("Players by Team", fontsize=15, color="#08F7FE")
        centre_circle = plt.Circle((0,0),0.60,fc='#232946')
        fig.gca().add_artist(centre_circle)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Visualization error: {e}")

elif page == "Clustering & Segmentation":
    st.title("🧬 Player Segmentation (Clustering)")
    try:
        features = ["total_runs", "total_balls_faced", "total_wickets_taken", "total_matches_played"]
        features = [col for col in features if col in df.columns]
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        X_clustered = X.copy()
        X_clustered["cluster"] = clusters

        fig, ax = plt.subplots(figsize=(8,5))
        palette = sns.color_palette("bright", 3)
        sns.scatterplot(data=X_clustered, x="total_runs", y="total_wickets_taken", hue="cluster", palette=palette, s=120, ax=ax, alpha=0.8, edgecolor="#11111B")
        ax.set_title("Player Segments by Runs & Wickets", fontsize=14, color="#08F7FE")
        ax.set_xlabel("Total Runs")
        ax.set_ylabel("Total Wickets")
        ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc='upper left', labelcolor='#F7F7FF')
        st.pyplot(fig)

        st.markdown("**Cluster Counts:**")
        st.dataframe(prettify_table(X_clustered["cluster"].value_counts().rename("Count").reset_index()), use_container_width=True)

    except Exception as e:
        st.error(f"Clustering error: {e}")

elif page == "Regression":
    st.title("📈 Regression: Predict Runs")
    try:
        features = ["total_balls_faced", "total_wickets_taken", "total_matches_played"]
        features = [col for col in features if col in df.columns]
        target = "total_runs"
        if all(col in df.columns for col in features + [target]):
            data = df[features + [target]].dropna()
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            st.markdown(f"""
                <div style="background:#1A1A2E;padding:0.7rem 1.2rem;border-radius:11px;width:fit-content;margin-bottom:0.5rem;color:#08F7FE;">
                <b>R2 Score:</b> {r2_score(y_test, y_pred):.3f} &nbsp; | &nbsp; <b>RMSE:</b> {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}
                </div>
                """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7,5))
            sns.scatterplot(x=y_test, y=y_pred, color="#08F7FE", s=80, ax=ax)
            ax.set_xlabel("Actual Runs")
            ax.set_ylabel("Predicted Runs")
            ax.set_title("Actual vs. Predicted Runs", fontsize=14, color="#08F7FE")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'w--', lw=2)
            st.pyplot(fig)
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(y_test - y_pred, bins=30, color="#F2C12E", kde=True, ax=ax)
            ax.set_title("Residuals (Actual - Predicted)", fontsize=13, color="#08F7FE")
            st.pyplot(fig)
        else:
            st.warning("Required columns for regression not found.")
    except Exception as e:
        st.error(f"Regression error: {e}")

elif page == "Classification":
    st.title("🔮 Classification: Predict High Scorer")
    try:
        if "total_runs" in df.columns:
            df["highscorer"] = (df["total_runs"] >= 3000).astype(int)
            features = ["total_balls_faced", "total_wickets_taken", "total_matches_played"]
            features = [col for col in features if col in df.columns]
            if all(col in df.columns for col in features):
                X = df[features].dropna()
                y = df.loc[X.index, "highscorer"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.markdown(f"""
                <div style="background:#1A1A2E;padding:0.7rem 1.2rem;border-radius:11px;width:fit-content;margin-bottom:0.5rem;color:#08F7FE;">
                <b>Accuracy:</b> {acc:.2%}
                </div>
                """, unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                ax.set_title("Confusion Matrix", fontsize=13, color="#08F7FE")
                st.pyplot(fig)
                importances = pd.Series(clf.feature_importances_, index=features)
                fig, ax = plt.subplots()
                importances.sort_values().plot.barh(color="#F2C12E", ax=ax)
                ax.set_title("Feature Importances", fontsize=12, color="#08F7FE")
                st.pyplot(fig)
                st.text(classification_report(y_test, y_pred))
            else:
                st.warning("Required columns for classification not found.")
        else:
            st.warning("total_runs column not found for classification.")
    except Exception as e:
        st.error(f"Classification error: {e}")

elif page == "Association Rules":
    st.title("🧩 Association Rules Mining")
    try:
        assoc_df = pd.DataFrame()
        assoc_df["highscorer"] = df["total_runs"] > 1000
        assoc_df["highwickets"] = df["total_wickets_taken"] > 25
        assoc_df["allrounder"] = ((df["total_runs"] > 500) & (df["total_wickets_taken"] > 10))
        assoc_df["veteran"] = df["total_matches_played"] > 50

        st.markdown("<b>Support for each rule (fraction True):</b>", unsafe_allow_html=True)
        st.dataframe(prettify_table(assoc_df.mean().to_frame("support")), use_container_width=True)

        if assoc_df.shape[1] >= 2:
            assoc_df = assoc_df.astype(int)
            freq_items = apriori(assoc_df, min_support=0.05, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.3)
            if not rules.empty:
                st.dataframe(prettify_table(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10)), use_container_width=True)
                st.write("Example: [highscorer, highwickets] → allrounder")

                # ---- PLOT 1: Support vs Confidence Bubble Plot (Label Top 5 by Lift) ----
                st.subheader("Support vs Confidence (bubble size = Lift)")
                fig, ax = plt.subplots(figsize=(7,5))
                sc = ax.scatter(rules['support'], rules['confidence'], 
                                s=300*rules['lift'], alpha=0.7, c=rules['lift'], cmap="plasma")
                rules_sorted = rules.sort_values("lift", ascending=False).head(5)
                for i, row in rules_sorted.iterrows():
                    label = ','.join([str(a) for a in row['antecedents']]) + "→" + ','.join([str(c) for c in row['consequents']])
                    ax.annotate(label, (row['support'], row['confidence']), fontsize=11, weight='bold', alpha=0.85, color="#F7F7FF")
                ax.set_xlabel("Support", fontsize=12, color="#F7F7FF")
                ax.set_ylabel("Confidence", fontsize=12, color="#F7F7FF")
                ax.set_title("Association Rules: Support vs Confidence (Top 5)", fontsize=14, color="#08F7FE")
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("Lift", color="#08F7FE")
                st.pyplot(fig)

                # ---- PLOT 2: Heatmap of Top Rules by Confidence ----
                st.subheader("Heatmap: Confidence of Top Association Rules")
                plot_rules = rules.copy()
                plot_rules['antecedent'] = plot_rules['antecedents'].apply(lambda x: ','.join(list(x)))
                plot_rules['consequent'] = plot_rules['consequents'].apply(lambda x: ','.join(list(x)))
                pivot = plot_rules.pivot(index='antecedent', columns='consequent', values='confidence').fillna(0)
                fig, ax = plt.subplots(figsize=(7,5))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="plasma", linewidths=1, ax=ax)
                ax.set_title("Confidence Heatmap (Antecedent → Consequent)", fontsize=13, color="#08F7FE")
                st.pyplot(fig)

            else:
                st.warning("No rules found: Try lowering thresholds further or reduce min_support/confidence.")
        else:
            st.warning("Not enough columns for association rules analysis.")
    except Exception as e:
        st.error(f"Association rules error: {e}")

st.markdown(
    """
    <hr style="height:2px;border:none;color:#08F7FE;background-color:#08F7FE;">
    <center>
        <span style='color:#08F7FE;font-size:16px;'>
            Built with Streamlit, scikit-learn, matplotlib, seaborn, and mlxtend<br>
            <a href='https://github.com/RajKKapadia1996/Aviation' style='color:#F2C12E;'>View on GitHub</a>
        </span>
    </center>
    """,
    unsafe_allow_html=True
)
