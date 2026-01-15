import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="India Air Quality Application",
    layout="wide"
)

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("full_df.csv")

df = load_data()

# Identify columns
AQI_COL = "AQI"

pollutant_cols = [
    "PM2.5","PM10","NO","NO2","NOx","NH3",
    "CO","SO2","O3","Benzene","Toluene","Xylene"
]
pollutant_cols = [c for c in pollutant_cols if c in df.columns]

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to section:",
    ["Data Overview", "Exploratory Data Analysis", "Modelling & Prediction"]
)

# -----------------------------
# App title
# -----------------------------
st.title("India Air Quality Analysis Application")

st.write(
    "This application was developed to support the research question: "
    "**Which pollutants are most responsible for poor air quality (AQI) in India?** "
    "The app integrates dataset exploration, exploratory data analysis, and "
    "machine learning model results."
)

# =========================================================
# PAGE 1 — DATA OVERVIEW
# =========================================================
if page == "Data Overview":
    st.header("1️⃣ Data Overview")

    st.write(
        "This section provides a general overview of the air quality dataset "
        "used in this study."
    )

    col1, col2 = st.columns(2)
    col1.metric("Number of rows", df.shape[0])
    col2.metric("Number of columns", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.info(
        "Inference: Understanding the structure and size of the dataset "
        "helps contextualise subsequent analysis and modelling results."
    )

# =========================================================
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# =========================================================
elif page == "Exploratory Data Analysis":
    st.header("2️⃣ Exploratory Data Analysis (EDA)")

    st.write(
        "This section explores the relationship between pollutant concentrations "
        "and the Air Quality Index (AQI)."
    )

    selected_pollutant = st.selectbox(
        "Select a pollutant to analyse:",
        pollutant_cols
    )

    eda_df = df[[selected_pollutant, AQI_COL]].dropna()

    st.subheader(f"{selected_pollutant} vs AQI")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=eda_df,
        x=selected_pollutant,
        y=AQI_COL,
        alpha=0.5,
        ax=ax
    )
    ax.set_xlabel(selected_pollutant)
    ax.set_ylabel("AQI")
    st.pyplot(fig)

    corr_val = eda_df.corr().iloc[0, 1]
    st.write(f"**Correlation coefficient:** {corr_val:.2f}")

    st.info(
        "Inference: A positive correlation indicates that higher concentrations "
        "of the selected pollutant are associated with higher AQI values, "
        "suggesting poorer air quality."
    )

# =========================================================
# PAGE 3 — MODELLING & PREDICTION (TASK 3 INTEGRATION)
# =========================================================
else:
    st.header("3️⃣ Modelling & Prediction")

    st.write(
        "This section presents the results of machine learning models developed "
        "in Task 3 to identify which pollutants contribute most to poor air quality."
    )

    # -----------------------------
    # Model performance (from Task 3)
    # -----------------------------
    st.subheader("Model Performance Comparison")

    results_df = pd.DataFrame({
        "Model": ["Random Forest", "Ridge Regression"],
        "MAE": [20.94, 30.99],
        "RMSE": [41.45, 53.44],
        "R²": [0.91, 0.85]
    })

    st.dataframe(results_df, use_container_width=True)

    st.write(
        "Inference: The Random Forest model achieved lower prediction errors "
        "and a higher R² score compared to Ridge Regression. Therefore, "
        "Random Forest was selected for interpreting pollutant importance."
    )

    # -----------------------------
    # Pollutant importance (from Task 3)
    # -----------------------------
    st.subheader("Pollutant Importance (Random Forest)")

    importance_df = pd.DataFrame({
        "Pollutant": [
            "PM2.5","CO","PM10","NO","O3","NOx",
            "SO2","NO2","Toluene","Benzene","Xylene","NH3"
        ],
        "Importance": [
            0.483224,0.365851,0.046539,0.036107,0.013908,0.013037,
            0.009728,0.008970,0.007383,0.006456,0.005533,0.003262
        ]
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(importance_df["Pollutant"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Pollutant Importance in Predicting AQI")
    st.pyplot(fig)

    st.info(
        "Inference: PM2.5 is identified as the most influential pollutant, "
        "followed by carbon monoxide (CO). This confirms that particulate "
        "matter and combustion-related emissions are the primary drivers of "
        "poor air quality in the dataset, directly answering the research question."
    )
