import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from analysis.tyre_analysis import multi_stint_analysis

st.set_page_config(page_title="F1 Analytics System", layout="wide")

st.title("🏎️ Formula 1 Performance Analytics & Prediction System")

# Sidebar Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Race Prediction", "Driver Analytics", "Podium Predictor", "Tyre Degradation & Pace Analysis"]
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    constructor_model = joblib.load("C:/F1-Project/models/constructor_points_model.joblib")
    finish_model = joblib.load("C:/F1-Project/models/finishing_position_model.joblib")
    laptime_model = joblib.load("C:/F1-Project/models/avg_lap_time_model.joblib")
    podium_model = joblib.load("C:/F1-Project/models/podium_model.joblib")


    constructor_features = joblib.load("C:/F1-Project/models/constructor_points_features.joblib")
    finish_features = joblib.load("C:/F1-Project/models/finishing_position_features.joblib")
    laptime_features = joblib.load("C:/F1-Project/models/avg_lap_time_features.joblib")
    podium_features = joblib.load("C:/F1-Project/models/podium_features.joblib")

    return (
        constructor_model,
        finish_model,
        laptime_model,
        podium_model,
        constructor_features,
        finish_features,
        laptime_features,
        podium_features
    )


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_lap_data():
    lap_df = pd.read_csv("C:/F1-Project/data/processed/lap_times_processed.csv")
    lap_df["lap_time_s"] = pd.to_numeric(lap_df["lap_time_s"], errors="coerce")
    lap_df = lap_df.dropna(subset=["lap_time_s"])
    return lap_df


@st.cache_data
def load_data():
    df = pd.read_csv("C:/F1-Project/data/processed/features_v2.csv")
    driver_stats = pd.read_csv("C:/F1-Project/data/processed/driver_stats_v1.csv")
    driver_ratings = pd.read_csv("C:/F1-Project/data/processed/driver_ratings.csv")
    drivers_lookup = pd.read_csv("C:/F1-Project/data/processed/drivers.csv")
    circuits_lookup = pd.read_csv("C:/F1-Project/data/processed/circuits_clean.csv")
    races_lookup = pd.read_csv("C:/F1-Project/data/processed/races.csv")

    # Create full name
    drivers_lookup["driver_name"] = (
        drivers_lookup["forename"] + " " + drivers_lookup["surname"]
    )

    races_lookup["race_name"] = (
        races_lookup["year"].astype(str) + " - " + races_lookup["name"]
    )

    return df, driver_stats, driver_ratings, drivers_lookup, circuits_lookup, races_lookup


(
    constructor_model,
    finish_model,
    laptime_model,
    podium_model,
    constructor_features,
    finish_features,
    laptime_features,
    podium_features
) = load_models()

lap_df = load_lap_data()

df, driver_stats, driver_ratings, drivers_lookup, circuits_lookup, races_lookup = load_data()

# =========================
# HOME PAGE
# =========================
if page == "Home":

    st.header("Project Overview")

    st.write("""
    This system predicts Formula 1 driver performance using machine learning models.
    
    It includes:
    - Constructor Points Prediction
    - Finish Position Prediction
    - Average Lap Time Prediction
    - Driver Performance Clustering
    - Composite Driver Rating System
    - Podium Probability 
    - Tyre Degradation & Pace Analysis
    """)

    st.subheader("Models Used")

    st.write("""
    - Random Forest Regressor
    - KMeans Clustering
    """)

    st.success("Developed for Final Year Project — Machine Learning for Sports Analytics")

# =========================
# RACE PREDICTION PAGE
# =========================
if page == "Race Prediction":

    st.header("Race Performance Prediction")

    # --------------------------------------------------
    # 1️⃣ Filter drivers to only those in feature dataset
    # --------------------------------------------------
    valid_driver_ids = df["driverId"].unique()

    driver_options = drivers_lookup[
        drivers_lookup["driverId"].isin(valid_driver_ids)
    ].sort_values("driver_name")

    driver_selected = st.selectbox(
        "Select Driver",
        driver_options["driver_name"]
    )

    driver_id = driver_options[
        driver_options["driver_name"] == driver_selected
    ]["driverId"].iloc[0]

    # --------------------------------------------------
    # 2️⃣ Grid Position Input
    # --------------------------------------------------
    grid_position = st.number_input(
        "Grid Position",
        min_value=1,
        max_value=20,
        value=10
    )

    # --------------------------------------------------
    # 3️⃣ Circuit Selection (Filtered)
    # --------------------------------------------------
    valid_circuits = df["circuitId"].unique()

    circuits_filtered = circuits_lookup[
        circuits_lookup["circuitId"].isin(valid_circuits)
    ].sort_values("circuit_name")

    circuit_selected_name = st.selectbox(
        "Select Circuit",
        circuits_filtered["circuit_name"]
    )

    circuit_id = circuits_filtered[
        circuits_filtered["circuit_name"] == circuit_selected_name
    ]["circuitId"].iloc[0]

    # --------------------------------------------------
    # 4️⃣ Predict Button
    # --------------------------------------------------
    if st.button("Predict Performance"):

        # Get most recent record for that driver
        driver_history = df[df["driverId"] == driver_id]

        if driver_history.empty:
            st.error("No historical feature data available for this driver.")
            st.stop()

        driver_data = driver_history.iloc[-1:].copy()

        # Update interactive inputs
        driver_data.loc[:, "grid"] = grid_position
        driver_data.loc[:, "circuitId"] = circuit_id

        # --------------------------------------------------
        # Safety: Ensure all required features exist
        # --------------------------------------------------
        missing_constructor = set(constructor_features) - set(driver_data.columns)
        missing_finish = set(finish_features) - set(driver_data.columns)
        missing_laptime = set(laptime_features) - set(driver_data.columns)

        if missing_constructor or missing_finish or missing_laptime:
            st.error("Feature mismatch detected between training and app.")
            st.write("Constructor missing:", missing_constructor)
            st.write("Finish missing:", missing_finish)
            st.write("Lap time missing:", missing_laptime)
            st.stop()

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------
        X_constructor = driver_data[constructor_features]
        X_finish = driver_data[finish_features]
        X_laptime = driver_data[laptime_features]

        constructor_pred = constructor_model.predict(X_constructor)[0]
        finish_pred = finish_model.predict(X_finish)[0]
        laptime_pred = laptime_model.predict(X_laptime)[0]

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------
        st.subheader("Prediction Results")
        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            label="Constructor Points",
            value=f"{constructor_pred:.1f}",
            help="Predicted points scored in race"
        )

        col2.metric(
            label="Finish Position",
            value=f"P{int(round(finish_pred))}"
        )

        col3.metric(
            label="Average Lap Time",
            value=f"{laptime_pred:.2f} sec"
        )

        # --------------------------------------------------
        # Sanity Checks
        # --------------------------------------------------
        if finish_pred < 1 or finish_pred > 25:
            st.warning("Finish prediction outside realistic range.")

        if laptime_pred < 60 or laptime_pred > 200:
            st.warning("Lap time looks unrealistic.")

# =========================
# DRIVER ANALYTICS PAGE
# =========================
if page == "Driver Analytics":

    st.header("Driver Performance Analytics")

    # --------------------------------------------------
    # 1️⃣ Filter drivers to only those with ratings
    # --------------------------------------------------
    valid_driver_ids = driver_ratings["driverId"].unique()

    driver_options = drivers_lookup[
        drivers_lookup["driverId"].isin(valid_driver_ids)
    ].sort_values("driver_name")

    driver_selected = st.selectbox(
        "Select Driver",
        driver_options["driver_name"]
    )

    # Convert selected name → driverId
    driver_id = driver_options[
        driver_options["driver_name"] == driver_selected
    ]["driverId"].iloc[0]

    # --------------------------------------------------
    # 2️⃣ Get Driver Rating Info Safely
    # --------------------------------------------------
    driver_info_df = driver_ratings[
        driver_ratings["driverId"] == driver_id
    ]

    if driver_info_df.empty:
        st.warning("No analytics available for this driver.")
        st.stop()

    driver_info = driver_info_df.iloc[0]

    # --------------------------------------------------
    # 3️⃣ Display Rating + Cluster
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    col1.metric("Driver Rating", f"{driver_info['rating']:.3f}")

    cluster = driver_info["cluster_label"]

    if cluster == "Elite":
        col2.success(f"Cluster: {cluster}")
    elif cluster == "Midfield":
        col2.info(f"Cluster: {cluster}")
    else:
        col2.warning(f"Cluster: {cluster}")

    # --------------------------------------------------
    # 4️⃣ Performance Statistics Table
    # --------------------------------------------------
    st.subheader("Performance Statistics")
    st.divider()

    stats_df = pd.DataFrame({
        "Metric": [
            "Average Finish Position",
            "Finish Position Std Dev",
            "Average Lap Time",
            "Average Points"
        ],
        "Value": [
            driver_info["avg_finish"],
            driver_info["finish_std"],
            driver_info["avg_laptime"],
            driver_info["avg_points"]
        ]
    })

    st.dataframe(stats_df, use_container_width=True)

    # --------------------------------------------------
    # 5️⃣ Finish Position Trend
    # --------------------------------------------------
    st.subheader("Finish Position Over Time")
    st.divider()

    driver_history = df[df["driverId"] == driver_id]

    if driver_history.empty:
        st.warning("No historical race data available.")
        st.stop()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        driver_history["year"],
        driver_history["positionOrder"],
        marker="o"
    )

    ax.set_title(f"{driver_selected} Finish Positions Over Time")
    ax.set_xlabel("Season")
    ax.set_ylabel("Finish Position")
    ax.invert_yaxis()
    ax.grid(True)

    st.pyplot(fig)

    # --------------------------------------------------
    # 6️⃣ Optional: Average Lap Time Trend (Nice Bonus)
    # --------------------------------------------------
    if "avg_lap_time" in driver_history.columns:

        st.subheader("Average Lap Time Trend")
        st.divider()

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        ax2.plot(
            driver_history["year"],
            driver_history["avg_lap_time"],
            marker="o"
        )

        ax2.set_title(f"{driver_selected} Average Lap Time Over Time")
        ax2.set_xlabel("Season")
        ax2.set_ylabel("Lap Time (seconds)")
        ax2.grid(True)

        st.pyplot(fig2)

# =========================
# PODIUM PREDICTOR
# =========================

if page == "Podium Predictor":

    st.header("Podium Probability Predictor")

    # Filter valid drivers
    valid_driver_ids = df["driverId"].unique()

    driver_options = drivers_lookup[
        drivers_lookup["driverId"].isin(valid_driver_ids)
    ].sort_values("driver_name")

    driver_selected = st.selectbox(
        "Select Driver",
        driver_options["driver_name"]
    )

    driver_id = driver_options[
        driver_options["driver_name"] == driver_selected
    ]["driverId"].iloc[0]

    grid_position = st.number_input(
        "Grid Position",
        min_value=1,
        max_value=20,
        value=5
    )

    valid_circuits = df["circuitId"].unique()

    circuits_filtered = circuits_lookup[
        circuits_lookup["circuitId"].isin(valid_circuits)
    ].sort_values("circuit_name")

    circuit_selected_name = st.selectbox(
        "Select Circuit",
        circuits_filtered["circuit_name"]
    )

    circuit_id = circuits_filtered[
        circuits_filtered["circuit_name"] == circuit_selected_name
    ]["circuitId"].iloc[0]

    if st.button("Predict Podium Probability"):

        driver_history = df[df["driverId"] == driver_id]

        if driver_history.empty:
            st.error("No feature data available.")
            st.stop()

        driver_data = driver_history.iloc[-1:].copy()

        driver_data.loc[:, "grid"] = grid_position
        driver_data.loc[:, "circuitId"] = circuit_id

        # Safety check
        missing = set(podium_features) - set(driver_data.columns)

        if missing:
            st.error("Feature mismatch detected.")
            st.write("Missing:", missing)
            st.stop()

        X_input = driver_data[podium_features]

        probability = podium_model.predict_proba(X_input)[0][1]

        st.subheader("Podium Probability")
        st.divider()

        percentage = probability * 100

        if percentage > 70:
            st.success(f"{percentage:.1f}% chance of podium finish")
        elif percentage > 40:
            st.warning(f"{percentage:.1f}% chance of podium finish")
        else:
            st.error(f"{percentage:.1f}% chance of podium finish")

        st.progress(int(percentage))

# =========================
# Tyre Degradation & Pace Analysis
# =========================

if page == "Tyre Degradation & Pace Analysis":

    st.header("Tyre Degradation & Multi-Stint Pace Analysis")

    # --------------------------------------------------
    # Filter races that exist in lap data
    # --------------------------------------------------
    valid_race_ids = lap_df["raceId"].unique()

    races_filtered = races_lookup[
        races_lookup["raceId"].isin(valid_race_ids)
    ].sort_values("race_name")

    race_selected = st.selectbox(
        "Select Race",
        races_filtered["race_name"]
    )

    race_id = races_filtered[
        races_filtered["race_name"] == race_selected
    ]["raceId"].iloc[0]

    # --------------------------------------------------
    # Filter drivers in that race
    # --------------------------------------------------
    valid_driver_ids = lap_df[
        lap_df["raceId"] == race_id
    ]["driverId"].unique()

    drivers_filtered = drivers_lookup[
        drivers_lookup["driverId"].isin(valid_driver_ids)
    ].sort_values("driver_name")

    driver_selected = st.selectbox(
        "Select Driver",
        drivers_filtered["driver_name"]
    )

    driver_id = drivers_filtered[
        drivers_filtered["driver_name"] == driver_selected
    ]["driverId"].iloc[0]

    # --------------------------------------------------
    # Run Analysis
    # --------------------------------------------------
    if st.button("Run Tyre Analysis"):

        results = multi_stint_analysis(
            lap_df,
            race_id,
            driver_id
        )

        if results is None:
            st.error("No lap data available.")
            st.stop()

        st.subheader("Stint Metrics")
        st.dataframe(results["stint_metrics"], use_container_width=True)

        st.write("Estimated Pit Laps:", results["pit_laps"])

        st.subheader("Degradation Curve")

        fig, ax = plt.subplots(figsize=(10, 6))

        data = results["data"]

        for stint in data["stint"].unique():
         stint_data = data[data["stint"] == stint]
         ax.scatter(stint_data["lap"], stint_data["lap_time_s"])

        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time (seconds)")
        ax.set_title("Multi-Stint Tyre Degradation")
        ax.grid(True)

        st.pyplot(fig)
        

st.divider()
st.caption("Final Year Project | Machine Learning in Formula 1 Analytics | 2026")





