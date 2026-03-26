import pandas as pd
from sklearn.linear_model import LinearRegression


def multi_stint_analysis(lap_df, race_id, driver_id):

    # Filter race + driver
    data = lap_df[
        (lap_df["raceId"] == race_id) &
        (lap_df["driverId"] == driver_id)
    ].copy()

    if data.empty:
        return None

    data = data.sort_values("lap").reset_index(drop=True)

    # Detect pit stops
    data["lap_diff"] = data["lap_time_s"].diff()
    pit_threshold = 12
    data["pit_stop"] = data["lap_diff"] > pit_threshold

    pit_laps = data[data["pit_stop"]]["lap"].tolist()

    # Assign stints
    data["stint"] = 1
    stint_counter = 1

    for i in range(len(data)):
        data.loc[i, "stint"] = stint_counter
        if data.loc[i, "pit_stop"]:
            stint_counter += 1

    stint_results = []

    # Analyse each stint
    for stint in data["stint"].unique():

        stint_data = data[data["stint"] == stint].copy()

        if len(stint_data) < 5:
            continue

        # Remove internal outliers
        q_low = stint_data["lap_time_s"].quantile(0.05)
        q_high = stint_data["lap_time_s"].quantile(0.95)

        stint_data = stint_data[
            (stint_data["lap_time_s"] > q_low) &
            (stint_data["lap_time_s"] < q_high)
        ]

        if len(stint_data) < 5:
            continue

        X = stint_data[["lap"]]
        y = stint_data["lap_time_s"]

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r2 = model.score(X, y)
        std_dev = y.std()

        stint_results.append({
            "stint": int(stint),
            "laps": len(stint_data),
            "degradation_slope_sec_per_lap": slope,
            "r2_fit": r2,
            "consistency_index": 1 / std_dev
        })

    return {
        "data": data,
        "stint_metrics": pd.DataFrame(stint_results),
        "pit_laps": pit_laps
    }