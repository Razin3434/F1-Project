import sys
import os
import json
import joblib
import pandas as pd


# -------------------------
# Validate arguments
# -------------------------
if len(sys.argv) != 4:
    print("Usage: python predict.py <target> <input_file> <output_file>")
    sys.exit(1)

target = sys.argv[1].lower()
input_path = sys.argv[2]
output_path = sys.argv[3]


# -------------------------
# Map target to model files
# -------------------------
model_map = {
    "constructor": {
        "model": "models/constructor_points_model.joblib",
        "features": "models/constructor_points_features.joblib"
    },
    "finish": {
        "model": "models/finishing_position_model.joblib",
        "features": "models/finishing_position_features.joblib"
    },
    "laptime": {
        "model": "models/avg_lap_time_model.joblib",
        "features": "models/avg_lap_time_features.joblib"
    }
}

if target not in model_map:
    raise ValueError("Target must be: constructor, finish, or laptime")


# -------------------------
# Load model + features
# -------------------------
model = joblib.load(model_map[target]["model"])
feature_list = joblib.load(model_map[target]["features"])


# -------------------------
# Load input file
# -------------------------
if input_path.endswith(".csv"):
    df = pd.read_csv(input_path)
elif input_path.endswith(".json"):
    df = pd.read_json(input_path)
else:
    raise ValueError("Input must be CSV or JSON")


# -------------------------
# Handle missing features
# -------------------------
def prepare_features(df, feature_list):
    df_copy = df.copy()

    # Add missing columns as 0
    for col in feature_list:
        if col not in df_copy.columns:
            print(f"Warning: Missing feature '{col}' — filled with 0")
            df_copy[col] = 0

    return df_copy[feature_list]


X = prepare_features(df, feature_list)


# -------------------------
# Make prediction
# -------------------------
predictions = model.predict(X)


# -------------------------
# Format output
# -------------------------
results = []

for i in range(len(df)):
    results.append({
        "target": target,
        "prediction": float(predictions[i])
    })


# -------------------------
# Save JSON output
# -------------------------
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Predictions saved to {output_path}")
