# =========================================
# Activity 2: Flood Risk Prediction (India)
# Tool: Google Colab (Python + Logistic Regression)
# =========================================

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Step 3: Load dataset
csv_path = "/content/drive/MyDrive/AI_FloodRisk/india_flood_risk.csv"  # update if needed
df = pd.read_csv(csv_path)

print("Dataset Shape:", df.shape)
print(df.head())

# Step 4: Features & Target
drop_cols = [col for col in df.columns if "ID" in col or "Id" in col]
if "Date" in df.columns:
    drop_cols.append("Date")

X = df.drop(["Flood"] + drop_cols, axis=1)
y = df["Flood"]

# Categorical and numeric features
cat_features = ["State", "Region", "River_Basin", "Land_Use", "Month"]
num_features = [col for col in X.columns if col not in cat_features]

print("Categorical features:", cat_features)
print("Numeric features:", num_features)

# Step 5: Preprocessing + Model
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Train model
model.fit(X_train, y_train)

# ================================
# Visualizations
# ================================

# 1. Flood Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Flood", data=df, palette="coolwarm")
plt.title("Flood Distribution")
plt.xticks([0,1], ["No Flood", "Flood"])
plt.show()

# 2. Rainfall vs Flood
plt.figure(figsize=(7,5))
sns.boxplot(x="Flood", y="Rainfall_mm", data=df, palette="Set2")
plt.title("Rainfall vs Flood Occurrence")
plt.xticks([0,1], ["No Flood", "Flood"])
plt.show()

# 3. River Level vs Flood
plt.figure(figsize=(7,5))
sns.boxplot(x="Flood", y="River_Level_m", data=df, palette="Set1")
plt.title("River Level vs Flood Occurrence")
plt.xticks([0,1], ["No Flood", "Flood"])
plt.show()

# 4. State-wise Flood Distribution
plt.figure(figsize=(10,6))
sns.countplot(y="State", hue="Flood", data=df,
              order=df["State"].value_counts().index, palette="coolwarm")
plt.title("State-wise Flood Occurrence")
plt.legend(["No Flood", "Flood"])
plt.show()

# ================================
# Scenario Simulator
# ================================
def simulate_flood(rainfall, river_level, soil_moisture, release, distance, slope, elevation,
                   drainage, month, state, region, basin, land_use):
    sample = pd.DataFrame([{
        "Rainfall_mm": rainfall,
        "River_Level_m": river_level,
        "Soil_Moisture": soil_moisture,
        "Upstream_Release_cumecs": release,
        "Distance_to_River_km": distance,
        "Catchment_Slope_deg": slope,
        "Elevation_m": elevation,
        "Drainage_Density_km_per_km2": drainage,
        "Month": month,
        "State": state,
        "Region": region,
        "River_Basin": basin,
        "Land_Use": land_use
    }])
    
    prob = model.predict_proba(sample)[:, 1][0]
    print(f"Flood Probability = {prob:.2%}")

# Example Scenarios
print("\n--- Scenario 1 ---")
simulate_flood(
    rainfall=300, river_level=8.5, soil_moisture=0.7, release=1500,
    distance=2, slope=5, elevation=50, drainage=2.0,
    month="July", state="Assam", region="East", basin="Brahmaputra", land_use="Urban"
)

print("\n--- Scenario 2 ---")
simulate_flood(
    rainfall=120, river_level=3.5, soil_moisture=0.3, release=200,
    distance=10, slope=8, elevation=250, drainage=1.2,
    month="January", state="Rajasthan", region="West", basin="Indus", land_use="Agriculture"
)

# ================================
# Interactive Flood Risk Map (Plotly)
# ================================
import plotly.express as px
import requests, json

# Load India states GeoJSON (from GitHub)
india_states_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
geojson_data = requests.get(india_states_url).json()

# Prepare state-wise flood data
state_floods = df.groupby("State")["Flood"].mean().reset_index()
state_floods.rename(columns={"Flood":"Flood_Risk"}, inplace=True)

# Choropleth map
fig = px.choropleth(
    state_floods,
    geojson=geojson_data,
    locations="State",
    featureidkey="properties.NAME_1",  # State names key in geojson
    color="Flood_Risk",
    color_continuous_scale="Blues",
    range_color=(0, 1),
    title="Flood Risk Heatmap Across Indian States"
)

fig.update_geos(fitbounds="locations", visible=False)
fig.show()
