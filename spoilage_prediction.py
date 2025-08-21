import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Generate dataset with ~15% spoilage risk
np.random.seed(42)
dates = pd.date_range(start="2025-08-01", periods=168, freq="H")

temperature = np.random.normal(loc=28, scale=4, size=168).round(1)
humidity = np.random.normal(loc=65, scale=12, size=168).round(1)

# Introduce more high-risk values artificially (force ~15% to be high risk)
for i in np.random.choice(range(168), size=int(0.15*168), replace=False):
    temperature[i] = np.random.uniform(31, 36)
    humidity[i] = np.random.uniform(76, 90)

df = pd.DataFrame({
    "Date": dates.date,
    "Time": dates.time,
    "Temperature (°C)": temperature,
    "Humidity (%)": humidity
})

# Define spoilage condition
df["Spoilage Risk"] = ((df["Temperature (°C)"] > 30) & (df["Humidity (%)"] > 75)).map({True: "High", False: "Low"})

# Logistic regression model
X = df[["Temperature (°C)", "Humidity (%)"]]
y = (df["Spoilage Risk"] == "High").astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
df["Spoilage Risk Probability (%)"] = model.predict_proba(X_scaled)[:, 1] * 100

# Visualization 1: Line chart with spoilage markers
plt.figure(figsize=(14,6))
plt.plot(df.index, df["Temperature (°C)"], label="Temperature (°C)", color="red")
plt.plot(df.index, df["Humidity (%)"], label="Humidity (%)", color="blue")
high_risk_idx = df[df["Spoilage Risk"]=="High"].index
plt.scatter(high_risk_idx, df.loc[high_risk_idx, "Temperature (°C)"], 
            color="darkred", label="High Spoilage Risk", s=60)
plt.title("Spoilage Risk Prediction Based on Temperature & Humidity")
plt.xlabel("Hourly Readings")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Scatter plot Temperature vs Humidity with spoilage zones
plt.figure(figsize=(8,6))
colors = df["Spoilage Risk"].map({"Low":"green", "High":"red"})
plt.scatter(df["Temperature (°C)"], df["Humidity (%)"], c=colors, alpha=0.7)
plt.axvline(30, color="black", linestyle="--", label="Temp Threshold (30°C)")
plt.axhline(75, color="gray", linestyle="--", label="Humidity Threshold (75%)")
plt.title("Temperature vs Humidity Scatter Plot with Spoilage Risk Zones")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)
plt.show()

# Summary
risk_summary = df["Spoilage Risk"].value_counts(normalize=True) * 100
risk_summary.round(2)
