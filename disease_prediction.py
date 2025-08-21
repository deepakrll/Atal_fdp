# ===========================================
# Activity 2: Predicting Heart Disease Risk from Health Data
# Tool: Google Colab (Logistic Regression + Visualizations)
# ===========================================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Step 2: Upload Dataset
from google.colab import files
uploaded = files.upload()  # Upload health_data_large.csv (200 patients)

# Step 3: Load Data
df = pd.read_csv("health_data_large.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# ===========================================
# 🖼 Visualization 1: Heart Disease Distribution
# ===========================================
# This shows how many patients are healthy (0) vs. have heart disease (1).
# Helps us see if the dataset is balanced or skewed.

plt.figure(figsize=(6,4))
sns.countplot(x="HeartDisease", data=df, palette="coolwarm")
plt.title("Heart Disease Distribution (0 = Healthy, 1 = Disease)")
plt.xlabel("Heart Disease Risk")
plt.ylabel("Number of Patients")
plt.show()

# ===========================================
# 🖼 Visualization 2: Correlation Heatmap
# ===========================================
# Correlation heatmap tells us which factors (Age, Cholesterol, BP, Smoking, Diabetes)
# are most related to heart disease.
# Darker colors = stronger impact.

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap of Features vs. Heart Disease")
plt.show()

# ===========================================
# Step 4: Split Data & Train Model
# ===========================================
# We split the dataset into training & testing parts
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===========================================
# 🖼 Visualization 3: ROC Curve (Model Performance)
# ===========================================
# ROC curve shows how well the model distinguishes Healthy vs Disease.
# AUC closer to 1.0 means better performance.

y_probs = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Prediction")
plt.legend(loc="lower right")
plt.show()

# ===========================================
# 🧑‍⚕️ Step 5: Predict Risk for New Patients
# ===========================================
# Let's test the model on NEW patients to see who is prone to heart disease.

new_patients = pd.DataFrame({
    "Age": [40, 60],
    "Sex": [1, 0],
    "Cholesterol": [190, 250],
    "BloodPressure": [120, 150],
    "Smoking": [0, 1],
    "Diabetes": [0, 1]
})

print("\nNew Patients Data:")
print(new_patients)

predictions = model.predict(new_patients)
probs = model.predict_proba(new_patients)[:, 1]

# Print predictions in plain English
for i, risk in enumerate(predictions):
    status = "⚠️ High Risk of Heart Disease" if risk == 1 else "✅ Likely Healthy"
    print(f"Patient {i+1}: {status} (Probability: {probs[i]:.2f})")

# ===========================================
# 🖼 Visualization 4: Prediction Bar Chart (New Patients)
# ===========================================
# This bar chart shows the predicted risk probability for each new patient.
# Taller bar = Higher risk of heart disease.

plt.figure(figsize=(6,4))
sns.barplot(x=[f"Patient {i+1}" for i in range(len(probs))],
            y=probs, palette="viridis")
plt.ylim(0,1)
plt.ylabel("Predicted Probability of Heart Disease")
plt.title("Disease Risk Prediction for New Patients")
plt.show()

# ===========================================
# 🖼 (Optional Extra) Scatter Plot: Age vs. Cholesterol
# ===========================================
# This scatter plot shows patients by Age & Cholesterol.
# Red = High risk (1), Blue = Healthy (0).
# Participants can visually see how risky patients cluster together.

plt.figure(figsize=(7,5))
sns.scatterplot(x="Age", y="Cholesterol", hue="HeartDisease",
                data=df, palette={0:"blue", 1:"red"}, alpha=0.7)
plt.title("Scatter Plot: Age vs Cholesterol by Heart Disease Risk")
plt.xlabel("Age")
plt.ylabel("Cholesterol Level")
plt.show()
