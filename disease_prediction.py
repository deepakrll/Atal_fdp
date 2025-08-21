# Activity 2: Predicting Disease Risk from Health Data with Visualization

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Step 2: Upload Dataset
from google.colab import files
uploaded = files.upload()  # Upload "health_data.csv"

# Step 3: Load Data
df = pd.read_csv("health_data.csv")
print("Dataset Preview:")
print(df.head())

# ==============================
# 📊 VISUALIZATION 1: Target Distribution
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x="HeartDisease", data=df, palette="coolwarm")
plt.title("Heart Disease Distribution (0 = Healthy, 1 = Disease)")
plt.show()

# ==============================
# 📊 VISUALIZATION 2: Feature Correlation Heatmap
# ==============================
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# Step 4: Split Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 📊 VISUALIZATION 3: ROC Curve
# ==============================
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

# Step 8: Predict Risk for New Patients
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

# ==============================
# 📊 VISUALIZATION 4: New Patient Prediction Risk
# ==============================
plt.figure(figsize=(6,4))
sns.barplot(x=[f"Patient {i+1}" for i in range(len(probs))],
            y=probs, palette="viridis")
plt.ylim(0,1)
plt.ylabel("Predicted Probability of Disease")
plt.title("Disease Risk Prediction for New Patients")
plt.show()

# Show results in text
for i, risk in enumerate(predictions):
    print(f"Patient {i+1}: Predicted Risk = {risk} (Probability: {probs[i]:.2f})")
