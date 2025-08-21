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
uploaded = files.upload()  # Upload health_data_large.csv

# Step 3: Load Data
df = pd.read_csv("health_data_large.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())
