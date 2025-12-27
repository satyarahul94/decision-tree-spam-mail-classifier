# =====================================================
# Decision Tree Classification - FULL CODE
# (Using YOUR uploaded dataset)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

# -----------------------------------------------------
# 1️⃣ Load YOUR dataset
# -----------------------------------------------------
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

df = pd.read_csv("hard_spam_dataset_40000.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------------------------------
# 2️⃣ Handle missing values
# -----------------------------------------------------
for col in df.columns:
    if col != "is_spam":   # target column
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------------------------------
# 3️⃣ Remove duplicate rows
# -----------------------------------------------------
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"\nRemoved {before - after} duplicate rows")
print("Final shape:", df.shape)

# -----------------------------------------------------
# 4️⃣ Features (X) and Target (y)
# -----------------------------------------------------
X = df.drop("is_spam", axis=1)
y = df["is_spam"]

print("\nTarget distribution:")
print(y.value_counts())

# -----------------------------------------------------
# 5️⃣ Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -----------------------------------------------------
# 6️⃣ Train Decision Tree Model
# -----------------------------------------------------
dt_model = DecisionTreeClassifier(
    criterion="gini",   # or "entropy"
    max_depth=6,        # prevents overfitting
    random_state=42
)

dt_model.fit(X_train, y_train)

# -----------------------------------------------------
# 7️⃣ Predictions
# -----------------------------------------------------
y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

# -----------------------------------------------------
# 8️⃣ Evaluation Metrics
# -----------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL METRICS (Decision Tree)")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-score  : {f1:.2f}")
print(f"ROC-AUC   : {roc_auc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 9️⃣ Confusion Matrix
# -----------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 🔟 ROC Curve
# -----------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 1️⃣1️⃣ Feature Importance
# -----------------------------------------------------
importances = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 1️⃣2️⃣ Decision Tree Visualization
# -----------------------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Class 0", "Class 1"],
    filled=True,
    max_depth=3
)
plt.title("Decision Tree Visualization (Top Levels)")
plt.show()

# -----------------------------------------------------
# 1️⃣3️⃣ User Input Prediction
# -----------------------------------------------------
print("\n--- User Input Prediction (Decision Tree) ---")

try:
    user_input = {}

    for col in X.columns:
        user_input[col] = float(input(f"Enter value for {col}: "))

    user_df = pd.DataFrame([user_input])

    user_prob = dt_model.predict_proba(user_df)[0][1]
    user_class = dt_model.predict(user_df)[0]

    result='spam' if user_class == 1 else 'not spam'

    print("\n✅ Prediction Result")
    print(f"Predicted Class:{result}")
    print(f"Probability of Class 1: {user_prob:.2f}")

except Exception as e:
    print("Error:", e)
