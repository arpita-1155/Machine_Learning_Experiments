# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("iris.csv")   # Ensure your file is in the same directory
print("Dataset Shape:", df.shape)
print(df.head())

# Convert all column names to lowercase for consistency
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# Step 2: Encode Target Column
# -----------------------------
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# -----------------------------
# Step 3: Split Data into Features and Labels
# -----------------------------
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Step 4: Initialize Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# -----------------------------
# Step 5: Train and Evaluate Models
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec
    })

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Step 6: Display Results
# -----------------------------
results_df = pd.DataFrame(results)
print("\nComparison of Classification Algorithms:\n")
print(results_df)

# -----------------------------
# Step 7: Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars="Model"), x="Model", y="value", hue="variable")
plt.title("Comparison of Classification Algorithms using Accuracy, Precision & Recall")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()
