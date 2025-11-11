# knn_diabetes.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, classification_report,
    roc_auc_score, roc_curve
)
from mlxtend.plotting import plot_decision_regions
import missingno as msno

# Load dataset
data = pd.read_csv("diabetes.csv")
print("Data Loaded Successfully ✅")
print(data.head())

# Check missing values
print("\nMissing values before replacement:")
print(data.isnull().any())

# Describe data
print("\nStatistical summary:")
print(data.describe().T)

# Replace 0 with NaN for selected columns (fixed np.nan)
data_copy = data.copy(deep=True)
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data_copy[cols] = data_copy[cols].replace(0, np.nan)

# Show number of missing values
print("\nMissing values after replacement:")
print(data_copy.isnull().sum())

# Plot original data histogram
data.hist(figsize=(20, 20))
plt.suptitle("Original Data Distribution")
plt.show()

# Fill missing values with mean or median
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)

# Histogram after filling missing data
data_copy.hist(figsize=(20, 20))
plt.suptitle("Cleaned Data Distribution")
plt.show()

# Visualize missing data
msno.bar(data)
plt.title("Missing Values Visualization")
plt.show()

# Bar plot for outcome distribution
data.Outcome.value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

# Pairplot visualization
sns.pairplot(data_copy, hue='Outcome')
plt.show()

# Heatmap of correlation
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
plt.title("Correlation Heatmap - Original Data")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(data_copy.corr(), annot=True, cmap='RdYlGn')
plt.title("Correlation Heatmap - Cleaned Data")
plt.show()

# Standardization
sc_X = StandardScaler()
X = pd.DataFrame(
    sc_X.fit_transform(data_copy.drop(["Outcome"], axis=1)),
    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
)
y = data_copy['Outcome']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=42, stratify=y
)

# Find optimal K value
train_scores = []
test_scores = []

for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

max_test_score = max(test_scores)
best_k = [i+1 for i, v in enumerate(test_scores) if v == max_test_score]
print(f"\n✅ Max test score: {max_test_score*100:.2f}% at K = {best_k}")

# Plot train vs test accuracy
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 15), y=train_scores, marker='*', label='Train Score')
sns.lineplot(x=range(1, 15), y=test_scores, marker='o', label='Test Score')
plt.title("Train vs Test Accuracy for Different K values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Use best K = 11
knn = KNeighborsClassifier(11)
knn.fit(X_train, y_train)
print(f"\nKNN Test Accuracy: {knn.score(X_test, y_test)*100:.2f}%")

# Decision Region Plot
value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2,
                      filler_feature_values={i: value for i in range(2, 8)},
                      filler_feature_ranges={i: width for i in range(2, 8)},
                      X_highlight=X_test.values)
plt.title("KNN Decision Region (Diabetes Data)")
plt.show()

# Confusion Matrix
y_pred = knn.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Model evaluation function
def model_evaluation(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)
    results = pd.DataFrame([[model_name, acc, prec, rec, f1, f2]],
                           columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
    return results

results = model_evaluation(y_test, y_pred, "KNN")
print("\nModel Evaluation:\n", results)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
y_pred_proba = knn.predict_proba(X_test)[:, -1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN (n_neighbors=11)')
plt.legend()
plt.show()

# GridSearchCV to find best K
param_grid = {"n_neighbors": np.arange(1, 50)}
knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
knn_gs.fit(X, y)

print("\nBest Parameters from GridSearchCV:", knn_gs.best_params_)
print("Best Cross-Validation Score:", knn_gs.best_score_)
