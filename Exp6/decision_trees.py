# -------------------------------------------------------------
# Decision Tree Classification on Diabetes Dataset (Visualized)
# -------------------------------------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------
df = pd.read_csv("diabetes.csv")  # make sure diabetes.csv is in same folder
print("✅ Dataset loaded successfully!")
print(df.head())

# -------------------------------------------------------------
# Feature and Target selection
# -------------------------------------------------------------
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# -------------------------------------------------------------
# Split dataset
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# -------------------------------------------------------------
# Train Decision Tree (Gini)
# -------------------------------------------------------------
model_gini = DecisionTreeClassifier(criterion="gini", random_state=1)
model_gini.fit(X_train, y_train)
y_pred_gini = model_gini.predict(X_test)

# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------
print("\n📊 Accuracy (Gini):", round(metrics.accuracy_score(y_test, y_pred_gini)*100, 2), "%")
print("\nConfusion Matrix (Gini):\n", confusion_matrix(y_test, y_pred_gini))
print("\nClassification Report (Gini):\n", classification_report(y_test, y_pred_gini))

# -------------------------------------------------------------
# Prediction for a sample patient
# -------------------------------------------------------------
sample = [[6,148,72,35,0,33.6,0.627,50]]
prediction = model_gini.predict(sample)
print("\nPrediction for sample patient (1 = Diabetic, 0 = Non-Diabetic):", prediction[0])

# -------------------------------------------------------------
# Visualize Decision Tree (Gini) directly using matplotlib
# -------------------------------------------------------------
plt.figure(figsize=(20,10))
plot_tree(
    model_gini,
    filled=True,
    rounded=True,
    feature_names=X.columns,
    class_names=['Non-Diabetic','Diabetic']
)
plt.title("Decision Tree Visualization (Gini)")
plt.show()

# -------------------------------------------------------------
# Train Decision Tree (Entropy)
# -------------------------------------------------------------
model_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)

print("\n📊 Accuracy (Entropy):", round(metrics.accuracy_score(y_test, y_pred_entropy)*100, 2), "%")
print("\nConfusion Matrix (Entropy):\n", confusion_matrix(y_test, y_pred_entropy))
print("\nClassification Report (Entropy):\n", classification_report(y_test, y_pred_entropy))

# -------------------------------------------------------------
# Visualize Decision Tree (Entropy)
# -------------------------------------------------------------
plt.figure(figsize=(15,8))
plot_tree(
    model_entropy,
    filled=True,
    rounded=True,
    feature_names=X.columns,
    class_names=['Non-Diabetic','Diabetic']
)
plt.title("Decision Tree Visualization (Entropy)")
plt.show()
