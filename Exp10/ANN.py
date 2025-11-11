# single_layer_ann.py
# Simple ANN (1 hidden layer) for Iris dataset classification

# -------------------------------
# Import libraries
# -------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
iris = load_iris()
X = iris.data      # features (sepal & petal length/width)
y = iris.target    # labels (0=setosa, 1=versicolor, 2=virginica)

print("Dataset shape:", X.shape)
print("Unique classes:", iris.target_names)

# Convert to DataFrame for display
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print("\nFirst 5 rows of dataset:\n", df.head())

# -------------------------------
# Split the dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Build and train ANN model
# -------------------------------
# A single hidden layer with 5 neurons, ReLU activation
model = MLPClassifier(hidden_layer_sizes=(5,), activation='relu',
                      solver='adam', max_iter=500, random_state=1)

print("\nTraining the single-layer ANN model...")
model.fit(X_train, y_train)
print("✅ Training complete.")

# -------------------------------
# Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)

print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# -------------------------------
# Visualize training loss curve
# -------------------------------
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve (Single-Layer ANN)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------
# Predict a single new sample
# -------------------------------
# Example: sepal length=5.1, sepal width=3.5, petal length=1.4, petal width=0.2
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print("\nPrediction for sample [5.1, 3.5, 1.4, 0.2]:", iris.target_names[prediction[0]])
