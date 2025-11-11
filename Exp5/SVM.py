# ============================================
# Support Vector Machine (SVM) Classification
# ============================================

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ============================================
# Step 1: Load the dataset
# ============================================
datasets = pd.read_csv('Social_Network_Ads.csv')
print("\nFirst 5 rows of the dataset:")
print(datasets.head())

# Select features and target
X = datasets.iloc[:, [2, 3]].values   # Age, Estimated Salary
Y = datasets.iloc[:, 4].values        # Purchased (0 or 1)

# ============================================
# Step 2: Split into Training and Test sets
# ============================================
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

# ============================================
# Step 3: Feature Scaling
# ============================================
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# ============================================
# Step 4: Train the SVM Classifier
# ============================================
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_Train, Y_Train)
print("\nModel training completed successfully!")

# ============================================
# Step 5: Predict the Test results
# ============================================
Y_Pred = classifier.predict(X_Test)

# Evaluate model performance
print("\nConfusion Matrix:")
cm = confusion_matrix(Y_Test, Y_Pred)
print(cm)

print("\nClassification Report:")
print(classification_report(Y_Test, Y_Pred))

print("Accuracy:", round(accuracy_score(Y_Test, Y_Pred) * 100, 2), "%")

# ============================================
# Step 6: Visualize the Training Set Results
# ============================================
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(
    np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01)
)

plt.figure(figsize=(8,6))
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(
        X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
        c=ListedColormap(('red', 'green'))(i), label=j
    )
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# ============================================
# Step 7: Visualize the Test Set Results
# ============================================
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(
    np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01)
)

plt.figure(figsize=(8,6))
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(
        X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
        c=ListedColormap(('red', 'green'))(i), label=j
    )
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
