# ===========================================
# EXPERIMENT 9: K-MEANS CLUSTERING
# ===========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# STEP 1: Load the dataset
# ------------------------------------------------------------
data = pd.read_csv("iris.csv")   # Ensure iris.csv is in the same folder
print("Dataset Loaded Successfully!\n")
print(data.head())

# ------------------------------------------------------------
# STEP 2: Data Preprocessing
# ------------------------------------------------------------
# Extracting features (sepal_length, sepal_width, petal_length, petal_width)
X = data.iloc[:, :-1].values

# Encode the target labels (for reference only, not used in KMeans)
le = LabelEncoder()
y = le.fit_transform(data['species'])

# ------------------------------------------------------------
# STEP 3: Determine the optimal number of clusters using the Elbow Method
# ------------------------------------------------------------
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# STEP 4: Apply K-Means with optimal number of clusters (K=3 for Iris)
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# ------------------------------------------------------------
# STEP 5: Add Cluster info to dataset
# ------------------------------------------------------------
data['Cluster'] = y_kmeans
print("\nCluster assignments:\n")
print(data.head())

# ------------------------------------------------------------
# STEP 6: Visualize Clusters
# ------------------------------------------------------------
plt.figure(figsize=(8,6))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=80, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=80, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=80, c='green', label='Cluster 3')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', marker='*', label='Centroids')

plt.title('K-Means Clustering (Iris Dataset)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# ------------------------------------------------------------
# STEP 7: Evaluate using Silhouette Score
# ------------------------------------------------------------
silhouette_avg = silhouette_score(X, y_kmeans)
print(f"\nSilhouette Score for K=3: {silhouette_avg:.3f}")

# ------------------------------------------------------------
# STEP 8: Summary
# ------------------------------------------------------------
print("\nSummary:")
print("→ Optimal number of clusters: 3")
print("→ Clustering successfully performed using K-Means.")
print("→ Visualization shows clear cluster separations.")
print("→ Silhouette score closer to 1 indicates better clustering quality.")
