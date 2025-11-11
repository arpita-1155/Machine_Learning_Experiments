# compare_regression_algorithms.py

# -----------------------------
# Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("USA_Housing.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Check and Clean Data
# -----------------------------
print("\nDataset Info:")
print(df.info())

# Remove non-numeric columns if any (like address)
if 'Address' in df.columns:
    df = df.drop('Address', axis=1)

# -----------------------------
# Split Dataset into X and y
# -----------------------------
X = df.drop('Price', axis=1)
y = df['Price']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale Data (for SVR and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Initialize Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf'),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

# -----------------------------
# Train & Evaluate Models
# -----------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ["Support Vector Regressor", "KNN Regressor"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, mae, mse, rmse, r2])
    print(f"{name} completed!")

# -----------------------------
# Compare Results
# -----------------------------
results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"])
print("\n=== Model Comparison Results ===")
print(results_df)

# -----------------------------
# Visualize Results
# -----------------------------
plt.figure(figsize=(10,6))
sns.barplot(x="R2 Score", y="Model", data=results_df, palette="viridis")
plt.title("Comparison of Regression Algorithms")
plt.xlabel("R² Score (Higher = Better)")
plt.ylabel("Regression Models")
plt.grid(True)
plt.show()
