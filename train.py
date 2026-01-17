import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# =========================
# LOAD DATA
# =========================
data = {
    "gdp_per_capita": [
        167640, 90176, 47691, 47334, 68357, 57713, 45753, 36518,
        72619, 57051, 127285, 110450, 60763, 55061, 73297, 60391,
        49206, 110506, 96138, 159385, 44671, 80027, 58029, 50474,
        54285, 67545, 34059, 49455, 56445, 54684, 58557
    ],
    "urbanization_rate": [
        86.50, 83.15, 56.43, 58.41, 62.71, 68.10, 56.98, 59.55,
        61.18, 51.71, 69.61, 68.90, 54.69, 56.00, 60.30, 56.02,
        47.52, 65.82, 70.70, 88.13, 50.22, 65.50, 52.29, 47.69,
        31.14, 58.13, 47.69, 54.47, 58.88, 50.91, 59.06
    ]
}

df = pd.DataFrame(data)

# =========================
# FEATURES & TARGET
# =========================
X = df[["gdp_per_capita"]]   # HARUS 2D
y = df["urbanization_rate"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL (RANDOM FOREST)
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully using Random Forest")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# =========================
# TEST PREDICTION
# =========================
example_gdp = 150000
prediction = model.predict(
    pd.DataFrame([[example_gdp]], columns=["gdp_per_capita"])
)

print(f"Prediction for GDP {example_gdp}: {prediction[0]:.6f}")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "urbanization_growth_model.pkl")

print("Model saved as urbanization_growth_model.pkl")
