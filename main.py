from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================
# FASTAPI APP
# =============================
app = FastAPI(title="Loan Default Prediction API (FINAL)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# FILE
# =============================
DATASET_FILE = "loan_default.csv"
MODEL_FILE = "loan_default_model.pkl"

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv(DATASET_FILE)

# normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# rename common variants
df.rename(columns={
    "loan_amount": "loanamount",
    "credit_score": "creditscore"
}, inplace=True)

FEATURES = ["age", "income", "loanamount", "creditscore"]
TARGET = "default"

# =============================
# TRAIN / LOAD MODEL
# =============================
if not os.path.exists(MODEL_FILE):

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained successfully | Accuracy: {acc:.2f}")

    joblib.dump(model, MODEL_FILE)

else:
    model = joblib.load(MODEL_FILE)

# =============================
# INPUT SCHEMA
# =============================
class LoanInput(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float

# =============================
# ROUTES
# =============================
@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "Loan Default Prediction API is running (FINAL)"
    }

@app.post("/predict")
def predict_default(data: LoanInput):

    input_df = pd.DataFrame([{
        "age": data.Age,
        "income": data.Income,
        "loanamount": data.LoanAmount,
        "creditscore": data.CreditScore
    }])

    input_df = input_df.astype(float)

    proba = model.predict_proba(input_df)
    probability = float(proba[0][1])

    # keputusan default (threshold)
    prediction = 1 if probability >= 0.3 else 0

    # LEVEL RISIKO (INI YANG BARU)
    if probability < 0.2:
        risk_level = "LOW"
    elif probability < 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "default_prediction": prediction,
        "default_probability": round(probability, 4),
        "risk_level": risk_level,
        "threshold_used": 0.3
    }
