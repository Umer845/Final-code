
# ------------------ Risk Score ------------------
def calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age):
    """
    Estimate risk score based on normalized factors instead of a fixed point system.
    Returns (risk_percent, risk_label).
    """

    # Normalize values to a 0-1 scale
    age_factor = min(vehicle_age / 20, 1)            # Older vehicles more risky
    sum_factor = min(sum_insured / 10_000_000, 1)    # Higher sum insured more risky
    driver_factor = 1 if driver_age < 25 or driver_age > 70 else 0.5
    use_factor = 1 if vehicle_use == "commercial" else (0.7 if vehicle_use == "other" else 0.4)

    # Weighted risk calculation
    risk_percent = (
        (age_factor * 0.25) +
        (sum_factor * 0.25) +
        (driver_factor * 0.25) +
        (use_factor * 0.25)
    ) * 100

    # Risk label
    if risk_percent >= 70:
        risk_label = "High Risk"
    elif risk_percent >= 40:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    return risk_percent, risk_label






# utils.py
import joblib
import pandas as pd
import os

# ----------------- Load Model & Features -----------------
MODEL_PATH = os.path.join("models", "catboost_premium_model_2.pkl")
FEATURES_PATH = os.path.join("models", "model_features_2.pkl")
CAT_FEATURES_PATH = os.path.join("models", "model_cat_features_2.pkl")

try:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    feature_cols = joblib.load(FEATURES_PATH)
    categorical_cols = joblib.load(CAT_FEATURES_PATH)

except Exception as e:
    model = None
    feature_cols = None
    categorical_cols = None
    print(f"⚠️ Error loading model or features: {e}")


# ----------------- Premium Calculation -----------------
def calculate(vehicle_make, vehicle_model, vehicle_year, sum_insured):
    """
    Predict premium value and premium rate using CatBoost model.
    Returns (premium_value, premium_rate).
    """

    if model is None or feature_cols is None:
        raise ValueError("Premium model or features not loaded properly.")

    # Derive vehicle age
    vehicle_age = 2025 - vehicle_year

    # Create input dataframe
    input_data = pd.DataFrame([{
        "vehicle_make": vehicle_make,
        "vehicle_model": vehicle_model,
        "vehicle_year": vehicle_year,
        "vehicle_age": vehicle_age,
        "sum_insured": sum_insured
    }])

    # Align features with training
    input_encoded = input_data.reindex(columns=feature_cols, fill_value=0)

    # Predict premium
    premium_value = model.predict(input_encoded)[0]

    # Premium rate (%) = premium ÷ sum insured × 100
    premium_rate = (premium_value / sum_insured) * 100 if sum_insured > 0 else 0

    return premium_value, premium_rate






